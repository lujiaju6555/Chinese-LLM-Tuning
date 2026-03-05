import json
import time
from openai import OpenAI
from tqdm import tqdm
import os
import re
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="大模型评估脚本")
    
    # API 配置
    parser.add_argument("--api_key", type=str, default="sk-075f638c1cbc4d2b91c04d38d04ccfdb", help="DASHSCOPE_API_KEY")
    parser.add_argument("--judge_model", type=str, default="qwen3.5-flash", help="用于评估的大模型名称")
    
    # 目标模型配置
    parser.add_argument("--target", type=str, default="dpo", choices=["baseline", "sft", "dpo"], help="目标评估模型")
    parser.add_argument("--sft_model_path", type=str, default="./models/sft", help="SFT 模型路径")
    parser.add_argument("--baseline_model_path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="基础模型路径")
    parser.add_argument("--eval_data_path", type=str, default="./data/belle_eval.json", help="用于评估的BELLE数据路径")

    # 文件路径配置
    parser.add_argument("--input_file", type=str, help="输入文件路径")
    parser.add_argument("--output_path", type=str, help="输出结果路径")
    parser.add_argument("--statistics_path", type=str, help="统计结果保存路径")
    
    # 评估配置
    parser.add_argument("--delay", type=float, default=1.0, help="请求间隔（秒）")
    parser.add_argument("--max_workers", type=int, default=5, help="最大并行工作线程数")
    
    return parser.parse_args()


# 路径配置
BASELINE_PATH = {
        'input_file': "./results/baseline/eval_response.json",
        'output_path': "./results/baseline/eval_result.json",
        'statistics_path': "./results/baseline/score_statistics.json"
    }
SFT_PATH = {
    'input_file': "./results/sft/eval_response.json",
    'output_path': "./results/sft/eval_result.json",
    'statistics_path': "./results/sft/score_statistics.json"
}
DPO_PATH = {
    'input_file': "./results/dpo/eval_response.json",
    'output_path': "./results/dpo/eval_result.json",
    'statistics_path': "./results/dpo/score_statistics.json"
}

class Evaluator:
    def __init__(self, api_key=None, base_url=None, model=None):
        """
        初始化 Qwen 评估器

        Args:
            api_key: DASHSCOPE_API_KEY，若为 None 则从环境变量读取
            base_url: API 地址
            model: 模型名称
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model

    def create_evaluation_prompt(self, instruction, ground_truth, response):
        """
        构建评估提示词
        """
        prompt = f"""
你是一个专业的 AI 模型评估员，负责评估模型回答的质量。请根据以下标准对模型的回答进行评分（1-10分）：

评估标准：
1. **准确性**：回答是否准确反映了指令的要求
2. **完整性**：回答是否全面覆盖了问题要点
3. **流畅性**：语言表达是否自然流畅
4. **相关性**：回答是否与指令紧密相关

指令：{instruction}

参考答案：{ground_truth}

模型回答：{response}

请按照以下格式输出：

<evaluation>
[在这里给出 1-10 分的评分，并简要说明理由]
</evaluation>

<score>
[只输出数字分数：1-10]
</score>
        """.strip()
        return prompt

    def extract_score(self, text):
        """
        从文本中提取分数（1-10）
        """
        # 优先匹配 <score> 标签
        score_match = re.search(r'<score>\s*(\d+(?:\.\d+)?)\s*</score>', text)
        if score_match:
            score = float(score_match.group(1))
            return max(1, min(10, score))  # 限制在 1-10 范围

        # 备选：匹配独立数字
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
        for num_str in numbers:
            num = float(num_str)
            if 1 <= num <= 10:
                return num

        return 5.0  # 默认分数

    def evaluate_single(self, item):  # 修复：改为实例方法
        """
        使用 API 进行单个判断
        """
        instruction = item["instruction"]
        ground_truth = item.get("ground_truth", "")
        response = item.get("response", "")

        prompt = self.create_evaluation_prompt(instruction, ground_truth, response)

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"enable_thinking": True},
                max_tokens=1024,
                timeout=60
            )

            # 直接获取内容
            response_text = completion.choices[0].message.content

            # 提取分数
            score = self.extract_score(response_text)

            result = item.copy()
            result.update({
                "judge_response": response_text,
                "score": round(score, 2),
                "status": "success"
            })

            return result

        except Exception as e:
            result = item.copy()
            result.update({
                "judge_response": f"[ERROR: {str(e)}]",
                "score": 0.0,
                "status": f"error: {str(e)}"
            })
            return result

    def evaluate_batch(self, data_path, output_path, delay=1, max_workers=5):
        """
        批量评估

        Args:
            data_path: 输入数据路径
            output_path: 输出结果路径
            delay: 请求间隔（秒），避免频率限制
            max_workers: 最大并行工作线程数
        """
        # 检查文件是否存在
        if not os.path.exists(data_path):
            print(f"❌ 文件不存在: {data_path}")
            return []

        # 先检查 JSON 文件格式
        print(f"检查 JSON 格式: {data_path}")
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    print(f"❌ 文件为空: {data_path}")
                    return []

                # 尝试解析 JSON
                test_json = json.loads(content)
                print(f"✅ JSON 格式正确，共 {len(test_json)} 条数据")
        except json.JSONDecodeError as e:
            print(f"❌ JSON 格式错误: {e}")
            print("文件内容预览:")
            with open(data_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split('\n')
                for i, line in enumerate(lines[max(0, e.lineno - 3):e.lineno + 2],
                                         max(1, e.lineno - 2)):
                    marker = " >>> " if i == e.lineno else "     "
                    print(f"{marker}{i:3d}: {line}")
            return []
        except Exception as e:
            print(f"❌ 读取文件错误: {e}")
            return []

        # 加载数据
        with open(data_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        # 过滤出有效的数据项
        valid_items = []
        for i, item in enumerate(eval_data):
            if "response" not in item:
                print(f"Warning: Item {i} missing 'response' field, skipping")
                continue
            valid_items.append(item)

        results = []

        # 使用线程池并行评估
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(f"开始并行评估，最大工作线程数: {max_workers}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_item = {executor.submit(self.evaluate_single, item): item for item in valid_items}
            
            # 处理完成的任务
            for future in tqdm(as_completed(future_to_item), total=len(future_to_item), desc=f"Evaluating"):
                try:
                    result = future.result()
                    results.append(result)
                    # 控制请求频率
                    if delay > 0:
                        time.sleep(delay)
                except Exception as e:
                    print(f"  任务执行出错: {str(e)}")
                    import traceback
                    traceback.print_exc()

        # 保存结果
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 统计信息
        successful = [r for r in results if r["status"] == "success"]
        avg_score = sum(r["score"] for r in successful) / len(successful) if successful else 0

        print(f"\n✅ 评估完成！")
        print(f"总数: {len(results)}, 成功: {len(successful)}, 失败: {len(results) - len(successful)}")
        print(f"平均分数: {avg_score:.2f}")
        print(f"结果已保存至: {output_path}")

        return results


def calculate_statistics(input_path="./results/baseline/eval_with_scores.json",
                         output_path="./results/baseline/score_statistics.json"):
    """
    计算评估结果的统计信息

    Args:
        input_path: 评估结果文件路径
        output_path: 统计结果保存路径
    """
    # 检查文件是否存在
    if not os.path.exists(input_path):
        print(f"❌ 文件不存在: {input_path}")
        return None

    # 加载评估结果
    with open(input_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # 过滤成功的评估结果
    successful_results = [item for item in results if item["status"] == "success"]

    if not successful_results:
        print("❌ 没有成功的评估结果")
        return None

    # 计算各项统计指标
    scores = [item["score"] for item in successful_results]
    total_count = len(results)
    success_count = len(successful_results)
    failed_count = total_count - success_count
    avg_score = sum(scores) / len(scores) if scores else 0
    max_score = max(scores) if scores else 0
    min_score = min(scores) if scores else 0
    median_score = sorted(scores)[len(scores) // 2] if scores else 0

    # 统计分布
    score_ranges = {
        "9-10分": len([s for s in scores if 9 <= s <= 10]),
        "8-8.9分": len([s for s in scores if 8 <= s < 9]),
        "7-7.9分": len([s for s in scores if 7 <= s < 8]),
        "6-6.9分": len([s for s in scores if 6 <= s < 7]),
        "5分以下": len([s for s in scores if s < 5])
    }

    # 构建统计结果
    statistics = {
        "summary": {
            "total_items": total_count,
            "successful_evaluations": success_count,
            "failed_evaluations": failed_count,
            "success_rate": round(success_count / total_count * 100, 2) if total_count > 0 else 0
        },
        "scores": {
            "average_score": round(avg_score, 2),
            "max_score": max_score,
            "min_score": min_score,
            "median_score": median_score
        },
        "distribution": score_ranges,
        "details": {
            "all_scores": [round(s, 2) for s in scores],
            "score_variance": round(sum((s - avg_score) ** 2 for s in scores) / len(scores), 4) if scores else 0,
            "score_std_dev": round((sum((s - avg_score) ** 2 for s in scores) / len(scores)) ** 0.5, 4) if scores else 0
        }
    }

    # 保存统计结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)

    # 输出统计信息
    print("=" * 50)
    print("📊 评估结果统计报告")
    print("=" * 50)
    print(f"总项目数: {statistics['summary']['total_items']}")
    print(f"成功评估: {statistics['summary']['successful_evaluations']}")
    print(f"失败评估: {statistics['summary']['failed_evaluations']}")
    print(f"成功率: {statistics['summary']['success_rate']}%")
    print("-" * 30)
    print(f"🏆 总平均分: {statistics['scores']['average_score']}")
    print(f"📈 最高分: {statistics['scores']['max_score']}")
    print(f"📉 最低分: {statistics['scores']['min_score']}")
    print(f"📊 中位数: {statistics['scores']['median_score']}")
    print(f"🔍 方差: {statistics['details']['score_variance']}")
    print(f"📏 标准差: {statistics['details']['score_std_dev']}")
    print("-" * 30)
    print("分数分布:")
    for range_name, count in statistics['distribution'].items():
        percentage = round(count / success_count * 100 if success_count > 0 else 0, 2)
        print(f"  {range_name}: {count} 条 ({percentage}%)")
    print("=" * 50)
    print(f"📈 统计结果已保存至: {output_path}")

    return statistics


def load_sft_model(sft_model_path, baseline_model_path):
    """
    加载 SFT 模型

    Args:
        sft_model_path: SFT 模型路径
        baseline_model_path: 基础模型路径

    Returns:
        model: 加载好的模型
        tokenizer: 加载好的 tokenizer
    """
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)

    # 检查是否是已经合并的模型还是带 LoRA 的模型
    adapter_config_path = os.path.join(sft_model_path, "adapter_config.json")

    if os.path.exists(adapter_config_path):
        # 如果存在 adapter_config.json，说明是 LoRA 模型
        print("检测到 LoRA 模型，正在加载...")

        # 先加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            baseline_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        # 加载 LoRA 权重并合并
        model = PeftModel.from_pretrained(model, sft_model_path)
        model = model.merge_and_unload()
    else:
        # 如果没有 adapter_config.json，说明已经是合并后的模型
        print("检测到已合并模型，直接加载...")
        model = AutoModelForCausalLM.from_pretrained(
            sft_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    return model, tokenizer


def load_baseline_model(args):
    # 加载 tokenizer
    tokenizer_baseline = AutoTokenizer.from_pretrained(args.baseline_model_path, trust_remote_code=True)

    # 加载模型
    model_baseline = AutoModelForCausalLM.from_pretrained(
        args.baseline_model_path,
        device_map="auto",
        trust_remote_code=True
    )

    return model_baseline, tokenizer_baseline



def generate_responses(
        data_path,
        tokenizer,
        model,
        save_path,
        max_new_tokens=512,
        device="cuda"
):
    """
    使用给定的 tokenizer 和 model 对 BELLE 评估数据生成完整回答。

    Args:
        data_path (str): BELLE eval JSON 文件路径，格式为 [{"instruction": "...", "input": "...", "output": "..."}, ...]
        tokenizer: Hugging Face tokenizer（需支持 apply_chat_template）
        model: Hugging Face causal language model（已加载到 device）
        save_path (str): 生成结果保存路径（.json）
        max_new_tokens (int): 最大生成长度，默认 512（足够完整回答）
        device (str): 推理设备，默认 "cuda"
    """
    # 加载评估数据
    with open(data_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    results = []

    model.eval()  # 确保在 eval 模式
    with torch.no_grad():
        for item in tqdm(eval_data, desc="Generating responses"):
            # 构建 messages（BELLE 是单轮）
            messages = [
                {"role": "user",
                 "content": item["instruction"] + ("\n" + item["input"] if item["input"].strip() else "")}
            ]

            # 使用 tokenizer 的 chat template 构建 prompt
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True  # 添加 assistant 开始标记
            )

            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=2048  # 防止超长输入
            ).to(device)

            # 生成
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 贪心解码，确保可复现
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            # 解码生成部分（跳过 prompt）
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # 保存结果（保留原字段 + 新增）
            result_item = {
                "instruction": item["instruction"],
                "input": item["input"],
                "ground_truth": item.get("output", ""),  # 原始答案（可选）
                "response": response.strip()
            }
            results.append(result_item)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 保存结果
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 生成完成！共 {len(results)} 条，已保存至: {save_path}")


# 使用示例
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 如果待评估回答数据不存在，则使用目标模型生成待评估回答数据
    if not os.path.exists(args.input_file):

        # 加载模型
        if args.target == 'baseline':
            model, tokenizer = load_baseline_model(args)
        else:
            model, tokenizer = load_sft_model(args.sft_model_path, args.baseline_model_path)

        # 生成回答
        generate_responses(
            data_path=args.eval_data_path,
            tokenizer=tokenizer,
            model=model,
            save_path=args.input_file,
            max_new_tokens=2048
        )

    # 初始化评估器
    evaluator = Evaluator(api_key=args.api_key, model=args.judge_model)

    # 检查输入文件
    config = {}
    if args.target == 'baseline':
        config = BASELINE_PATH
    elif args.target == 'sft':
        config = SFT_PATH
    elif args.target == 'dpo':
        config = DPO_PATH
    
    # 使用命令行参数覆盖默认路径
    input_file = args.input_file or config['input_file']
    output_path = args.output_path or config['output_path']
    statistics_path = args.statistics_path or config['statistics_path']

    if os.path.exists(input_file):
        print(f"✅ 找到输入文件: {input_file}")
    else:
        print(f"❌ 输入文件不存在: {input_file}")
        # 创建一个示例文件用于测试
        sample_data = [
            {
                "instruction": "这是一个测试指令",
                "input": "",
                "ground_truth": "这是参考答案",
                "response": "这是模型回答"
            }
        ]
        os.makedirs(os.path.dirname(input_file), exist_ok=True)
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        print(f"已创建示例文件: {input_file}")

    # 批量评估
    if os.path.exists(output_path):
        print(f"⚠️ 评估结果文件已存在: {output_path}")
    else:
        evaluator.evaluate_batch(
            data_path=input_file,
            output_path=output_path,
            delay=args.delay,
            max_workers=args.max_workers
        )
    # 在主评估完成后运行统计
    stats = calculate_statistics(input_path=output_path, output_path=statistics_path)

    if stats:
        print(f"\n🎉 总平均分: {stats['scores']['average_score']}")

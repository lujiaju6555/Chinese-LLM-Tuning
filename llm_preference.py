import json
import time
from openai import OpenAI
from tqdm import tqdm
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="大模型偏好数据生成脚本")
    
    # 功能选择
    parser.add_argument("--mode", type=str, default="rank", choices=["generate", "rank", "both"], help="运行模式: generate(生成候选回答), rank(排序偏好), both(先生成后排序)")
    
    # API 配置
    parser.add_argument("--api_key", type=str, default="sk-2c3bde2ec4ea430f9b1daccaf20313e5", help="DASHSCOPE_API_KEY")
    parser.add_argument("--model", type=str, default="qwen-flash", help="评估模型名称")
    
    # SFT 模型配置
    parser.add_argument("--sft_model_path", type=str, default="./models/sft", help="SFT 模型路径")
    parser.add_argument("--baseline_model_path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="基础模型路径")
    
    # 文件路径配置
    parser.add_argument("--preference_data_path", type=str, default="./data/belle_eval.json", help="偏好数据路径")
    parser.add_argument("--candidates_path", type=str, default="./data/belle_preference_response.json", help="候选回答保存路径")
    parser.add_argument("--output_path", type=str, default="./data/preference_data.json", help="偏好数据输出路径")
    
    # 生成配置
    parser.add_argument("--num_samples", type=int, default=4, help="每条指令生成的候选回答数量")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度参数")
    parser.add_argument("--top_p", type=float, default=0.9, help="top_p 参数")
    
    # 评估配置
    parser.add_argument("--delay", type=float, default=0.5, help="请求间隔（秒）")
    parser.add_argument("--max_workers", type=int, default=5, help="最大并行工作线程数")
    
    return parser.parse_args()


class PreferenceRanker:
    def __init__(self, api_key=None, base_url=None, model="qwen-flash"):
        """
        初始化 Qwen 偏好排序器

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

    def rank_responses_with_qwen(self, instruction, responses):
        """
        对同一指令的多个模型回答进行偏好排序

        Args:
            instruction: 指令内容
            responses: 模型回答列表

        Returns:
            元组 (sorted_indices, response_text): 排序后的索引列表和 API 返回的原始响应
        """
        # 构造排序提示词
        prompt = f"""
你是一个专业的 AI 模型评估员，负责对多个模型回答进行偏好排序。

请根据以下标准对模型回答按质量从高到低排序：
1. **准确性**：回答是否准确反映了指令的要求
2. **完整性**：回答是否全面覆盖了问题要点
3. **流畅性**：语言表达是否自然流畅
4. **相关性**：回答是否与指令紧密相关

指令：{instruction}

回答列表：
"""
        
        # 添加每个回答
        for i, response in enumerate(responses, 1):
            prompt += f"{i}. {response}\n"
        
        # 添加输出格式要求
        prompt += """

**请严格按照以下格式输出，不要添加任何额外内容：**

排名：1 > 2 > 3 > 4

其中数字对应回答的序号，从最好到最差排列。

**重要要求：**
- 必须包含"排名："前缀
- 必须使用" > "作为分隔符
- 必须包含所有回答的序号
- 序号必须是1到{len(responses)}之间的整数
- 不能包含其他任何文字或解释
""".strip()
        
        try:
            # 调用 API
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                timeout=60
            )
            
            # 获取响应
            response_text = completion.choices[0].message.content
            # print(f"排序结果: {response_text}")
            
            # 解析排序结果
            # 查找 "排名：" 后面的内容
            rank_prefix = "排名："
            if rank_prefix in response_text:
                rank_part = response_text.split(rank_prefix)[1].strip()
                # 提取数字
                ranks = re.findall(r'\d+', rank_part)
                # 转换为索引（减1）
                sorted_indices = [int(r) - 1 for r in ranks if r.isdigit()]
                
                # 验证结果是否完整
                if len(sorted_indices) == len(responses) and all(0 <= i < len(responses) for i in sorted_indices):
                    return sorted_indices, response_text
                else:
                    print(f"解析结果不完整或无效: {sorted_indices}")
            else:
                print("未找到排名结果")
                
        except Exception as e:
            print(f"排序过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # 出错时返回 (None, 错误信息)
        return None, str(e) if 'e' in locals() else "Unknown error"

    def rank_responses_batch(self, rank_data, output_path, delay=0.5, max_workers=5):
        """
        批量对多个指令的模型回答进行偏好排序

        Args:
            rank_data: 排序数据列表，每个元素包含 "instruction" 和 "responses" 字段
            output_path: 输出结果路径
            delay: 请求间隔（秒），避免频率限制
            max_workers: 最大并行工作线程数

        Returns:
            排序结果列表
        """
        import hashlib
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 加载已完成的结果
        completed_results = {}
        if os.path.exists(output_path):
            print(f"加载已完成的结果: {output_path}")
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)
                    for result in existing_results:
                        # 生成唯一标识符
                        if "instruction" in result:
                            key = hashlib.md5(result["instruction"].encode()).hexdigest()
                            # 只有当 sorted_indices 不为空时才算有结果
                            if "sorted_indices" in result and result["sorted_indices"] is not None:
                                completed_results[key] = result
                print(f"已完成 {len(completed_results)} 条数据")
            except Exception as e:
                print(f"加载已完成结果时出错: {e}")
                completed_results = {}
        
        # 过滤出未完成的数据
        pending_data = []
        for item in rank_data:
            if "instruction" in item:
                key = hashlib.md5(item["instruction"].encode()).hexdigest()
                if key not in completed_results:
                    pending_data.append(item)

        print(f"待处理数据: {len(pending_data)} 条")
        
        if not pending_data:
            print("✅ 所有数据已处理完成！")
            return list(completed_results.values())

        # 开始处理未完成的数据
        print(f"开始并行排序，最大工作线程数: {max_workers}")

        # 使用线程池并行处理
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_item = {
                executor.submit(self.rank_responses_with_qwen, item["instruction"], item["responses"]): item for
                item in pending_data}

            # 处理完成的任务
            for future in tqdm(as_completed(future_to_item), total=len(future_to_item),
                               desc="Ranking responses"):
                try:
                    item = future_to_item[future]
                    sorted_indices, response_text = future.result()

                    # 生成唯一标识符
                    key = hashlib.md5(item["instruction"].encode()).hexdigest()

                    # 构建结果
                    result = item.copy()
                    result["sorted_indices"] = sorted_indices
                    result["api_response"] = response_text  # 保存 API 返回的结果

                    # 更新已完成结果
                    completed_results[key] = result

                    # 实时保存结果（流式写入）
                    # 先写入临时文件，再原子替换，确保文件完整性
                    temp_output_path = output_path + ".tmp"
                    with open(temp_output_path, "w", encoding="utf-8") as f:
                        json.dump(list(completed_results.values()), f, ensure_ascii=False, indent=2)
                    # 原子替换文件
                    os.replace(temp_output_path, output_path)

                    # 控制请求频率
                    if delay > 0:
                        time.sleep(delay)
                except Exception as e:
                    print(f"  任务执行出错: {str(e)}")
                    import traceback
                    traceback.print_exc()

        print(f"\n✅ 排序完成！")
        print(f"总数: {len(completed_results)}")
        print(f"结果已保存至: {output_path}")

        return list(completed_results.values())


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
        print("加载 SFT 模型的 LoRA 权重并合并...")
        model = PeftModel.from_pretrained(model, sft_model_path)
        model = model.merge_and_unload()
        print("SFT 模型 LoRA 权重合并完成")
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


def generate_multiple_responses(
        data_path,
        tokenizer,
        model,
        save_path,
        num_samples=4,  # 每条指令生成多少个回答
        max_new_tokens=2048,
        temperature=0.7,  # 控制多样性（>0 才有多样性）
        top_p=0.9,
        seed=42  # 可复现
):
    """
    构造偏好数据：对每条指令生成多个不同回答。

    输出格式：
    [
      {
        "instruction": "...",
        "input": "...",
        "ground_truth": "...",
        "responses": ["resp1", "resp2", "resp3", "resp4"]  # 多个回答
      },
      ...
    ]
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 加载数据
    with open(data_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    results = []
    model.eval()

    with torch.no_grad():
        for item in tqdm(eval_data, desc=f"Generating {num_samples} responses per prompt"):
            messages = [
                {"role": "user",
                 "content": item["instruction"] + ("\n" + item["input"] if item["input"].strip() else "")}
            ]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=2048
            ).to(model.device)

            responses = []
            for i in range(num_samples):
                # 关键：开启 do_sample 并设置 temperature/top_p
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,  # 必须为 True 才有多样性
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    # 防止重复
                    repetition_penalty=1.1
                )

                generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                responses.append(response.strip())

            result_item = {
                "instruction": item["instruction"],
                "input": item["input"],
                "ground_truth": item.get("output", ""),
                "responses": responses  # 注意这里是列表
            }
            results.append(result_item)

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 候选回答生成完成！共 {len(results)} 条指令，每条 {num_samples} 个回答")
    print(f"📁 已保存至: {save_path}")
    
    return results

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()
    
    # 生成候选回答
    if args.mode in ["generate", "both"]:
        print("\n=== 开始生成候选回答 ===")
        # 加载 SFT 模型
        model, tokenizer = load_sft_model(args.sft_model_path, args.baseline_model_path)
        
        # 生成候选回答
        generate_multiple_responses(
            data_path=args.preference_data_path,
            tokenizer=tokenizer,
            model=model,
            save_path=args.candidates_path,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        # 释放显存
        del model
        torch.cuda.empty_cache()
    
    # 排序偏好数据
    if args.mode in ["rank", "both"]:
        print("\n=== 开始排序偏好数据 ===")
        # 初始化排序器
        ranker = PreferenceRanker(api_key=args.api_key, model=args.model)
        
        # 从指定路径加载数据
        data_path = args.candidates_path
        output_path = args.output_path
        
        # 检查文件是否存在
        if not os.path.exists(data_path):
            print(f"❌ 文件不存在: {data_path}")
            exit(1)
        
        # 先检查 JSON 文件格式
        print(f"检查 JSON 格式: {data_path}")
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    print(f"❌ 文件为空: {data_path}")
                    exit(1)

                # 尝试解析 JSON
                rank_data = json.loads(content)
                print(f"✅ JSON 格式正确，共 {len(rank_data)} 条数据")
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
            exit(1)
        except Exception as e:
            print(f"❌ 读取文件错误: {e}")
            exit(1)
        
        # 过滤出有效的数据项
        valid_data = []
        for i, item in enumerate(rank_data):
            if "instruction" not in item or "responses" not in item:
                print(f"Warning: Item {i} missing 'instruction' or 'responses' field, skipping")
                continue
            if not isinstance(item["responses"], list) or len(item["responses"]) < 2:
                print(f"Warning: Item {i} has invalid 'responses' field, skipping")
                continue
            valid_data.append(item)
        
        print(f"过滤后有效数据: {len(valid_data)} 条")

        # 批量排序
        if valid_data:
            ranker.rank_responses_batch(
                rank_data=valid_data,
                output_path=output_path,
                delay=args.delay,
                max_workers=args.max_workers
            )
        else:
            print("❌ 没有有效的数据项")
            exit(1)
    
    print("\n✅ 任务完成！")
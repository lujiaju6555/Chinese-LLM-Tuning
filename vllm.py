import os
import argparse
from vllm import LLM, SamplingParams
import json


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="vLLM 推理脚本")

    # 模型配置
    parser.add_argument("--model_path", type=str, default="./models/dpo", help="模型路径")

    # 推理配置
    parser.add_argument("--data_path", type=str, default="./data/belle_eval.json", help="评估数据路径")
    parser.add_argument("--output_path", type=str, default="./results/dpo/eval_response.json", help="推理结果保存路径")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.0, help="温度参数")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p 参数")

    return parser.parse_args()


def vllm_inference(args):
    """
    使用 vLLM 进行推理
    """
    # 加载模型
    print(f"加载模型: {args.model_path}")
    llm = LLM(model=args.model_path, trust_remote_code=True)

    # 配置采样参数
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )

    # 加载评估数据
    print(f"加载评估数据: {args.data_path}")
    with open(args.data_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # 准备 prompts
    prompts = []
    for item in eval_data:
        instruction = item["instruction"]
        input_text = item["input"]
        prompt = instruction + ("\n" + input_text if input_text.strip() else "")
        # 构建聊天模板
        messages = [
            {"role": "user", "content": prompt}
        ]
        # 使用 vLLM 的默认聊天模板
        prompt = llm.get_tokenizer().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)

    # 批量推理
    print(f"开始推理，共 {len(prompts)} 条数据")
    outputs = llm.generate(prompts, sampling_params)

    # 处理结果
    results = []
    for i, output in enumerate(outputs):
        item = eval_data[i]
        generated_text = output.outputs[0].text.strip()
        result_item = {
            "instruction": item["instruction"],
            "input": item["input"],
            "ground_truth": item.get("output", ""),
            "response": generated_text
        }
        results.append(result_item)

    # 保存结果
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"推理完成，结果已保存到: {args.output_path}")


if __name__ == "__main__":
    args = parse_args()
    vllm_inference(args)
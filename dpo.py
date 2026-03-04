import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig
from utils import load_preference_data, build_dpo_dataset, train_dpo


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="DPO 模型训练脚本")

    # 模型配置
    parser.add_argument("--base_model", type=str, default="./models/sft", help="SFT 模型路径")
    parser.add_argument("--output_dir", type=str, default="./models/dpo", help="DPO 模型保存路径")

    # 数据配置
    parser.add_argument("--preference_data_path", type=str, default="./data/preference_data.json", help="偏好数据路径")
    parser.add_argument("--max_samples", type=int, default=5000, help="最大样本数")

    # 训练配置
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="每个设备的训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="学习率")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志步数")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta 参数")

    return parser.parse_args()


def main(args):
    """
    DPO 训练主流程
    """
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # 加载 SFT 模型
    print(f"加载 SFT 模型: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        trust_remote_code=True
    )

    # 加载并清洗偏好数据
    preference_data = load_preference_data(args.preference_data_path, args.max_samples)

    # 构建 DPO 数据集
    dpo_dataset = build_dpo_dataset(preference_data, tokenizer)

    # 配置 DPO 训练参数
    dpo_training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_strategy="no",

        # DPO 核心超参
        beta=args.beta,

        # 精度与性能
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,

        # 数据处理
        remove_unused_columns=False,
        dataloader_num_workers=2,

        # 其他
        report_to="none",
        seed=42
    )

    # 训练 DPO 模型
    save_path, merged_model, tokenizer = train_dpo(
        tokenizer=tokenizer,
        base_model=base_model,
        dpo_dataset=dpo_dataset,
        dpo_training_args=dpo_training_args,
        save_path=args.output_dir
    )

    # 释放显存
    del base_model, merged_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    main(args)
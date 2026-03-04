import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset
from utils import get_sft_data


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="SFT 模型训练脚本")

    # 模型配置
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="基础模型名称或路径")
    parser.add_argument("--output_dir", type=str, default="./models/sft", help="SFT 模型保存路径")

    # 数据配置
    parser.add_argument("--belle_data_path", type=str, default="./data/belle.json", help="BELLE 数据路径")

    # 训练配置
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="每个设备的训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="学习率")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热比例")
    parser.add_argument("--save_steps", type=int, default=1000, help="保存步数")
    parser.add_argument("--logging_steps", type=int, default=100, help="日志步数")

    return parser.parse_args()


def train_sft(args):
    """
    训练 SFT 模型
    """
    sft_path = args.output_dir
    if os.path.exists(os.path.join(sft_path, "config.json")):
        print(f"SFT 模型已存在，路径: {sft_path}")
        return sft_path
    else:
        print("SFT 模型不存在，开始训练...")

        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

        # 加载模型（4-bit 量化）
        qlora_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=qlora_config,
            device_map="auto",
            trust_remote_code=True
        )

        # 添加 LoRA 适配器
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )

        model = get_peft_model(model, lora_config)
        print("模型参数:")
        model.print_trainable_parameters()

        # 获取 SFT 数据
        dataset = get_sft_data(args.belle_data_path)

        # 数据预处理函数
        def preprocess_function(example):
            # 按照 Qwen2.5 的格式构建输入
            batch_encoding = tokenizer.apply_chat_template(
                example["messages"],
                add_generation_prompt=False,
                return_tensors="pt"
            )
            # 从 BatchEncoding 中提取 input_ids
            input_ids = batch_encoding.input_ids.squeeze(0)
            return {"input_ids": input_ids, "labels": input_ids}

        # 应用预处理函数
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=False,
            remove_columns=dataset.column_names
        )

        # 训练参数
        training_args = TrainingArguments(
            output_dir=sft_path,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            warmup_ratio=args.warmup_ratio,
            save_strategy="steps",
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
            fp16=not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
            optim="paged_adamw_32bit",
            lr_scheduler_type="cosine",
            report_to="none"
        )

        # 创建 Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=lambda data: {
                "input_ids": torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(item["input_ids"]) for item in data],
                    batch_first=True,
                    padding_value=tokenizer.pad_token_id
                ),
                "labels": torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(item["labels"]) for item in data],
                    batch_first=True,
                    padding_value=-100
                ),
            }
        )

        # 开始训练
        print("开始 SFT 训练...")
        trainer.train()

        # 保存模型
        print("保存 SFT 模型...")
        model.save_pretrained(sft_path)
        tokenizer.save_pretrained(sft_path)
        print(f"SFT 模型已保存到: {sft_path}")

        # 释放显存
        del model, trainer
        torch.cuda.empty_cache()

        return sft_path


if __name__ == "__main__":
    args = parse_args()
    train_sft(args)
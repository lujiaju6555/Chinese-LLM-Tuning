import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset, Dataset
from tqdm import tqdm
import pandas as pd

# 加载 CMMLU 数据集
def load_cmmlu_data(cmmlu_data_path, cmmlu_subsets):
    """
    加载 CMMLU 数据集
    
    Args:
        cmmlu_data_path: CMMLU 数据集文件路径
        cmmlu_subsets: 要使用的子集列表
    
    Returns:
        按子集分组的数据集
    """
    print("从指定路径加载 CMMLU 数据集...")

    # 检查文件是否存在
    if not os.path.exists(cmmlu_data_path):
        raise FileNotFoundError(f"CMMLU 数据集文件不存在: {cmmlu_data_path}")

    # 首先查看数据结构
    with open(cmmlu_data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 按子集分组
    subset_data = {}
    # 处理不同格式的数据
    if isinstance(dataset, list):
        for item in dataset:
            # 尝试多种可能的字段名
            subset = item.get("subject", "")

            # 尝试多种可能的问题字段名
            question = item.get("Question", "")

            # 尝试多种可能的答案字段名
            answer = item.get("Answer", "")

            # 确保选项字段存在
            options = []
            if "A" in item and "B" in item and "C" in item and "D" in item:
                options = [item.get("A", ""), item.get("B", ""), item.get("C", ""), item.get("D", "")]

            # 只有当问题和答案都非空时才添加
            if question and answer and options:
                # 重新格式化为标准格式
                formatted_item = {
                    "question": question,
                    "options": options,
                    "answer": answer,
                    "category": subset
                }

                if subset not in subset_data:
                    subset_data[subset] = []
                subset_data[subset].append(formatted_item)
    elif isinstance(dataset, dict):
        # 处理字典格式，可能是 {"data": [...]} 或 {"subject1": [...], "subject2": [...]}
        if 'data' in dataset and isinstance(dataset['data'], list):
            # 处理 {"data": [...]} 格式
            for item in dataset['data']:
                subset = item.get("subject", item.get("category", "default"))
                question = item.get("question", "")
                answer = item.get("answer", "")

                options = []
                if "A" in item and "B" in item and "C" in item and "D" in item:
                    options = [item.get("A", ""), item.get("B", ""), item.get("C", ""), item.get("D", "")]
                elif "options" in item and isinstance(item["options"], list):
                    options = item["options"]

                if question and answer and options:
                    formatted_item = {
                        "question": question,
                        "options": options,
                        "answer": answer,
                        "category": subset
                    }

                    if subset not in subset_data:
                        subset_data[subset] = []
                    subset_data[subset].append(formatted_item)
        else:
            # 处理 {"subject1": [...], "subject2": [...]} 格式
            for subset, items in dataset.items():
                if isinstance(items, list):
                    for item in items:
                        question = item.get("question", "")
                        answer = item.get("answer", "")

                        options = []
                        if "A" in item and "B" in item and "C" in item and "D" in item:
                            options = [item.get("A", ""), item.get("B", ""), item.get("C", ""), item.get("D", "")]
                        elif "options" in item and isinstance(item["options"], list):
                            options = item["options"]

                        if question and answer and options:
                            formatted_item = {
                                "question": question,
                                "options": options,
                                "answer": answer,
                                "category": subset
                            }

                            if subset not in subset_data:
                                subset_data[subset] = []
                            subset_data[subset].append(formatted_item)
    else:
        # 如果是其他格式，抛出异常
        raise ValueError(f"未知的数据格式: {type(dataset)}")

    # 只保留指定的子集
    selected_subset_data = {}
    for subset in cmmlu_subsets:
        if subset in subset_data:
            selected_subset_data[subset] = subset_data[subset]
            print(f"子集 {subset}: {len(selected_subset_data[subset])} 题")

    if not selected_subset_data:
        print("警告：没有找到指定子集的数据")
        print(f"可用的子集: {list(subset_data.keys())}")

    return selected_subset_data


# 构造 CMMLU 格式的 prompt
def format_cmmlu_prompt(question, options, subject):
    """
    构造 CMMLU 格式的 prompt

    Args:
        question: 问题
        options: 选项列表
        subject: 科目/子集

    Returns:
        格式化后的 prompt
    """
    prompt = f"以下是中国关于{subject}的单项选择题，请选出其中的正确答案，仅输出A/B/C/D中的一个选项（仅输出一个大写字母）。\n\n"
    prompt += f"问题：{question}\n"
    for idx, opt in enumerate(options):
        prompt += f"{chr(65 + idx)}. {opt}\n"
    prompt += "答案："
    return prompt


# 评估模型的CMMLU准确率
def evaluate_model(model_path, result_path, model_name, model, tokenizer, cmmlu_data_path, cmmlu_subsets, limit=None):
    """
    评估模型的CMMLU准确率
    
    Args:
        model_path: 模型路径
        result_path: 结果保存路径
        model_name: 模型名称
        model: 模型实例
        tokenizer: 分词器实例
        cmmlu_data_path: CMMLU 数据集路径
        cmmlu_subsets: CMMLU 子集列表
        limit: 限制评估样本数
    
    Returns:
        评估结果
    """
    cmmlu_result_path = os.path.join(result_path, "cmmlu.json")
    cmmlu_details_path = os.path.join(result_path, "cmmlu_details.json")

    # 加载 CMMLU 数据集
    print("加载 CMMLU 数据集...")
    subset_data = load_cmmlu_data(cmmlu_data_path, cmmlu_subsets)

    # 评估结果
    eval_results = {
        "results": {},
        "versions": {"cmmlu_native": "1.0"},
        "config": {}
    }

    # 详细评估过程
    evaluation_details = {
        "model_name": model_name,
        "details": {}
    }

    total_correct = 0
    total_samples = 0

    # 遍历每个子集
    for subset, items in subset_data.items():
        print(f"\n评估子集: {subset}")

        # 初始化该子集的详细记录
        evaluation_details["details"][subset] = []

        # 限制样本数（如果指定了 limit）
        if limit is not None:
            items = items[:limit]

        correct = 0
        samples = len(items)

        # 遍历每个问题
        for i, item in enumerate(items):
            # 构造 prompt
            question = item.get("question", "")
            options = item.get("options", [])
            answer = item.get("answer", "")

            # 构造 CMMLU 格式的 prompt
            prompt = format_cmmlu_prompt(question, options, subset)

            # 构建聊天模板
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            # 生成回答
            # 应用聊天模板
            chat_input = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            )

            # 修正：正确提取 input_ids tensor
            input_ids = chat_input['input_ids'].to(model.device)  # ✅ 提取 tensor 再移动设备

            # 生成回答
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=10,
                    temperature=0.0,
                    top_p=1.0,
                    do_sample=False
                )

            # 解码输出
            output = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

            # 提取答案
            predicted_answer = ""
            # 尝试从输出中提取 A/B/C/D
            for char in output:
                if char in ["A", "B", "C", "D"]:
                    predicted_answer = char
                    break

            # 检查答案
            is_correct = predicted_answer == answer
            if is_correct:
                correct += 1

            # 记录详细信息
            evaluation_details["details"][subset].append({
                "question_id": i + 1,
                "question": question,
                "options": options,
                "prompt": prompt,
                "model_output": output,
                "predicted_answer": predicted_answer,
                "correct_answer": answer,
                "is_correct": is_correct
            })

            # 进度反馈
            if (i + 1) % 100 == 0:
                print(f"  已评估 {i + 1}/{samples} 题，正确率: {correct / (i + 1):.4f}")

        # 计算子集准确率
        accuracy = correct / samples if samples > 0 else 0.0
        eval_results["results"][f"cmmlu_{subset}"] = {
            "acc": accuracy,
            "samples": samples
        }

        # 累计总正确数和总样本数
        total_correct += correct
        total_samples += samples

        print(f"  子集 {subset} 评估完成，正确率: {accuracy:.4f} ({correct}/{samples})")

    # 计算总体准确率
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    eval_results["results"]["cmmlu"] = {
        "acc": overall_accuracy,
        "samples": total_samples
    }

    # 为了与原有格式兼容，添加 cmmlu:{subset} 格式的结果
    for subset in cmmlu_subsets:
        if f"cmmlu_{subset}" in eval_results["results"]:
            eval_results["results"][f"cmmlu:{subset}"] = {
                "acc": eval_results["results"][f"cmmlu_{subset}"]["acc"],
                "acc_stderr": 0.0
            }
        else:
            eval_results["results"][f"cmmlu:{subset}"] = {
                "acc": 0.0,
                "acc_stderr": 0.0
            }

    print(f"\n总体评估完成，总正确率: {overall_accuracy:.4f} ({total_correct}/{total_samples})")
    results = eval_results

    # 保存结果
    os.makedirs(result_path, exist_ok=True)
    with open(cmmlu_result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"{model_name} 评估结果已保存到: {cmmlu_result_path}")

    # 保存详细评估过程
    with open(cmmlu_details_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_details, f, ensure_ascii=False, indent=2)
    print(f"{model_name} 详细评估过程已保存到: {cmmlu_details_path}")

    return results


# 准备SFT的数据
def get_sft_data(belle_data_path):
    """
    准备SFT的数据
    
    Args:
        belle_data_path: BELLE 数据路径
    
    Returns:
        格式化后的数据集
    """
    # 加载并准备数据
    with open(belle_data_path, 'r', encoding='utf-8') as f:
        belle_data = json.load(f)  # 假设是列表格式 [{"instruction": "...", "input": "...", "output": "..."}, ...]

    # 转换为 Dataset 格式
    dataset = Dataset.from_list(belle_data)

    # 格式化数据
    def format_example(example):
        # Qwen2.5 模型的输入格式
        messages = [
            {
                "role": "user",
                "content": example["instruction"]
            },
            {
                "role": "assistant",
                "content": example["output"]
            }
        ]
        return {"messages": messages}

    # 应用格式化函数
    formatted_dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    # 过滤掉空的输入或输出
    formatted_dataset = formatted_dataset.filter(
        lambda x: x["messages"][0]["content"] and x["messages"][1]["content"]
    )

    print(f"格式化后的数据量: {len(formatted_dataset)}")

    return formatted_dataset


# 使用给定的 tokenizer 和 model 对 BELLE 评估数据生成完整回答。
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

    # 保存结果
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 生成完成！共 {len(results)} 条，已保存至: {save_path}")


# 构造偏好数据：对每条指令生成多个不同回答
def generate_multiple_responses(
        data_path,
        tokenizer,
        model,
        save_path,
        num_samples=4,  # 每条指令生成多少个回答
        max_new_tokens=2048,
        device="cuda",
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

    # 加载前 N 条数据（这里取全部，你可以在调用时传入切片）
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
            ).to(device)

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


# 加载并清洗偏好数据
def load_preference_data(preference_data_path, max_samples=None):
    """
    加载并清洗偏好数据
    
    Args:
        preference_data_path: 偏好数据文件路径
        max_samples: 最大样本数
    
    Returns:
        清洗后的偏好数据
    """
    print(f"加载偏好数据: {preference_data_path}")

    # 检查文件是否存在
    if not os.path.exists(preference_data_path):
        raise FileNotFoundError(f"偏好数据文件不存在: {preference_data_path}")

    # 加载数据
    with open(preference_data_path, "r", encoding="utf-8") as f:
        preference_data = json.load(f)

    print(f"原始数据量: {len(preference_data)}")

    # 数据清洗
    cleaned_data = []
    for item in preference_data:
        # 过滤掉 sorted_indices is None 的样本
        if "sorted_indices" not in item or item["sorted_indices"] is None:
            continue

        # 过滤掉 responses 中存在任意两个完全相同回答的样本
        responses = item.get("responses", [])
        if len(responses) != len(set(responses)):
            continue

        # 确保有足够的回答
        if len(responses) < 2:
            continue

        cleaned_data.append(item)

    print(f"清洗后数据量: {len(cleaned_data)}")

    # 取前 N 条数据
    if max_samples is not None and len(cleaned_data) > max_samples:
        cleaned_data = cleaned_data[:max_samples]
        print(f"限制数据量为: {max_samples} 条")

    return cleaned_data


# 从偏好数据构建 DPO 数据集
def build_dpo_dataset(preference_data, tokenizer):
    """
    从偏好数据构建 DPO 数据集（使用 Qwen 官方 chat template）

    Args:
        preference_data: 原始偏好数据列表
        tokenizer: 已加载的 Qwen tokenizer（需支持 apply_chat_template）
    """
    dpo_data = []

    for item in preference_data:
        instruction = item.get("instruction", "").strip()
        responses = [r.strip() for r in item.get("responses", [])]
        sorted_indices = item.get("sorted_indices", [])

        if len(sorted_indices) >= 2:
            chosen_idx = sorted_indices[0]
            rejected_idx = sorted_indices[-1]

            if (0 <= chosen_idx < len(responses)) and (0 <= rejected_idx < len(responses)):
                chosen = responses[chosen_idx]
                rejected = responses[rejected_idx]

                # ✅ 关键修改：用 apply_chat_template 生成标准 prompt
                messages = [{"role": "user", "content": instruction}]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True  # 包含 <|im_start|>assistant\n
                )

                dpo_item = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected
                }
                dpo_data.append(dpo_item)

    print(f"DPO 数据集大小: {len(dpo_data)}")

    # 转换为 Dataset 对象
    dataset = Dataset.from_list(dpo_data)
    return dataset


# 训练 DPO 模型
def train_dpo(tokenizer, base_model, dpo_dataset, dpo_training_args, save_path):
    """
    训练 DPO 模型
    - base_model: 可以是已合并的完整模型（如 SFT 合并后）
    - 函数内部会重新添加 LoRA 适配器用于 DPO 微调
    
    Args:
        tokenizer: 分词器实例
        base_model: 基础模型实例
        dpo_dataset: DPO 数据集
        dpo_training_args: DPO 训练参数
        save_path: 模型保存路径
    
    Returns:
        保存路径, 合并后的模型, 分词器
    """
    # 1. 给 base_model 重新加上 LoRA（用于 DPO 微调）
    print("为模型添加 LoRA 适配器用于 DPO...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model_for_dpo = get_peft_model(base_model, lora_config)

    # 2. 创建 DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model_for_dpo,
        args=dpo_training_args,
        train_dataset=dpo_dataset,
        processing_class=tokenizer,
    )

    print("开始 DPO 训练...")
    dpo_trainer.train()
    print("DPO 训练完成")

    # 3. 合并 LoRA 权重（现在 model_for_dpo 是 PeftModel，有 merge_and_unload）
    print("合并 DPO 的 LoRA 权重...")
    merged_model = dpo_trainer.model.merge_and_unload()

    # 4. 保存模型和 tokenizer
    os.makedirs(save_path, exist_ok=True)
    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"DPO 模型已保存到: {save_path}")

    # 5. 清理显存
    del model_for_dpo, dpo_trainer
    torch.cuda.empty_cache()
    print("显存已释放")

    return save_path, merged_model, tokenizer
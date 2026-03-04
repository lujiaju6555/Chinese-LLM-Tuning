# Chinese-LLM-Tuning

这是一个用于微调中文大语言模型的项目，包含 SFT（监督微调）和 DPO（直接偏好优化）训练流程，以及 vLLM 推理部署。

## 项目结构

```
Chinese-LLM-Tuning/
├── README.md
├── requirements.txt
├── sft.py          # SFT 训练主流程
├── dpo.py          # DPO 训练主流程
├── vllm.py     # vLLM 推理/部署
├── utils.py              # 共享工具函数
├── get_data.py      # 数据下载脚本
├── data/                 # 数据目录
├── models/               # 模型目录
└── results/              # 结果目录
```

## 环境准备

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 下载数据

```bash
python download_data.py --data_dir ./data
```

## 训练流程

### 1. SFT 训练

```bash
python sft_qlora.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --output_dir ./models/sft \
    --belle_data_path ./data/belle.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-7 \
    --num_train_epochs 1
```

### 2. 生成偏好数据

使用 `llm_preference.py` 生成偏好数据，然后使用 `llm_judge.py` 进行排序。

### 3. DPO 训练

```bash
python dpo_train.py \
    --base_model ./models/sft \
    --output_dir ./models/dpo \
    --preference_data_path ./data/preference_data.json \
    --max_samples 5000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-6 \
    --num_train_epochs 1
```

## 推理部署

使用 vLLM 进行高效推理：

```bash
python vllm_inference.py \
    --model_path ./models/dpo \
    --data_path ./data/belle_eval.json \
    --output_path ./results/dpo/eval_response.json \
    --max_new_tokens 512
```

## 评估

使用 `llm_judge.py` 对模型输出进行评估：

```bash
python llm_judge.py
```

## 主要功能

- **SFT 训练**：使用 QLoRA 方法对模型进行监督微调
- **DPO 训练**：使用偏好数据对模型进行对齐训练
- **vLLM 推理**：使用 vLLM 进行高效推理部署
- **数据下载**：本地下载所需数据集
- **模型评估**：使用大模型对模型输出进行评估

## 注意事项

- 训练前请确保有足够的 GPU 内存
- 首次运行会自动下载模型和数据
- 可以根据硬件条件调整 batch size 和 gradient accumulation steps
- 偏好数据生成需要调用外部 API，请确保 API_KEY 已正确设置
- 本项目仅为本人技术实现，仅供参考，若有不正确之处欢迎指出

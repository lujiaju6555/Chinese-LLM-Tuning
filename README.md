# Chinese-LLM-Tuning

本项目用于微调中文大语言模型，包含 **SFT（监督微调）**、**DPO（直接偏好优化）** 和 **vLLM 高效推理部署** 的完整后训练 pipeline。  
基于 **Qwen2.5-1.5B-Instruct** 模型，在保持中文知识能力的同时，显著提升其中文任务表现与推理效率。

## 📊 核心效果

| 阶段 | 指标 | 提升效果 |
|------|------|--------|
| **SFT** | CMMLU 准确率 | **+4%**（vs. Baseline） |
|         | LLM-as-a-Judge 评分 | **+0.18 分（+2.5%）** |
| **DPO** | CMMLU 准确率 | -1%（vs. SFT） |
|         | LLM-as-a-Judge 评分 | **+0.16 分（+2.1%）** |
| **vLLM 部署** | 吞吐量（tokens/s） | **≈60×** 提升（vs. Hugging Face Transformers 原生推理） |

> 💡 **说明**：  
> - SFT 使用 **50k 条 BELLE 中文指令数据**，DPO 使用 **5k 条 GPT-4 构建的偏好对**；  
> - vLLM 吞吐量优势部分源于 HF 原生推理在消费级显卡（如 RTX 4090）上受限于 `per_device_train_batch_size=1`，无法有效利用 GPU 并行能力。

欢迎参与项目并进一步改进。

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
├── llm_judge.py      # 大模型评估脚本
├── llm_preference.py      # 大模型生成偏好数据脚本
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

CMMLU数据已放于data目录下，无需下载。此处下载BELLE数据，包含50k条指令数据、500条评估数据和5k条偏好对数据，可调整数据量。
注意需要链接VPN。

```bash
python get_data.py --data_dir ./data --train_sample_num 50000 --eval_sample_num 500 --preference_sample_num 5000
```

## 训练流程

### 1. SFT 训练

注意需要VPN，如果无法连接，可改用modelscope先下载到本地，然后指定本地路径（即修改base_model参数）。

```bash
python sft.py \
    --base_model Qwen/Qwen2.5-1.5B-Instruct \
    --output_dir ./models/sft \
    --belle_data_path ./data/belle.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-7 \
    --num_train_epochs 1
```

### 2. 生成偏好数据

使用 `llm_preference.py` 生成偏好数据，支持三种运行模式：

#### 2.1 先生成候选回答，再排序偏好（推荐）

```bash
python llm_preference.py \
    --mode both \
    --api_key "your_api_key" \
    --baseline_model_path ./models/baseline \
    --sft_model_path ./models/sft \
    --preference_data_path ./data/belle_preference.json \
    --candidates_path ./data/belle_preference_response.json \
    --output_path ./data/preference_data.json \
    --model qwen3.5-flash
```

#### 2.2 仅生成候选回答

```bash
python llm_preference.py \
    --mode generate \
    --baseline_model_path ./models/baseline \
    --sft_model_path ./models/sft \
    --preference_data_path ./data/belle_preference.json \
    --candidates_path ./data/belle_preference_response.json \
    --num_samples 4 \
    --temperature 0.7
```

#### 2.3 仅排序偏好数据

```bash
python llm_preference.py \
    --mode rank \
    --api_key "your_api_key" \
    --candidates_path ./data/belle_preference_response.json \
    --output_path ./data/preference_data.json \
    --model qwen3.5-flash
```

### 3. DPO 训练

DPO训练流程：首先加载SFT模型，其次加载并清洗偏好数据，使用清洗后的数据构建适用于DPO的数据集，最后进行DPO训练并保存。

```bash
python dpo.py \
    --baseline_model_path ./models/baseline \
    --sft_model_path ./models/sft \
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
python vllm.py \
    --model_path ./models/dpo \
    --data_path ./data/belle_eval.json \
    --output_path ./results/dpo/eval_response.json \
    --max_new_tokens 2048
```

## 评估

使用 `llm_judge.py` 对模型输出进行评估，使用 Qwen3-max 作为 Judge 模型，通过 LLM-as-a-Judge 方式对模型输出进行自动评估。可分别对Baseline模型、SFT模型和DPO模型进行评估 ：

```bash
python llm_judge.py \
    --api_key "your_api_key" \
    --target dpo \
    --input_file ./results/dpo/eval_response.json \
    --output_path ./results/dpo/eval_result.json \
    --statistics_path ./results/dpo/score_statistics.json \
    --judge_model qwen3-max
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
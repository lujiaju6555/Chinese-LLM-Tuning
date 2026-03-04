# 面向中文指令的大模型后训练与部署系统

本项目旨在使用 Qwen2.5-1.5B-Instruct 作为基座模型，通过 SFT（监督微调）和 DPO（对齐优化）技术，构建一个性能良好的中文大模型。

## 项目结构

```
Chinese-LLM-Tuning/
├── requirements.txt       # 依赖包列表
├── .gitignore             # Git 忽略文件
├── main.ipynb          # 主程序，包括微调和偏好对齐
├── llm_judge.py         # 使用大模型评估
├── llm_preference.py     # 使用大模型构建偏好数据
├── check_preference.py     # 检查偏好数据
├── vllm.ipynb        # 部署vllm加速推理
├── download_data.ipynb        # 下载数据
├── data/
│   └── belle_eval.json  # 用于评估的belle数据
│   └── cmmlu.json  # 用于评估的cmmlu数据
└── README.md             # 项目说明
```

### 依赖安装

在项目根目录运行：

```bash
pip install -r requirements.txt
```

## 训练流程

### 1. 数据下载

使用 `download_data.ipynb` 脚本下载 Belle 中文指令数据集，作为SFT的微调数据，以及偏好数据的基础。

### 2. 监督微调（SFT）

使用 QLoRA 技术对 Qwen2.5-1.5B-Instruct 模型进行微调，数据集为 Belle 中文指令数据集。

代码：`main.ipynb` 

### 3. 评估结果

使用`main.ipynb` 中代码评估CMMLU准确率，并生成评估回答，再使用 `llm_judge.py` 评估回答质量。
注意，后一步使用大模型评估回答质量时，需要在代码开头填写自己的阿里云API_KEY。若要修改使用的模型，修改对应参数即可，默认使用qwen3.5-flash
### 3. 对齐优化（DPO）

使用微调后的 SFT 模型生成多个回答，再使用 `llm_preference.py` 构造偏好对，最后进行 DPO 训练。

代码：`main.ipynb`
注意，使用大模型构造偏好对时，需要在代码开头填写自己的阿里云API_KEY。若要修改使用的模型，修改对应参数即可，默认使用qwen-flash

## 技术细节

### 模型配置
- 基座模型：Qwen2.5-1.5B-Instruct
- 微调技术：QLORA（4-bit + LoRA）
- 对齐方法：DPO

### 数据集
- SFT 数据集：BelleGroup/train_2M_CN（Hugging Face）选取50000条，可调节
- 评估数据集：CMMLU中文知识选择题+Belle最后500条数据
- 偏好数据集：基于Belle最后开始、删除评估的500条数据的10000条数据构造，过滤得到5000条高质量数据，每条数据4个候选回答。

### 评估
- 知识能力：CMMLU benchmark（zero-shot）
- 对齐效果：CMMLU准确率提升3%，llm-as-a-judge评分提升4.5%

## 注意事项

1. 需要准备好数据，并放置于对应的路径
2. DPO 训练需要先完成 SFT 训练，生成 SFT 模型
3. 本项目仅为本人技术实现，仅供参考，若有不正确之处欢迎指出
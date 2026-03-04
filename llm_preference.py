import json
import time
from openai import OpenAI
from tqdm import tqdm
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# API_KEY = "sk-50164861345148c1ac4f371a3674f127"  # 1
# API_KEY = "sk-075f638c1cbc4d2b91c04d38d04ccfdb"  # 2
API_KEY = "sk-2c3bde2ec4ea430f9b1daccaf20313e5"  # 3


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
            future_to_item = {executor.submit(self.rank_responses_with_qwen, item["instruction"], item["responses"]): item for item in pending_data}
            
            # 处理完成的任务
            for future in tqdm(as_completed(future_to_item), total=len(future_to_item), desc="Ranking responses with Qwen-Flash"):
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

if __name__ == '__main__':
    # 初始化排序器
    ranker = PreferenceRanker(api_key=API_KEY)
    
    # 从指定路径加载数据
    data_path = "./data/belle_preference_response.json"
    output_path = "./data/preference_data.json"
    
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
            delay=0.5,
            max_workers=5
        )
    else:
        print("❌ 没有有效的数据项")
        exit(1)
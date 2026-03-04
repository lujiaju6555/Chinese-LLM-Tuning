import json
import os

# 配置文件路径
DATA_FILE = "./data/preference_data.json"  # 替换为你的实际路径


def analyze_preference_data(data_file):
    if not os.path.exists(data_file):
        print(f"❌ 文件不存在: {data_file}")
        return

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    valid_count = 0
    invalid_count = 0

    for item in data:
        # 检查 sorted_indices 是否为 None（在 JSON 中是 null）
        if item.get("sorted_indices") is None:
            invalid_count += 1
        else:
            valid_count += 1

    print("=" * 50)
    print("📊 偏好数据质量分析报告")
    print("=" * 50)
    print(f"总条数:          {total}")
    print(f"有效条数:        {valid_count}")
    print(f"无效条数:        {invalid_count}")
    print(f"有效率:          {valid_count / total * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    analyze_preference_data(DATA_FILE)
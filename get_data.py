import os
import argparse
import requests
import json
from tqdm import tqdm
from datasets import load_dataset
import pandas as pd
import os


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="数据下载脚本")

    # 数据配置
    parser.add_argument("--data_dir", type=str, default="./data", help="数据保存目录")
    parser.add_argument("--train_sample_num", type=int, default=50000, help="训练样本数量")
    parser.add_argument("--eval_sample_num", type=int, default=500, help="评估样本数量")
    parser.add_argument("--preference_sample_num", type=int, default=10000, help="偏好样本数量")

    return parser.parse_args()


def download_belle(data_dir, args):
    """
    下载 BELLE 数据集
    """
    # 加载完整 BELLE 数据集（2M 中文版）
    print("正在加载 BelleGroup/train_2M_CN 数据集...")
    dataset = load_dataset("BelleGroup/train_2M_CN", split="train")
    total_size = len(dataset)
    print(f"数据集总大小: {total_size:,} 条")

    # 保存前50000条数据作为样本（避免文件过大）
    sample_size = min(args.train_sample_num, len(dataset))
    sample_dataset = dataset.select(range(sample_size))

    # 转换为 DataFrame 并保存
    df = pd.DataFrame(sample_dataset)
    save_path = os.path.join(data_dir, 'belle.json')
    df.to_json(save_path, orient='records', lines=False, force_ascii=False)

    print(f"\n样本数据已保存到: {save_path}")
    print(f"保存了 {len(df)} 条数据")

    # === 2. preference 偏好生成集（用于生成候选回答）===
    # 取倒数第 105500 到倒数第 501 条（共 10000 条）
    preference_start = total_size - args.eval_sample_num - args.preference_sample_num
    preference_end = total_size - args.eval_sample_num
    preference_indices = range(preference_start, preference_end)

    preference_dataset = dataset.select(preference_indices)
    df_preference = pd.DataFrame(preference_dataset)
    preference_path = os.path.join(data_dir, 'belle_preference.json')
    df_preference.to_json(preference_path, orient='records', lines=False, force_ascii=False)
    print(f"✅ preference 候选生成集已保存 ({len(df_preference)} 条): {preference_path}")

    # === 3. 评估集（用于自动评估）===
    # 取最后 500 条
    eval_indices = range(total_size - args.eval_sample_num, total_size)
    eval_dataset = dataset.select(eval_indices)
    df_eval = pd.DataFrame(eval_dataset)
    eval_path = os.path.join(data_dir, 'belle_eval.json')
    df_eval.to_json(eval_path, orient='records', lines=False, force_ascii=False)
    print(f"✅ 评估集已保存 ({len(df_eval)} 条): {eval_path}")


def main(args):
    """
    主函数
    """
    # 创建数据目录
    os.makedirs(args.data_dir, exist_ok=True)

    # 下载 BELLE 数据集
    print("\n开始下载 BELLE 数据集...")
    download_belle(args.data_dir, args)

    print("\n所有数据下载完成！")


if __name__ == "__main__":
    args = parse_args()
    main(args)
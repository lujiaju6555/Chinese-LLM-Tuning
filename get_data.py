import os
import argparse
import requests
import json
from tqdm import tqdm


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="数据下载脚本")

    # 数据配置
    parser.add_argument("--data_dir", type=str, default="./data", help="数据保存目录")

    return parser.parse_args()


def download_file(url, save_path):
    """
    下载文件

    Args:
        url: 文件下载地址
        save_path: 保存路径
    """
    print(f"下载文件: {url}")
    print(f"保存路径: {save_path}")

    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 发送请求
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # 获取文件大小
    total_size = int(response.headers.get('content-length', 0))

    # 下载文件
    with open(save_path, 'wb') as file, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

    print(f"文件下载完成: {save_path}")


def download_cmmlu(data_dir):
    """
    下载 CMMLU 数据集
    """
    cmmlu_url = "https://huggingface.co/datasets/haonan-li/CMMLU/resolve/main/cmmlu.json"
    save_path = os.path.join(data_dir, "cmmlu.json")
    download_file(cmmlu_url, save_path)


def download_belle(data_dir):
    """
    下载 BELLE 数据集
    """
    # BELLE 2M 中文指令数据
    belle_url = "https://huggingface.co/datasets/BelleGroup/train_2M_CN/resolve/main/train_2M_CN.json"
    save_path = os.path.join(data_dir, "belle.json")
    download_file(belle_url, save_path)

    # BELLE 评估数据
    belle_eval_url = "https://huggingface.co/datasets/BelleGroup/eval_1K_CN/resolve/main/eval_1K_CN.json"
    eval_save_path = os.path.join(data_dir, "belle_eval.json")
    download_file(belle_eval_url, eval_save_path)


def main(args):
    """
    主函数
    """
    # 创建数据目录
    os.makedirs(args.data_dir, exist_ok=True)

    # 下载 CMMLU 数据集
    print("开始下载 CMMLU 数据集...")
    download_cmmlu(args.data_dir)

    # 下载 BELLE 数据集
    print("\n开始下载 BELLE 数据集...")
    download_belle(args.data_dir)

    print("\n所有数据下载完成！")


if __name__ == "__main__":
    args = parse_args()
    main(args)
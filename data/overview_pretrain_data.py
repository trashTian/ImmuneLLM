import os
from datasets import load_from_disk
import random
import numpy as np

# ================= 配置区域 =================
# 请确保存放数据的路径正确
DATASET_PATH = "/mnt/lustre/guopeijin/data/LLAPA/dataset/pretrain/uniprot"

# ===========================================

def main():
    print(f"🚀 Loading dataset from: {DATASET_PATH}")
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: Path does not exist!")
        return

    try:
        dataset = load_from_disk(DATASET_PATH)
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return

    # 处理 DatasetDict (包含 train/test) 或 单个 Dataset
    if hasattr(dataset, "keys") and ("train" in dataset or "test" in dataset):
        print(f"DatasetDict detected. Keys: {dataset.keys()}")
        # 默认看 train
        ds = dataset["train"]
    else:
        print("Single Dataset detected.")
        ds = dataset

    # 1. 基础信息
    print("\n" + "="*40)
    print("📊 基本信息 (Basic Info)")
    print("="*40)
    print(f"Total Rows: {len(ds)}")
    print(f"Features:   {ds.column_names}")
    print(ds)

    # 2. 随机抽样检查
    print("\n" + "="*40)
    print("🔍 随机样本展示 (Random Sample)")
    print("="*40)
    
    # 随机取 3 个下标
    indices = random.sample(range(len(ds)), 3)
    
    for i, idx in enumerate(indices):
        item = ds[idx]
        print(f"\n[Sample #{i+1} | Index: {idx}]")
        
        # 打印各个字段
        for key, value in item.items():
            # 如果是列表（比如 proteins），只打印长度或者前几个
            if isinstance(value, list) and len(value) > 0:
                # 针对 proteins 字段特殊处理
                if key == 'proteins':
                    print(f"  > {key}: [List length {len(value)}]")
                    print(f"    First Protein (Len {len(value[0])}): {value[0][:50]}...")
                else:
                    print(f"  > {key}: {value}")
            # 如果是字符串，太长就截断显示
            elif isinstance(value, str):
                display_val = value if len(value) < 200 else (value[:200] + "... [Truncated]")
                print(f"  > {key}: \n    {display_val}")
            else:
                print(f"  > {key}: {value}")

    # 3. 长度统计 (用于优化显存配置)
    print("\n" + "="*40)
    print("📏 长度分布统计 (Length Stats)")
    print("="*40)
    
    # 为了速度，只采样 10000 条进行统计
    sample_size = min(10000, len(ds))
    subset = ds.select(range(sample_size))
    
    text_lengths = []
    prot_lengths = []
    
    print(f"Computing stats on {sample_size} samples...")
    for item in subset:
        # 统计文本长度 (Question + Answer)
        q_len = len(item.get('question', ''))
        a_len = len(item.get('answer', ''))
        text_lengths.append(q_len + a_len)
        
        # 统计蛋白质长度 (取第一个蛋白)
        prots = item.get('proteins', [])
        if isinstance(prots, list) and len(prots) > 0:
            prot_lengths.append(len(prots[0]))
        elif isinstance(prots, str):
            prot_lengths.append(len(prots))
            
    if text_lengths:
        print(f"\n[Text Lengths (Char count approx.)]")
        print(f"  Mean: {np.mean(text_lengths):.1f}")
        print(f"  Max:  {np.max(text_lengths)}")
        print(f"  P95:  {np.percentile(text_lengths, 95):.1f}")
        
    if prot_lengths:
        print(f"\n[Protein Lengths (AA count)]")
        print(f"  Mean: {np.mean(prot_lengths):.1f}")
        print(f"  Max:  {np.max(prot_lengths)}")
        print(f"  P95:  {np.percentile(prot_lengths, 95):.1f}")
        print(f"  P99:  {np.percentile(prot_lengths, 99):.1f}")

if __name__ == "__main__":
    main()

    """
    预训练的esm max length 可设置为1024
    llm max length 可设置为1024
    
    """
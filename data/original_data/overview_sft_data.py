import os
from datasets import load_from_disk
import numpy as np
import pandas as pd
from collections import Counter

# ================= 配置区域 =================
# 请替换为你实际的 Stage 2 数据集路径
# 例如你之前生成的: /mnt/lustre/guopeijin/Immune_LLM/code/data_prepare/stage2_sft_baseline_direct
DATASET_PATH = "/mnt/lustre/guopeijin/Immune_LLM/code/data_prepare/stage2_sft_baseline_direct"

# 特殊 Token (用于检查 Prompt 是否包含)
TOKEN_REC = "<|reserved_special_token_70|>"  # 或者 <|REC|>
TOKEN_LIG = "<|reserved_special_token_71|>"  # 或者 <|LIG|>
# ===========================================

def analyze_lengths(name, sequences):
    lengths = [len(s) for s in sequences]
    print(f"  [{name} Lengths]")
    print(f"    Mean: {np.mean(lengths):.1f}")
    print(f"    Min:  {np.min(lengths)}")
    print(f"    Max:  {np.max(lengths)}")
    print(f"    P95:  {np.percentile(lengths, 95):.1f}")
    print(f"    P99:  {np.percentile(lengths, 99):.1f}")

def main():
    print(f"🚀 Loading Stage 2 dataset from: {DATASET_PATH}")
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: Path does not exist!")
        return

    try:
        dataset_raw = load_from_disk(DATASET_PATH)
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return

    # 兼容 DatasetDict 和 Dataset
    if hasattr(dataset_raw, "keys") and ("train" in dataset_raw or "test" in dataset_raw):
        print(f"DatasetDict detected. Keys: {dataset_raw.keys()}")
        ds = dataset_raw["train"] # 默认分析 train
    else:
        print("Single Dataset detected.")
        ds = dataset_raw

    # 1. 基础概览
    print("\n" + "="*40)
    print("📊 数据集概览 (Overview)")
    print("="*40)
    print(f"Total Count: {len(ds)}")
    print(f"Columns:     {ds.column_names}")
    
    # 2. 标签分布 (检查正负样本平衡)
    if 'label' in ds.column_names:
        labels = ds['label']
        counts = Counter(labels)
        print(f"\n🏷️  Label Distribution:")
        total = len(labels)
        for lbl, count in counts.items():
            print(f"  Label {lbl}: {count} ({count/total*100:.2f}%)")
    
    # 3. 任务类型分布 (如果有 task_type 列)
    if 'task_type' in ds.column_names:
        tasks = ds['task_type']
        print(f"\n🧬 Task Distribution:")
        print(pd.Series(tasks).value_counts().to_string())

    # 4. 长度统计 (Receptor vs Ligand)
    print("\n" + "="*40)
    print("📏 序列长度统计 (Sequence Lengths)")
    print("="*40)
    # 采样 10000 条以加快速度
    sample_size = min(20000, len(ds))
    subset = ds.select(range(sample_size))
    
    if 'receptor_seq' in subset.column_names:
        analyze_lengths("Receptor (HLA/TCR)", subset['receptor_seq'])
    
    if 'ligand_seq' in subset.column_names:
        analyze_lengths("Ligand (Peptide)", subset['ligand_seq'])

    # 5. 样本详情 & Prompt 检查
    print("\n" + "="*40)
    print("🔍 样本详情 (Sample Inspection)")
    print("="*40)
    
    indices = [0, 100, len(ds)-1] # 查看开头、中间、结尾
    for idx in indices:
        item = ds[idx]
        print(f"\n[Index: {idx}]")
        print(f"  Task:   {item.get('task_type', 'N/A')}")
        print(f"  Label:  {item.get('label', 'N/A')}")
        print(f"  Rec:    {item.get('receptor_seq', '')[:50]}...")
        print(f"  Lig:    {item.get('ligand_seq', '')}")
        print("-" * 20)
        print(f"  Question (Input):\n  {repr(item.get('question', ''))}")
        print("-" * 20)
        print(f"  Answer (Output):\n  {repr(item.get('answer', ''))}")
        
        # 自动检查特殊 Token
        q_text = item.get('question', '')
        if TOKEN_REC not in q_text or TOKEN_LIG not in q_text:
            print(f"  ⚠️  WARNING: Special tokens not found in Question! Check your generation script.")
        else:
            print(f"  ✅ Format OK: Special tokens detected.")

if __name__ == "__main__":
    main()
    """
    self.max_len_esm = 64  # 原来可能是 1024 或 512
    self.max_len_llama = 256 # 简单 QA 文本很短，256 足够
    """
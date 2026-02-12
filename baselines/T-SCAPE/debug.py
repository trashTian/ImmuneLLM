import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import os
import sys
import numpy as np

# 导入 T-SCAPE 模块
from src.io_utils_fused import CSVDataset_test, Collater_test
from src.constants import PAD, PROTEIN_ALPHABET
from src.model_fused import task3, task9

def inspect_tensor(name, tensor):
    """可视化 Tensor 内容 (支持 One-Hot)"""
    print(f"\n[DEBUG] Inspecting {name}: Shape={tensor.shape}")
    
    # 检查是否为 One-Hot (3维张量: Batch, Length, Channels)
    if tensor.dim() == 3:
        # 将 One-Hot 转回索引 (Argmax)
        # tensor: [Batch, Length, Channels] -> indices: [Batch, Length]
        indices = torch.argmax(tensor, dim=-1).cpu().numpy()
        sample = indices[0] # 取 Batch 中的第一个样本
        print(f"  Converted Indices (One-Hot -> Index): {sample}")
    else:
        # 如果是简单的索引张量
        sample = tensor[0].cpu().numpy()
        print(f"  Raw Indices: {sample}")

    # 检查数据是否“单调”（全是一样的值，说明全是 Padding 或空）
    unique_vals = np.unique(sample)
    print(f"  Unique token indices found: {unique_vals}")
    
    # 判定逻辑：如果整个序列只有 1 种或 2 种数值（通常是 Pad 或 Start/Pad），且长度全是 Pad
    # 我们假设 Padding 的 Index 是出现次数最多的那个
    vals, counts = np.unique(sample, return_counts=True)
    most_freq_val = vals[np.argmax(counts)]
    most_freq_count = np.max(counts)
    
    print(f"  Most frequent token: {most_freq_val} (Count: {most_freq_count}/{len(sample)})")
    
    # 警告条件：如果 90% 以上都是同一个 token，可能是空数据
    if most_freq_count >= len(sample) * 0.9:
        print(f"  🔴 ALARM: {name} looks suspicious! It is mostly constant (likely all padding).")
    else:
        print(f"  🟢 {name} looks valid (contains varied sequence data).")

def train_debug(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Debugging on device: {device}")
    
    # 1. 读取数据
    print(f"Reading CSV: {args.train_csv}")
    df = pd.read_csv(args.train_csv)
    
    # 简单预处理
    df['peptide'] = df['peptide'].fillna("").astype(str)
    if 'pseudo' not in df.columns: df['pseudo'] = ""
    else: df['pseudo'] = df['pseudo'].fillna("").astype(str)
    
    # === 关键点：检查 CDR3b ===
    if 'CDR3b' not in df.columns:
        print("⚠️ Warning: CDR3b column missing, filling with empty strings.")
        df['CDR3b'] = ""
    else:
        df['CDR3b'] = df['CDR3b'].fillna("").astype(str)
        
    df['task'] = 1
    df['mhc'] = ""
    
    # 过滤
    df = df[df['peptide'].apply(len) <= 20]
    if args.task_type == 'ptcr_ba':
        df = df[df['CDR3b'].apply(len) <= 20]
    
    print(f"Dataset size: {len(df)}")
    
    # DataLoader
    dataset = CSVDataset_test(df)
    collater = Collater_test(alphabet=PROTEIN_ALPHABET, pad=True, backwards=False, pad_token=PAD)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collater)
    
    # 模型
    model = task9(d_model=280, n_tokens=29, kernel_size=1, n_layers=6, d_embedding=280, r=1, mask_condition=False)
    
    # 加载权重
    print(f"Loading weights from {args.pretrained_path}...")
    ckpt = torch.load(args.pretrained_path, map_location=device)
    # 模糊加载
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()}, strict=False)
    model.to(device)
    model.train()
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print("\n=== STARTING DEBUG LOOP ===")
    
    for i, (src, m1, m2, tcr, frac, p_lens, mhcs) in enumerate(loader):
        src, m1, m2, tcr = src.to(device), m1.to(device), m2.to(device), tcr.to(device)
        labels = torch.tensor(frac).float().to(device).unsqueeze(1)
        
        print(f"\n--- Batch {i} ---")
        # 检查 Peptide
        inspect_tensor("SRC (Peptide)", src)
        
        # 检查 TCR (重点看这里！)
        if args.task_type == 'ptcr_ba':
            inspect_tensor("TCR (CDR3b)", tcr)
        
        # Forward
        optimizer.zero_grad()
        output = model(src, m1, m2, tcr=tcr, task=[9])
        logits = output[-1]
        # === 加入这行打印 ===
        print(f"[SHAPE DEBUG] Logits shape: {logits.shape}, Labels shape: {labels.shape}")
        
        print(f"[DEBUG] Logits: {logits.detach().cpu().numpy().flatten()[:4]}")
        loss = criterion(logits, labels)
        print(f"[DEBUG] Loss: {loss.item():.6f}")
        
        loss.backward()
        optimizer.step()
        
        if i >= 0: # 只跑 1 个 batch 就够了
            print("\n=== DEBUG FINISHED ===")
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--task_type', type=str, default='ptcr_ba')
    parser.add_argument('--pretrained_path', type=str, required=True)
    args = parser.parse_args()
    
    train_debug(args)


    """
    python debug.py \
  --train_csv /mnt/lustre/guopeijin/Immune_LLM/code/baselines/titanian/T-SCAPE-main/train_datas/tcr/train_fold_1.csv \
  --pretrained_path /mnt/lustre/guopeijin/Immune_LLM/code/baselines/titanian/T-SCAPE-main/checkpoints/models/Uni_el-fused_ADV1.0_0.pt
    
    """
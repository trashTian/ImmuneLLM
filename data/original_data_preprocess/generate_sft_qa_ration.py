import pandas as pd
from datasets import Dataset, concatenate_datasets, DatasetDict
import os
import numpy as np
from sklearn.model_selection import train_test_split

# ==========================================
# 1. 配置区域
# ==========================================
# UnifyImmune 原始数据路径
HLA_TRAIN_PATH = "/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/train_fold_1.csv"
TCR_TRAIN_PATH = "/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/train_fold_1.csv"

# 验证集路径 (保持全量)
HLA_VAL_PATH = "/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/val_fold_1.csv"
TCR_VAL_PATH = "/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/val_fold_1.csv"

# 输出根目录
OUTPUT_ROOT = "/mnt/lustre/guopeijin/Immune_LLM/code/data_prepare/stage2_data_efficiency"

# 采样比例列表
# RATIOS = [0.01, 0.05, 0.1, 0.2] 
RATIOS = [0.25] 

# Token Config
TOKEN_REC_ID = "<|reserved_special_token_70|>" 
TOKEN_LIG_ID = "<|reserved_special_token_71|>"
BEST_PROMPT_TEMPLATE = (
    "Analyze the structural compatibility between {rec} and {lig}.\n"
    "Prediction of stable complex formation (Yes/No):"
)

# ==========================================
# 2. 核心函数
# ==========================================
def load_and_process_csv(file_path, task_type):
    """读取 CSV，清洗，增加 Prompt 列，返回 DataFrame"""
    print(f"   Loading {task_type}: {file_path}")
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)

    # 统一列名
    if task_type == 'HLA':
        if 'HLA' in df.columns: df = df.rename(columns={'HLA': 'receptor_seq', 'peptide': 'ligand_seq', 'label': 'label'})
    else:
        if 'tcr' in df.columns: df = df.rename(columns={'tcr': 'receptor_seq', 'peptide': 'ligand_seq', 'label': 'label'})

    # 清洗
    df = df.dropna(subset=['receptor_seq', 'ligand_seq', 'label'])
    df['label'] = df['label'].astype(int)
    df['task_type'] = task_type
    
    # 生成 Q/A
    df['question'] = BEST_PROMPT_TEMPLATE.format(rec=TOKEN_REC_ID, lig=TOKEN_LIG_ID)
    df['answer'] = df['label'].apply(lambda x: "Yes" if x == 1 else "No")
    
    return df[['receptor_seq', 'ligand_seq', 'label', 'task_type', 'question', 'answer']]

def stratified_sample(df, ratio, seed=42):
    """对 DataFrame 进行分层采样"""
    if ratio >= 1.0: return df
    
    # 尝试按 label 分层
    try:
        _, subset_df = train_test_split(df, test_size=ratio, stratify=df['label'], random_state=seed)
        return subset_df
    except ValueError:
        print(f"⚠️ Stratified sampling failed for size {len(df)} with ratio {ratio}. Fallback to random.")
        return df.sample(frac=ratio, random_state=seed)

# ==========================================
# 3. 主流程
# ==========================================
def main():
    # 1. 读取全量数据 (Pandas)
    print("Step 1: Loading Full Datasets...")
    df_hla_train = load_and_process_csv(HLA_TRAIN_PATH, 'HLA')
    df_tcr_train = load_and_process_csv(TCR_TRAIN_PATH, 'TCR')
    
    # 合并全量训练集
    df_train_full = pd.concat([df_hla_train, df_tcr_train], ignore_index=True)
    print(f"✅ Full Train Size: {len(df_train_full)}")
    
    # 读取全量验证集 (只读一次，所有子集共用)
    df_hla_val = load_and_process_csv(HLA_VAL_PATH, 'HLA')
    df_tcr_val = load_and_process_csv(TCR_VAL_PATH, 'TCR')
    df_val_full = pd.concat([df_hla_val, df_tcr_val], ignore_index=True)
    # 打乱验证集
    df_val_full = df_val_full.sample(frac=1.0, random_state=1314).reset_index(drop=True)
    ds_val_full = Dataset.from_pandas(df_val_full)
    print(f"✅ Full Validation Size: {len(ds_val_full)}")

    # 2. 循环生成子集
    print("\nStep 2: Generating Subsets...")
    for ratio in RATIOS:
        subset_name = f"subset_{int(ratio*100)}pct"
        output_dir = os.path.join(OUTPUT_ROOT, subset_name)
        
        print(f"\n👉 Processing ratio: {ratio} ({subset_name})")
        
        # 对 HLA 和 TCR 分别采样，然后再合并，确保 Task 比例也不变
        # (比直接对混合数据采样更稳，因为正负样本比例在不同任务中可能不同)
        df_hla_sub = stratified_sample(df_hla_train, ratio)
        df_tcr_sub = stratified_sample(df_tcr_train, ratio)
        
        # 合并并打乱
        df_train_sub = pd.concat([df_hla_sub, df_tcr_sub], ignore_index=True)
        df_train_sub = df_train_sub.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        # 统计信息
        print(f"   Size: {len(df_train_sub)}")
        print(f"   HLA Count: {len(df_hla_sub)} | TCR Count: {len(df_tcr_sub)}")
        print(f"   Pos Rate: {df_train_sub['label'].mean():.2%} (Target: ~{df_train_full['label'].mean():.2%})")
        
        # 转为 HF Dataset
        ds_train_sub = Dataset.from_pandas(df_train_sub)
        
        # 构建 DatasetDict
        dataset_dict = DatasetDict({
            'train': ds_train_sub,
            'validation': ds_val_full # 共用全量验证集
        })
        
        # 保存
        print(f"   Saving to {output_dir}...")
        dataset_dict.save_to_disk(output_dir)
        print("   ✅ Done.")

if __name__ == "__main__":
    main()
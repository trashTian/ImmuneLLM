import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

# ==========================================
# 1. 配置区域
# ==========================================
# UnifyImmune 原始数据路径 (Training Set)
HLA_TRAIN_PATH = "/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/train_fold_1.csv"
TCR_TRAIN_PATH = "/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/train_fold_1.csv"

# 输出根目录 (CSV 文件夹)
OUTPUT_ROOT = "/mnt/lustre/guopeijin/Immune_LLM/code/data_prepare/stage2_data_efficiency/original_csv"

# 采样比例列表
RATIOS = [0.01, 0.05, 0.1, 0.2, 0.25]

# ==========================================
# 2. 核心函数
# ==========================================
def load_and_standardize_csv(file_path, task_type):
    """读取 CSV，标准化列名，返回 DataFrame"""
    print(f"   Loading {task_type}: {file_path}")
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)

    # 统一列名: receptor_seq, ligand_seq, label
    # if task_type == 'HLA':
    #     col_map = {'HLA': 'receptor_seq', 'peptide': 'ligand_seq', 'label': 'label'}
    # else:
    #     col_map = {'tcr': 'receptor_seq', 'peptide': 'ligand_seq', 'label': 'label'}
    
    # # 重命名
    # df = df.rename(columns=col_map)
    
    # # 确保列名正确
    # required_cols = ['receptor_seq', 'ligand_seq', 'label']
    # if not all(col in df.columns for col in required_cols):
    #     print(f"❌ Missing columns in {task_type}. Found: {df.columns}")
    #     return None

    # 清洗
    # df = df.dropna(subset=required_cols)
    df['label'] = df['label'].astype(int)
    # df['task_type'] = task_type # 既然分开了，task_type 列可选，但我还是留着方便追溯
    
    return df

def stratified_sample(df, ratio, seed=42):
    """
    对 DataFrame 进行分层采样 (按 label 分层)。
    train_size = ratio (e.g., 0.1 取 10%)
    """
    if ratio >= 1.0: return df
    
    try:
        # train_size=ratio 取出指定比例的子集
        subset_df, _ = train_test_split(df, train_size=ratio, stratify=df['label'], random_state=seed)
        return subset_df
    except ValueError:
        print(f"⚠️ Stratified sampling failed (maybe too few samples). Fallback to random.")
        return df.sample(frac=ratio, random_state=seed)

# ==========================================
# 3. 主流程
# ==========================================
def main():
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)
        
    # 1. 读取全量数据
    print("Step 1: Loading Full Datasets...")
    df_hla_train = load_and_standardize_csv(HLA_TRAIN_PATH, 'HLA')
    df_tcr_train = load_and_standardize_csv(TCR_TRAIN_PATH, 'TCR')
    
    # 打印全量统计
    print(f"   Full HLA Count: {len(df_hla_train)}")
    print(f"   Full TCR Count: {len(df_tcr_train)}")

    # 2. 循环生成子集 CSV
    print("\nStep 2: Generating Separated Subset CSVs...")
    for ratio in RATIOS:
        pct_str = f"{int(ratio*100)}pct"
        
        # --- Process HLA ---
        print(f"\n👉 Processing HLA ratio: {ratio}")
        df_hla_sub = stratified_sample(df_hla_train, ratio)
        hla_out_path = os.path.join(OUTPUT_ROOT, f"train_subset_HLA_{pct_str}.csv")
        df_hla_sub.to_csv(hla_out_path, index=False)
        print(f"   Saved HLA: {len(df_hla_sub)} -> {hla_out_path}")
        print(f"   Pos Rate: {df_hla_sub['label'].mean():.2%}")

        # --- Process TCR ---
        print(f"👉 Processing TCR ratio: {ratio}")
        df_tcr_sub = stratified_sample(df_tcr_train, ratio)
        tcr_out_path = os.path.join(OUTPUT_ROOT, f"train_subset_TCR_{pct_str}.csv")
        df_tcr_sub.to_csv(tcr_out_path, index=False)
        print(f"   Saved TCR: {len(df_tcr_sub)} -> {tcr_out_path}")
        print(f"   Pos Rate: {df_tcr_sub['label'].mean():.2%}")

    print("\n✅ All done.")

if __name__ == "__main__":
    main()
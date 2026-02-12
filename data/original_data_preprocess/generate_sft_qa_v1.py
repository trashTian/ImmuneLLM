import pandas as pd
from datasets import Dataset, concatenate_datasets, DatasetDict
import os
import numpy as np

# ==========================================
# 1. 配置区域
# ==========================================
# 训练集路径  生成unifyimmune数据
# HLA_TRAIN_PATH = "/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/train_fold_1.csv"
# TCR_TRAIN_PATH = "/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/train_fold_1.csv"

# # 验证集路径
# HLA_VAL_PATH = "/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/val_fold_1.csv"
# TCR_VAL_PATH = "/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/val_fold_1.csv"

# # [修改] 输出路径  标准sftqa
# OUTPUT_PATH = "/mnt/lustre/guopeijin/Immune_LLM/code/data_prepare/stage2_sft_baseline"

# 生成 deepantigen 数据
HLA_TRAIN_PATH = "/mnt/lustre/guopeijin/Immune_LLM/code/data_prepare/deepantigen_data/deepantigen_hla_train.csv"
TCR_TRAIN_PATH = "/mnt/lustre/guopeijin/Immune_LLM/code/data_prepare/deepantigen_data/tcr_train_data.csv"

# 验证集路径
HLA_VAL_PATH = "/mnt/lustre/guopeijin/Immune_LLM/code/data_prepare/deepantigen_data/deepantigen_hla_val.csv"
TCR_VAL_PATH = "/mnt/lustre/guopeijin/Immune_LLM/code/data_prepare/deepantigen_data/tcr_val_data.csv"

# [修改] 输出路径
OUTPUT_PATH = "/mnt/lustre/guopeijin/Immune_LLM/code/data_prepare/stage2_sft_deepantigen"

# Llama 3.2 保留 Token 映射 (必须与 Stage 1 一致)
TOKEN_REC_ID = "<|reserved_special_token_70|>" 
TOKEN_LIG_ID = "<|reserved_special_token_71|>"

# ==========================================
# 2. Prompt 模板 (固定最佳模板)
# ==========================================
BEST_PROMPT_TEMPLATE = (
    "Analyze the structural compatibility between {rec} and {lig}.\n"
    "Prediction of stable complex formation (Yes/No):"
)

def get_answer_text(label):
    return "Yes" if label == 1 else "No"

# ==========================================
# 3. 处理逻辑 (保持不变，通用函数)
# ==========================================
def process_dataset(file_path, task_type):
    print(f"Processing {task_type} from {file_path}...")
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)

    # 1. 统一列名
    if task_type == 'HLA':
        # 兼容可能的列名差异
        if 'HLA' in df.columns:
            df = df.rename(columns={'HLA': 'receptor_seq', 'peptide': 'ligand_seq', 'label': 'label'})
        elif 'receptor_seq' not in df.columns:
            print(f"Warning: HLA columns not recognized in {file_path}")
    else:
        if 'tcr' in df.columns:
            df = df.rename(columns={'tcr': 'receptor_seq', 'peptide': 'ligand_seq', 'label': 'label'})
        elif 'receptor_seq' not in df.columns:
            print(f"Warning: TCR columns not recognized in {file_path}")

    # 2. 基础清洗
    original_len = len(df)
    # 确保列存在再 dropna
    req_cols = ['receptor_seq', 'ligand_seq', 'label']
    if not all(col in df.columns for col in req_cols):
        print(f"❌ Missing required columns in {file_path}. Columns found: {df.columns}")
        return None
        
    df = df.dropna(subset=req_cols)
    df['label'] = df['label'].astype(int)
    # print(f"  - Dropped {original_len - len(df)} rows (NaN). Current: {len(df)}")

    # 3. 生成 Question 和 Answer
    def format_row(row):
        question = BEST_PROMPT_TEMPLATE.format(rec=TOKEN_REC_ID, lig=TOKEN_LIG_ID)
        answer = get_answer_text(row['label'])
        return pd.Series([question, answer])

    df[['question', 'answer']] = df.apply(format_row, axis=1)
    
    # 4. 整理列
    df['task_type'] = task_type
    final_df = df[['receptor_seq', 'ligand_seq', 'label', 'task_type', 'question', 'answer']]
    
    print(f"  -> Generated {len(final_df)} samples for {task_type}.")
    return Dataset.from_pandas(final_df)

# ==========================================
# 4. 主程序
# ==========================================
def main():
    # --- 1. 处理训练集 ---
    print("\n=== Generating TRAIN Set ===")
    ds_hla_train = process_dataset(HLA_TRAIN_PATH, 'HLA')
    ds_tcr_train = process_dataset(TCR_TRAIN_PATH, 'TCR')
    
    valid_train = [d for d in [ds_hla_train, ds_tcr_train] if d is not None]
    if not valid_train:
        print("❌ No training data generated.")
        return
    
    full_train = concatenate_datasets(valid_train)
    full_train = full_train.shuffle(seed=921) # 打乱训练集
    full_train = full_train.shuffle(seed=1314) # 打乱训练集
    print(f"✅ Final Train Size: {len(full_train)}")

    # --- 2. 处理验证集 ---
    print("\n=== Generating VALIDATION Set ===")
    ds_hla_val = process_dataset(HLA_VAL_PATH, 'HLA')
    ds_tcr_val = process_dataset(TCR_VAL_PATH, 'TCR')
    
    valid_val = [d for d in [ds_hla_val, ds_tcr_val] if d is not None]
    if not valid_val:
        print("❌ No validation data generated.")
        return

    full_val = concatenate_datasets(valid_val)
    full_val = full_val.shuffle(seed=921) # 打乱验证集
    full_val = full_val.shuffle(seed=1314) # 打乱验证集
    print(f"✅ Final Validation Size: {len(full_val)}")

    # --- 3. 构建 DatasetDict 并保存 ---
    dataset_dict = DatasetDict({
        'train': full_train,
        'validation': full_val
    })
    
    dataset_dict.save_to_disk(OUTPUT_PATH)
    print(f"\n🎉 DatasetDict saved to: {OUTPUT_PATH}")
    print(f"Structure: {dataset_dict}")

    # ==========================================
    # 5. 校验环节
    # ==========================================
    print("\n" + "="*40)
    print("🔬 Data Inspection (Sanity Check)")
    print("="*40)
    
    # 检查 Train
    print("--- Train Sample ---")
    item = dataset_dict['train'][0]
    print(f"Task: {item['task_type']} | Label: {item['label']}")
    print(f"Q: {item['question']}")
    print(f"A: {item['answer']}")
    
    # 检查 Val
    print("\n--- Validation Sample ---")
    item = dataset_dict['validation'][0]
    print(f"Task: {item['task_type']} | Label: {item['label']}")
    print(f"A: {item['answer']}")

    # 检查特殊 Token
    if TOKEN_REC_ID not in item['question']:
        print("⚠️ WARNING: Special tokens missing!")

if __name__ == "__main__":
    main()

    """
    === Generating TRAIN Set ===
Processing HLA from /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/train_fold_1.csv...
  -> Generated 803145 samples for HLA.
Processing TCR from /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/train_fold_1.csv...
  -> Generated 206689 samples for TCR.
✅ Final Train Size: 1009834

=== Generating VALIDATION Set ===
Processing HLA from /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/val_fold_1.csv...
  -> Generated 200787 samples for HLA.
Processing TCR from /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/val_fold_1.csv...
  -> Generated 51673 samples for TCR.
✅ Final Validation Size: 252460
Saving the dataset (1/1 shards): 100%|██████████████████████████████████████████████████████████████████| 1009834/1009834 [00:06<00:00, 151848.15 examples/s]
Saving the dataset (1/1 shards): 100%|████████████████████████████████████████████████████████████████████| 252460/252460 [00:01<00:00, 148337.95 examples/s]

🎉 DatasetDict saved to: /mnt/lustre/guopeijin/Immune_LLM/code/data_prepare/stage2_sft_baseline
Structure: DatasetDict({
    train: Dataset({
        features: ['receptor_seq', 'ligand_seq', 'label', 'task_type', 'question', 'answer'],
        num_rows: 1009834
    })
    validation: Dataset({
        features: ['receptor_seq', 'ligand_seq', 'label', 'task_type', 'question', 'answer'],
        num_rows: 252460
    })
})

========================================
🔬 Data Inspection (Sanity Check)
========================================
--- Train Sample ---
Task: HLA | Label: 0
Q: Analyze the structural compatibility between <|reserved_special_token_70|> and <|reserved_special_token_71|>.
Prediction of stable complex formation (Yes/No):
A: No

--- Validation Sample ---
Task: HLA | Label: 0
A: No
    """
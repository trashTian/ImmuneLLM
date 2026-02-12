import pandas as pd
from datasets import Dataset, concatenate_datasets
import os
import numpy as np

# ==========================================
# 1. 配置区域
# ==========================================
HLA_CSV_PATH = "/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/dataset.csv"
TCR_CSV_PATH = "/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/dataset.csv"

# [修改] 输出路径：建议标记为 _baseline 或 _direct_sft 以区分 CoT 版本
OUTPUT_PATH = "/mnt/lustre/guopeijin/Immune_LLM/code/data_prepare/stage2_sft_baseline_direct"

# Llama 3.2 保留 Token 映射 (必须与 Stage 1 一致)
TOKEN_REC_ID = "<|reserved_special_token_70|>" 
TOKEN_LIG_ID = "<|reserved_special_token_71|>"

# ==========================================
# 2. Prompt 模板 (消融实验核心：固定最佳模板)
# ==========================================
# [关键修改] 不再使用随机列表，而是使用你在推理中验证效果最好的那个模板。
# 这样可以保证 SFT 训练时的分布与你最终推理时的分布完全一致。

BEST_PROMPT_TEMPLATE = (
    "Analyze the structural compatibility between {rec} and {lig}.\n"
    "Prediction of stable complex formation (Yes/No):"
)

# [关键修改] Answer 统一化
# SFT 基准不需要多样性回答，只需要模型精准输出 Label 对应的 Token
def get_answer_text(label):
    return "Yes" if label == 1 else "No"

# ==========================================
# 3. 处理逻辑
# ==========================================

def process_dataset(file_path, task_type):
    print(f"Processing {task_type} from {file_path}...")
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    df = pd.read_csv(file_path)

    # 1. 统一列名
    if task_type == 'HLA':
        # HLA 数据集通常列名: HLA, peptide, label
        df = df.rename(columns={'HLA': 'receptor_seq', 'peptide': 'ligand_seq', 'label': 'label'})
    else:
        # TCR 数据集通常列名: tcr, peptide, label
        df = df.rename(columns={'tcr': 'receptor_seq', 'peptide': 'ligand_seq', 'label': 'label'})

    # 2. 基础清洗
    original_len = len(df)
    df = df.dropna(subset=['receptor_seq', 'ligand_seq', 'label'])
    df['label'] = df['label'].astype(int)
    print(f"  - Dropped {original_len - len(df)} rows (NaN). Current: {len(df)}")

    # 3. 生成 Question 和 Answer
    # 使用 apply 直接生成，不进行随机采样，保证严谨性
    print("  - Generating Standard QA pairs...")
    
    def format_row(row):
        # 1. 构建 Question (User Prompt)
        # 这里只生成 User 的内容，System Prompt 和 Chat Template 格式交给训练脚本处理
        question = BEST_PROMPT_TEMPLATE.format(rec=TOKEN_REC_ID, lig=TOKEN_LIG_ID)
        
        # 2. 构建 Answer (Assistant Response)
        answer = get_answer_text(row['label'])
        
        return pd.Series([question, answer])

    df[['question', 'answer']] = df.apply(format_row, axis=1)
    
    # 4. 整理列
    df['task_type'] = task_type
    # 保留 raw sequences 是为了 ESM 编码
    # 保留 question/answer 是为了 Llama 训练
    final_df = df[['receptor_seq', 'ligand_seq', 'label', 'task_type', 'question', 'answer']]
    
    print(f"  - Final count for {task_type}: {len(final_df)}")
    return Dataset.from_pandas(final_df)

# ==========================================
# 4. 主程序
# ==========================================
def main():
    # 分别处理
    ds_hla = process_dataset(HLA_CSV_PATH, 'HLA')
    ds_tcr = process_dataset(TCR_CSV_PATH, 'TCR')
    
    valid_ds = [d for d in [ds_hla, ds_tcr] if d is not None]
    if not valid_ds:
        print("❌ No datasets generated.")
        return

    # 合并
    full_ds = concatenate_datasets(valid_ds)
    
    # 全局打乱 (非常重要，防止 Batch 内全是 HLA 或全是 TCR)
    full_ds = full_ds.shuffle(seed=921)
    
    print(f"\n✅ Total merged samples: {len(full_ds)}")
    
    # 保存
    full_ds.save_to_disk(OUTPUT_PATH)
    print(f"✅ Dataset saved to: {OUTPUT_PATH}")

    # ==========================================
    # 5. 校验环节
    # ==========================================
    print("\n" + "="*40)
    print("🔬 Data Inspection (Sanity Check)")
    print("="*40)
    
    # 抽取前 3 条看一眼
    for i in range(3):
        item = full_ds[i]
        print(f"\n[Sample {i}] Task: {item['task_type']} | Label: {item['label']}")
        print(f"Q: {item['question']}")
        print(f"A: {item['answer']}")
        
        # 检查是否包含特殊 Token
        if TOKEN_REC_ID not in item['question'] or TOKEN_LIG_ID not in item['question']:
            print("⚠️ WARNING: Special tokens missing in question!")
        
        # 检查 Answer 是否规范
        if item['answer'] not in ["Yes", "No"]:
            print(f"⚠️ WARNING: Unexpected answer format: {item['answer']}")

if __name__ == "__main__":
    main()
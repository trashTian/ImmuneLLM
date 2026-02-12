import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, accuracy_score, matthews_corrcoef, f1_score,
    average_precision_score, precision_score, recall_score, confusion_matrix,precision_recall_curve
)

# 导入项目模块
from src.io_utils_fused import CSVDataset_test, Collater_test
from src.constants import PAD, PROTEIN_ALPHABET
# [修改1] 导入 task9 (TCR任务模型)
from src.model_fused import task9 

# ================= 核心指标计算函数 =================
def calculate_metrics(np_labels, np_scores):
    try: auc = roc_auc_score(np_labels, np_scores)
    except: auc = 0.5
    
    pr_auc = average_precision_score(np_labels, np_scores)
    precision, recall, thresholds = precision_recall_curve(np_labels, np_scores)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    best_threshold = thresholds[np.argmax(f1_scores[:-1])] if len(thresholds) > 0 else 0.5
    np_preds = (np_scores > best_threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(np_labels, np_preds).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "AUC": auc,
        "Accuracy": accuracy_score(np_labels, np_preds),
        "MCC": matthews_corrcoef(np_labels, np_preds),
        "F1": f1_score(np_labels, np_preds),
        "PR_AUC": pr_auc,
        "Specificity": spec,
        "Precision": precision_score(np_labels, np_preds, zero_division=0),
        "Recall": recall_score(np_labels, np_preds, zero_division=0)
    }

def main(args):
    # ================= 1. 初始化设置 =================
    BOOTSTRAP_ROUNDS = args.n_bootstraps
    METRIC_ORDER = ["AUC", "Accuracy", "MCC", "F1", "PR_AUC", "Specificity", "Precision", "Recall"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 T-SCAPE (TCR Task) Evaluation Start on {device}")
    
    # 加载原始数据
    print(f"Loading raw data from: {args.csv_path}")
    df_raw = pd.read_csv(args.csv_path)
    
    # [修改2] 检查 CDR3b 列
    required_cols = ['peptide', 'label', 'CDR3b']
    for col in required_cols:
        if col not in df_raw.columns:
            raise ValueError(f"Missing required column for TCR task: {col}")
            
    # 填充缺失值
    df_raw['peptide'] = df_raw['peptide'].fillna("")
    df_raw['CDR3b'] = df_raw['CDR3b'].fillna("")
    # 辅助列 (pseudo 在这里可能为空，但 Dataset 需要这列存在)
    if 'pseudo' not in df_raw.columns:
        df_raw['pseudo'] = ""
    else:
        df_raw['pseudo'] = df_raw['pseudo'].fillna("")
        
    df_raw['task'] = 1 
    df_raw['mhc'] = ""

    # ================= [修改3] 双重长度过滤 (Peptide & CDR3b) =================
    # T-SCAPE 要求 Peptide 和 TCR CDR3b 长度都不超过 20
    initial_len = len(df_raw)
    
    # 确保是字符串类型
    df_raw['peptide'] = df_raw['peptide'].astype(str)
    df_raw['CDR3b'] = df_raw['CDR3b'].astype(str)
    
    # 过滤逻辑：保留 peptide <= 20 且 CDR3b <= 20 的行
    df_raw = df_raw[
        (df_raw['peptide'].apply(len) <= 20) & 
        (df_raw['CDR3b'].apply(len) <= 20)
    ]
    filtered_len = len(df_raw)
    
    if initial_len != filtered_len:
        print(f"⚠️ Warning: Filtered {initial_len - filtered_len} samples (Peptide or CDR3b > 20 AA). Remaining: {filtered_len}")
    # =========================================================================

    # ================= 2. 加载模型 (task9) =================
    print(f"Loading model from: {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    # [修改4] 初始化 task9
    d_model = 280
    embedding_dim = 280
    model = task9(
        d_model=d_model, n_tokens=29, kernel_size=1, n_layers=6, 
        d_embedding=embedding_dim, r=1, mask_condition=False
    )
    
    # 加载权重
    ckpt = torch.load(args.model_path, map_location=device)
    # 这里的 key 是 task9_encoder / task9_decoder
    model.shared_encoder.load_state_dict({k.replace('shared_encoder.', ''): v for k, v in ckpt.items() if 'shared_encoder.' in k})
    model.task9_encoder.load_state_dict({k.replace('task9_encoder.', ''): v for k, v in ckpt.items() if 'task9_encoder.' in k})
    model.task9_decoder.load_state_dict({k.replace('task9_decoder.', ''): v for k, v in ckpt.items() if 'task9_decoder.' in k})
    
    model = model.to(device)
    model.eval()

    # ================= 3. 严格的 Bootstrap 循环 =================
    print(f"\nStarting {BOOTSTRAP_ROUNDS} Bootstrap rounds (Resample -> Inference)...")
    
    ds_metrics = [] 

    for r in range(BOOTSTRAP_ROUNDS):
        print(f"   🔄 Round {r+1}/{BOOTSTRAP_ROUNDS} ...")
        
        # 1. 重采样
        df_sample = df_raw.sample(frac=1.0, replace=True, random_state=r)
        
        # 2. 重置索引 (防止 KeyError)
        df_sample = df_sample.reset_index(drop=True)
        
        # 构建 DataLoader
        test_dataset = CSVDataset_test(df_sample)
        inf_collator = Collater_test(alphabet=PROTEIN_ALPHABET, pad=True, backwards=False, pad_token=PAD)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=inf_collator)
        
        preds, targets = [], []
        
        # 推理循环
        with torch.no_grad():
            for i, (src, m1, m2, tcr, frac, p_lens, mhcs) in enumerate(tqdm(test_loader, leave=False, desc="Inference")):
                src, m1, m2, tcr = src.to(device), m1.to(device), m2.to(device), tcr.to(device)
                
                # [修改5] Task ID = 9
                output = model(src, m1, m2, tcr=tcr, task=[9])
                
                probs = torch.sigmoid(output[-1]).cpu().numpy()
                preds.extend(probs)
                targets.extend(frac) # 直接 extend list

        m = calculate_metrics(np.array(targets), np.array(preds))
        ds_metrics.append(m)
        print(f"      Result: AUC={m['AUC']:.4f}, F1={m['F1']:.4f}")

    # ================= 4. 结果汇总 =================
    df_res_detail = pd.DataFrame(ds_metrics)[METRIC_ORDER]
    detail_csv_path = str(Path(args.output_csv).with_suffix('.detail.csv'))
    df_res_detail.to_csv(detail_csv_path, index=False)
    
    mean_vals = df_res_detail.mean()
    std_vals = df_res_detail.std()
    
    print("\n" + "="*60)
    print(f"📊 Final TCR Task Results ({BOOTSTRAP_ROUNDS} Rounds)")
    print("="*60)
    
    summary_results = {}
    for col in METRIC_ORDER:
        mean_v = mean_vals[col]
        std_v = std_vals[col]
        print(f"{col:<15}: {mean_v:.4f} ± {std_v:.4f}")
        summary_results[f"{col}_Mean"] = mean_v
        summary_results[f"{col}_Std"] = std_v
        
    df_summary = pd.DataFrame([summary_results])
    df_summary.to_csv(args.output_csv, index=False)
    print(f"\n✅ Saved to: {args.output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='/mnt/lustre/guopeijin/Immune_LLM/code/baselines/titanian/T-SCAPE-main/test_datas/tcr/covid_set_tcr.csv', help="Must contain peptide, CDR3b, label")
    
    # 默认路径修改为您提供的 TCR 模型路径'/mnt/lustre/guopeijin/Immune_LLM/code/baselines/titanian/T-SCAPE-main/checkpoints/models/Uni_el-fused_ADV1.0_0.pt'
    parser.add_argument('--model_path', type=str, 
                        default='/mnt/lustre/guopeijin/Immune_LLM/code/baselines/titanian/T-SCAPE-main/checkpoints/models/Uni_el-fused_ADV1.0_0.pt')
                        
    parser.add_argument('--output_csv', type=str, default="res_covid_set_tcr.csv")
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--n_bootstraps', type=int, default=5)

    args = parser.parse_args()
    main(args)
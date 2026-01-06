import os
import torch
import pandas as pd
import numpy as np
import argparse
import esm
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, matthews_corrcoef, 
                             precision_recall_curve, auc)

# 导入你提供的工具库
from FT_utils import TCRBindingDataset, custom_collate_fn

# ==========================================
# 1. 模型定义 (需与训练代码完全一致)
# ==========================================
class ESM2ForBindingPrediction(torch.nn.Module):
    def __init__(self, esm_model):
        super().__init__()
        self.esm = esm_model
        self.hidden_size = esm_model.embed_dim

        # 预测头结构必须完全匹配
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, 2560), 
            nn.LayerNorm(2560), 
            nn.ReLU(),
            nn.Linear(2560, 256), 
            nn.LayerNorm(256), 
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, tokens):
        # 推理时一般不需要 steering_vectors，除非你要做特定分析
        outputs = self.esm(
            tokens,
            repr_layers=[36], 
            return_contacts=False
        )
        last_hidden_state = outputs['representations'][36]
        features = last_hidden_state[:, 0, :]
        predictions = self.prediction_head(features).squeeze(-1)
        return predictions

# ==========================================
# 2. 核心功能函数
# ==========================================

def load_model(model_path, device):
    """加载模型架构和权重"""
    print(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # 1. 加载 ESM2 基座
    print(f"Loading ESM2 base: {config['model_name']}...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(config['model_name'])
    batch_converter = alphabet.get_batch_converter()
    
    # 2. 初始化预测模型
    pred_model = ESM2ForBindingPrediction(esm_model=model)
    
    # 3. 加载微调权重 (处理可能的 module. 前缀)
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    pred_model.load_state_dict(new_state_dict)
    
    pred_model.to(device)
    pred_model.eval()
    
    return pred_model, batch_converter, config

def calculate_metrics(y_true, y_pred, y_prob):
    """计算全套指标"""
    # 基础指标
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0) # Sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # AUC & PR-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall_curve, precision_curve)
    except ValueError:
        roc_auc = 0.0
        pr_auc = 0.0

    # Specificity
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    else:
        specificity = 0.0

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Sensitivity (Recall)": rec,
        "Specificity": specificity,
        "F1 Score": f1,
        "MCC": mcc,
        "AUC": roc_auc,
        "PR-AUC": pr_auc
    }

def run_inference(model, dataloader, device, batch_converter):
    """执行一次完整的推理"""
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for samples, labels in tqdm(dataloader, desc="Inferencing", leave=False):
            # samples 是由 custom_collate_fn 返回的 [(name, seq), ...] 列表
            # labels 是列表
            
            # 使用 ESM batch_converter 将 samples 转为 tokens
            # samples 格式为 list of (id, seq)，符合 converter 要求
            _, _, tokens = batch_converter(samples)
            tokens = tokens.to(device)
            
            logits = model(tokens)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)
            
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# ==========================================
# 3. 主程序
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Repeated Inference with Bootstrapping')
    parser.add_argument('--model-path', type=str, required=True, help='Path to .pt model file')
    parser.add_argument('--test-file', type=str, required=True, help='Test dataset CSV')
    parser.add_argument('--output-dir', type=str, default='./results', help='Save results here')
    parser.add_argument('--repeat', type=int, default=5, help='Number of bootstrap repetitions')
    parser.add_argument('--batch-size', type=int, default=2048*3) # 32
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 列名参数
    parser.add_argument('--col1', type=str, default='tcr', help='Column 1 (e.g. TCR)')
    parser.add_argument('--col2', type=str, default='pep', help='Column 2 (e.g. Peptide)')
    parser.add_argument('--col3', type=str, default='label', help='Label column')

    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 加载模型
    model, batch_converter, config = load_model(args.model_path, device)
    
    # 2. 读取原始数据
    print(f"Reading test data from {args.test_file}...")
    original_df = pd.read_csv(args.test_file)
    max_len = config.get("max_seq_length", 1024)
    
    metrics_list = []
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting {args.repeat} rounds of Bootstrap Inference")
    print(f"{'='*60}\n")

    
    
    for i in range(args.repeat):
        print(f"🔄 Round {i+1}/{args.repeat}")
        # [关键步骤] Bootstrap 重采样：有放回地随机采样，生成新的测试分布
        # random_state=i 确保每次采样的种子不同，保证数据差异性
        resampled_df = original_df.sample(frac=1.0, replace=True, random_state=i)
        
        # 构建数据集 (复用 FT_utils)
        dataset = TCRBindingDataset(
            resampled_df, 
            col1=args.col1, 
            col2=args.col2, 
            col3=args.col3,
            max_length=max_len
        )
        
        # 构建 DataLoader (使用 FT_utils 中的 custom_collate_fn)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=4,
            pin_memory=True
        )
        # 推理
        y_true, y_pred, y_prob = run_inference(model, dataloader, device, batch_converter)
        
        # 计算指标
        round_metrics = calculate_metrics(y_true, y_pred, y_prob)
        metrics_list.append(round_metrics)
        
        print(f"   -> AUC: {round_metrics['AUC']:.4f} | F1: {round_metrics['F1 Score']:.4f}")

    # ==========================================
    # 4. 统计与输出
    # ==========================================
    
    df_metrics = pd.DataFrame(metrics_list)
    
    # 保存所有轮次的详细数据
    detail_save_path = os.path.join(args.output_dir, "bootstrap_metrics_details.csv")
    df_metrics.to_csv(detail_save_path, index_label="Round")
    
    print(f"\n{'='*60}")
    print(f"📊 Final Results (Mean ± Std over {args.repeat} bootstrap runs)")
    print(f"{'='*60}")
    
    # 计算平均值和标准差
    means = df_metrics.mean()
    stds = df_metrics.std()
    
    summary_results = {}
    
    # 按特定顺序打印
    display_order = ["AUC", "PR-AUC", "Accuracy", "F1 Score", "MCC", 
                     "Sensitivity (Recall)", "Specificity", "Precision"]
    
    for metric in display_order:
        if metric in means:
            m_val = means[metric]
            s_val = stds[metric]
            result_str = f"{m_val:.4f} ± {s_val:.4f}"
            print(f"{metric:<25}: {result_str}")
            summary_results[metric] = result_str
            
    # 保存最终摘要
    summary_path = os.path.join(args.output_dir, "final_summary_metrics.txt")
    with open(summary_path, "w") as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test File: {args.test_file}\n")
        f.write(f"Bootstrap Rounds: {args.repeat}\n\n")
        for k, v in summary_results.items():
            f.write(f"{k:<25}: {v}\n")
            
    print(f"\n✅ All results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()

    """
    python ESM2_3b_predict.py --model-path /mnt/lustre/guopeijin/Immune_LLM/code/baselines/logs/unifyimmune_HLA_FT/unifyimmune_HLA_FT_ESM2_3b.pt --test-file /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/independent_set.csv --col1 HLA --col2 peptide --col3 label --repeat 5 --device cuda:2
    ============================================================
    � Final Results (Mean ± Std over 5 bootstrap runs)
    ============================================================
    AUC                      : 0.9567 ± 0.0006
    PR-AUC                   : 0.9084 ± 0.0012
    Accuracy                 : 0.9010 ± 0.0010
    F1 Score                 : 0.8485 ± 0.0017
    MCC                      : 0.7752 ± 0.0024
    Sensitivity (Recall)     : 0.8626 ± 0.0016
    Specificity              : 0.9192 ± 0.0012
    Precision                : 0.8350 ± 0.0025

    python ESM2_3b_predict.py --model-path /mnt/lustre/guopeijin/Immune_LLM/code/baselines/logs/unifyimmune_HLA_fixed/unifyimmune_HLA_fixed_ESM2_3b.pt --test-file /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/independent_set.csv --col1 HLA --col2 peptide --col3 label --repeat 5 --device cuda:4
    ============================================================
    � Final Results (Mean ± Std over 5 bootstrap runs)
    ============================================================
    AUC                      : 0.8917 ± 0.0011
    PR-AUC                   : 0.7649 ± 0.0017
    Accuracy                 : 0.8101 ± 0.0012
    F1 Score                 : 0.7424 ± 0.0012
    MCC                      : 0.6082 ± 0.0025
    Sensitivity (Recall)     : 0.8510 ± 0.0025
    Specificity              : 0.7907 ± 0.0008
    Precision                : 0.6584 ± 0.0009


    python ESM2_3b_predict.py --model-path /mnt/lustre/guopeijin/Immune_LLM/code/baselines/logs/unifyimmune_HLA_fixed/unifyimmune_HLA_fixed_ESM2_3b.pt --test-file /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/external_set.csv --col1 HLA --col2 peptide --col3 label --repeat 5 --device cuda:5
    ============================================================
    � Final Results (Mean ± Std over 5 bootstrap runs)
    ============================================================
    AUC                      : 0.8645 ± 0.0012
    PR-AUC                   : 0.8604 ± 0.0008
    Accuracy                 : 0.8035 ± 0.0011
    F1 Score                 : 0.8007 ± 0.0010
    MCC                      : 0.6072 ± 0.0022
    Sensitivity (Recall)     : 0.7886 ± 0.0019
    Specificity              : 0.8184 ± 0.0019
    Precision                : 0.8132 ± 0.0013

    python ESM2_3b_predict.py --model-path /mnt/lustre/guopeijin/Immune_LLM/code/baselines/logs/unifyimmune_tcr_fixed/unifyimmune_tcr_fixed_ESM2_3b.pt --test-file /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/triple_set.csv --col1 tcr --col2 peptide --col3 label --repeat 5 --device cuda:6
    ============================================================
    � Final Results (Mean ± Std over 5 bootstrap runs)
    ============================================================
    AUC                      : 0.7268 ± 0.0021
    PR-AUC                   : 0.6663 ± 0.0025
    Accuracy                 : 0.7190 ± 0.0011
    F1 Score                 : 0.5858 ± 0.0025
    MCC                      : 0.3903 ± 0.0027
    Sensitivity (Recall)     : 0.5094 ± 0.0033
    Specificity              : 0.8531 ± 0.0009
    Precision                : 0.6894 ± 0.0019

    python ESM2_3b_predict.py --model-path /mnt/lustre/guopeijin/Immune_LLM/code/baselines/logs/unifyimmune_tcr_fixed/unifyimmune_tcr_fixed_ESM2_3b.pt --test-file /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/covid_set.csv --col1 tcr --col2 peptide --col3 label --repeat 5 --device cuda:7
    ============================================================
    � Final Results (Mean ± Std over 5 bootstrap runs)
    ============================================================
    AUC                      : 0.4753 ± 0.0006
    PR-AUC                   : 0.4914 ± 0.0013
    Accuracy                 : 0.4926 ± 0.0002
    F1 Score                 : 0.2667 ± 0.0011
    MCC                      : -0.0144 ± 0.0014
    Sensitivity (Recall)     : 0.1835 ± 0.0008
    Specificity              : 0.8052 ± 0.0007
    Precision                : 0.4878 ± 0.0023

    python ESM2_3b_predict.py --model-path /mnt/lustre/guopeijin/Immune_LLM/code/baselines/logs/unifyimmune_tcr_fixed/unifyimmune_tcr_fixed_ESM2_3b.pt --test-file /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/independent_set.csv --col1 tcr --col2 peptide --col3 label --repeat 5 --device cuda:6
    ============================================================
    � Final Results (Mean ± Std over 5 bootstrap runs) cuda 6  
    ============================================================
    AUC                      : 0.8692 ± 0.0007
    PR-AUC                   : 0.7742 ± 0.0030
    Accuracy                 : 0.8109 ± 0.0018
    F1 Score                 : 0.7376 ± 0.0024
    MCC                      : 0.5907 ± 0.0037
    Sensitivity (Recall)     : 0.7610 ± 0.0039
    Specificity              : 0.8377 ± 0.0027
    Precision                : 0.7157 ± 0.0033

    python ESM2_3b_predict.py --model-path /mnt/lustre/guopeijin/Immune_LLM/code/baselines/logs/unifyimmune_HLA_FT/unifyimmune_HLA_FT_ESM2_3b.pt --test-file /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/external_set.csv --col1 HLA --col2 peptide --col3 label --repeat 5 --device cuda:7
    ============================================================
    📊 Final Results (Mean ± Std over 5 bootstrap runs)
    ============================================================
    AUC                      : 0.9178 ± 0.0007
    PR-AUC                   : 0.9289 ± 0.0012
    Accuracy                 : 0.8289 ± 0.0012
    F1 Score                 : 0.8031 ± 0.0015
    MCC                      : 0.6824 ± 0.0018
    Sensitivity (Recall)     : 0.6969 ± 0.0025
    Specificity              : 0.9614 ± 0.0008
    Precision                : 0.9476 ± 0.0010

    

    python ESM2_3b_predict.py --model-path /mnt/lustre/guopeijin/Immune_LLM/code/baselines/logs/unifyimmune_tcr_FT/unifyimmune_tcr_FT_ESM2_3b.pt --test-file /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/independent_set.csv --col1 tcr --col2 peptide --col3 label --repeat 5 --device cuda:5
    ============================================================
    � Final Results (Mean ± Std over 5 bootstrap runs)
    ============================================================
    AUC                      : 0.9063 ± 0.0019
    PR-AUC                   : 0.7959 ± 0.0041
    Accuracy                 : 0.8700 ± 0.0009
    F1 Score                 : 0.8172 ± 0.0017
    MCC                      : 0.7166 ± 0.0022
    Sensitivity (Recall)     : 0.8317 ± 0.0040
    Specificity              : 0.8905 ± 0.0015
    Precision                : 0.8031 ± 0.0006

    python ESM2_3b_predict.py --model-path /mnt/lustre/guopeijin/Immune_LLM/code/baselines/logs/unifyimmune_tcr_FT/unifyimmune_tcr_FT_ESM2_3b.pt --test-file /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/triple_set.csv --col1 tcr --col2 peptide --col3 label --repeat 5 --device cuda:6
    ============================================================
    � Final Results (Mean ± Std over 5 bootstrap runs)
    ============================================================
    AUC                      : 0.7975 ± 0.0009
    PR-AUC                   : 0.6976 ± 0.0028
    Accuracy                 : 0.7780 ± 0.0008
    F1 Score                 : 0.6772 ± 0.0010
    MCC                      : 0.5237 ± 0.0013
    Sensitivity (Recall)     : 0.5967 ± 0.0023
    Specificity              : 0.8941 ± 0.0014
    Precision                : 0.7828 ± 0.0019

    python ESM2_3b_predict.py --model-path /mnt/lustre/guopeijin/Immune_LLM/code/baselines/logs/unifyimmune_tcr_FT/unifyimmune_tcr_FT_ESM2_3b.pt --test-file /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/covid_set.csv --col1 tcr --col2 peptide --col3 label --repeat 5 --device cuda:7


    """

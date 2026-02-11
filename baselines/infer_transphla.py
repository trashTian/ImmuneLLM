import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, accuracy_score, matthews_corrcoef, 
                             f1_score, average_precision_score, confusion_matrix, 
                             precision_score, recall_score)
from sklearn.utils import resample
from tqdm import tqdm

# 从原本的脚本中导入必要的类
# 确保 TransPHLA.py 在同一目录下
from TransPHLA import Config, Tokenizer, TransPHLA, PHLADataset, set_seed

def calculate_metrics(y_true, y_prob, threshold=0.5):
    """计算单个批次的各项指标"""
    y_pred = [1 if p > threshold else 0 for p in y_prob]
    
    try:
        # 基础指标
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # 防止分母为0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # 收集所有指标
        metrics = {
            'auc': roc_auc_score(y_true, y_prob),
            'accuracy': accuracy_score(y_true, y_pred),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'pr_auc': average_precision_score(y_true, y_prob),
            'specificity': specificity,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0) # Recall 即 Sensitivity
        }
    except ValueError as e:
        print(f"Error calculating metrics (possibly only one class in sample): {e}")
        return None
        
    return metrics

def predict(model, loader, device):
    """模型推理函数"""
    model.eval()
    y_true_list = []
    y_prob_list = []
    
    with torch.no_grad():
        for pep, hla, label in loader:
            pep = pep.to(device)
            hla = hla.to(device)
            
            # 模型前向传播
            outputs, _ = model(pep, hla)
            # 获取属于类别 1 的概率
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            y_true_list.extend(label.cpu().numpy())
            y_prob_list.extend(probs.cpu().numpy())
            
    return np.array(y_true_list), np.array(y_prob_list)

def bootstrap_evaluation(model, df, tokenizer, config, n_bootstraps=5):
    """执行 Bootstrap 采样并评估"""
    # 存储每次采样的结果
    bootstrap_results = {
        'auc': [], 'accuracy': [], 'mcc': [], 'f1': [], 
        'pr_auc': [], 'specificity': [], 'precision': [], 'recall': []
    }
    
    print(f"Starting {n_bootstraps}-fold bootstrap evaluation...")
    
    for i in range(n_bootstraps):
        # 1. 有放回重采样 (Resample with replacement)
        # 保持随机种子不同以获得不同的样本，但总体可复现
        df_resampled = resample(df, replace=True, n_samples=len(df), random_state=config.seed + i)
        
        # 2. 构建数据集和加载器
        dataset = PHLADataset(df_resampled, tokenizer, config)
        loader = DataLoader(dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=0)
        
        # 3. 推理
        y_true, y_prob = predict(model, loader, config.device)
        
        # 4. 计算指标
        metrics = calculate_metrics(y_true, y_prob, config.threshold)
        
        if metrics:
            for k, v in metrics.items():
                bootstrap_results[k].append(v)
        
        print(f"  Bootstrap {i+1}/{n_bootstraps} - AUC: {metrics['auc']:.4f}")

    # 5. 汇总统计 (Mean ± Std)
    final_stats = {}
    for k, v in bootstrap_results.items():
        mean_val = np.mean(v)
        std_val = np.std(v)
        final_stats[k] = f"{mean_val:.4f} ± {std_val:.4f}"
        
    return final_stats

def main():
    # ================= 配置路径 =================
    model_path = "/mnt/lustre/guopeijin/Immune_LLM/code/baselines/model/pHLAIformer/best_model.pth"
    
    test_files = {
        "Independent Set": "/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/independent_set.csv",
        "External Set": "/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/external_set.csv"
    }
    # ===========================================

    # 1. 初始化配置和工具
    config = Config()
    # 根据情况调整 batch size，推理时显存占用小，可以调大
    config.batch_size = 2048 
    set_seed(config.seed)
    
    # 强制检查设备，确保使用 GPU (如果可用)
    print(f"Using device: {config.device}")

    # 2. 初始化 Tokenizer (注意这里会触发我们之前修复的逻辑)
    tokenizer = Tokenizer(config.vocab_path)
    config.vocab_size = len(tokenizer)

    # 3. 加载模型
    print(f"Loading model from {model_path}...")
    model = TransPHLA(config)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=config.device)
        model.load_state_dict(state_dict)
        model.to(config.device)
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 4. 在各个测试集上运行评估
    all_results = []
    
    for name, csv_path in test_files.items():
        print(f"\n{'='*20} Evaluating on {name} {'='*20}")
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}, skipping...")
            continue
            
        # 读取 CSV (注意 index_col=False 修复之前的问题)
        df = pd.read_csv(csv_path, index_col=False)
        print(f"Data shape: {df.shape}")
        
        # 运行 Bootstrap 评估
        stats = bootstrap_evaluation(model, df, tokenizer, config, n_bootstraps=5)
        
        # 格式化输出结果
        print(f"\nResults for {name}:")
        row = {'Dataset': name}
        for k, v in stats.items():
            print(f"{k.ljust(12)}: {v}")
            row[k] = v
        all_results.append(row)

    # 5. 保存最终结果到 CSV (可选)
    results_df = pd.DataFrame(all_results)
    # 调整列顺序
    cols = ['Dataset', 'auc', 'accuracy', 'mcc', 'f1', 'pr_auc', 'specificity', 'precision', 'recall']
    results_df = results_df[cols]
    
    print("\n" + "="*50)
    print("Final Summary Table:")
    print(results_df.to_string(index=False))
    
    output_csv = "evaluation_results_bootstrap.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\nSummary saved to {output_csv}")

if __name__ == "__main__":
    main()

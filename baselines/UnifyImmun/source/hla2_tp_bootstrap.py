import time
import pandas as pd
import numpy as np
import random
import warnings
from collections import Counter
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.metrics import roc_auc_score, auc, accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from sklearn.utils import resample 
import os

# 假设 models.HLA 中包含了 Mymodel_HLA, vocab, MyDataSet_HLA, data_process_HLA
# 务必确保 models/HLA.py 中的 Encoder_padding forward 函数已修复 batch_size 问题
from models.HLA import * 

# ================= 配置区域 =================

# 1. 路径配置 (严格隔离)
# 训练集：用于反向传播
TRAIN_PATH = '/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/train_fold_1.csv'
# 验证集：同源分布，仅用于 Checkpointing (选最佳Epoch)
VAL_PATH   = '/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/val_fold_1.csv'
# 测试集1：异源分布 OOD，仅用于最终 Bootstrap 测试
INDEPENDENT_PATH = '/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/independent_set.csv'
# 测试集2：异源分布 OOD，仅用于最终 Bootstrap 测试
EXTERNAL_PATH    = '/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/external_set.csv'
PRETRAINED_HLA_ENCODER_PATH = '../trained_model/TCR_Bootstrap_stage1/encoder_P_best.pth'
SAVE_DIR = '../trained_model/HLA_Bootstrap_stage2'

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

warnings.filterwarnings("ignore")

# 2. 随机种子
seed = 921
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# 开启 cudnn 加速
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# 3. 超参数 (针对 A100 优化)
pep_max_len = 15
hla_max_len = 34
vocab_size = len(vocab)
batch_size = 81920  # 大 Batch Size
epochs = 30
threshold = 0.5
lr = 1e-2           # 适配大 Batch 的学习率
n_bootstrap = 5     # Bootstrap 采样次数

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# ================= 工具函数 =================

f_mean = lambda l: sum(l) / len(l)

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup1 = {}
        self.backup2 = {}
    def attack(self, epsilon=1., emb_name='emb'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if emb_name == 'encoder_H.src_emb':
                    self.backup1[name] = param.data.clone()
                if emb_name == 'encoder_P.src_emb':
                    self.backup2[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
    def restore(self, emb_name='emb'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if emb_name == 'encoder_H.src_emb':
                    assert name in self.backup1
                    param.data = self.backup1[name]
                if emb_name == 'encoder_P.src_emb':
                    assert name in self.backup2
                    param.data = self.backup2[name]
        if emb_name == 'encoder_H.src_emb':
            self.backup1 = {}
        if emb_name == 'encoder_P.src_emb':
            self.backup2 = {}

def calculate_metrics(y_true, y_pred, threshold=0.5):
    """计算单次指标"""
    y_pred_transfer = transfer(y_pred, threshold)
    accuracy = accuracy_score(y_true, y_pred_transfer)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_transfer, labels=[0, 1]).ravel().tolist()
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    precision = precision_score(y_true, y_pred_transfer, zero_division=0)
    recall = recall_score(y_true, y_pred_transfer, zero_division=0)
    f1 = f1_score(y_true, y_pred_transfer, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except:
        roc_auc = 0
    prec, reca, _ = precision_recall_curve(y_true, y_pred)
    aupr = auc(reca, prec)
    mcc = matthews_corrcoef(y_true, y_pred_transfer)
    
    return [roc_auc, accuracy, mcc, f1, aupr, sensitivity, specificity, precision, recall]

def train_epoch(model, train_loader, optimizer, criterion, epoch, epochs):
    model.train()
    loss_list = []
    fgm = FGM(model)
    
    loop = tqdm(train_loader, colour='yellow', desc=f'Train Epoch {epoch}')
    for train_pep_inputs, train_hla_inputs, train_labels in loop:
        train_pep_inputs = train_pep_inputs.to(device)
        train_hla_inputs = train_hla_inputs.to(device)
        train_labels = train_labels.to(device)
        
        # 1. Forward
        train_outputs, _ = model(train_pep_inputs, train_hla_inputs)
        train_loss = criterion(train_outputs, train_labels)
        train_loss.backward()
        
        # 2. 对抗攻击
        fgm.attack(emb_name='encoder_H.src_emb')
        fgm.attack(emb_name='encoder_P.src_emb')
        train_outputs2, _ = model(train_pep_inputs, train_hla_inputs)
        loss_sum = criterion(train_outputs2, train_labels)
        loss_sum.backward()
        fgm.restore(emb_name='encoder_H.src_emb')
        fgm.restore(emb_name='encoder_P.src_emb')
        
        # 3. Update
        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(train_loss.item())
        loop.set_postfix(loss=train_loss.item())

    print(f'Train Epoch {epoch}/{epochs} Mean Loss: {f_mean(loss_list):.4f}')

def validate_for_selection(model, val_loader):
    """
    仅用于在 Val Fold 1 上选择最佳模型。
    """
    model.eval()
    y_true_val_list, y_pred_val_list = [], []
    
    with torch.no_grad():
        for val_pep_inputs, val_hla_inputs, val_labels in val_loader:
            val_pep_inputs = val_pep_inputs.to(device)
            val_hla_inputs = val_hla_inputs.to(device)
            val_labels = val_labels.to(device)
            
            val_outputs, _ = model(val_pep_inputs, val_hla_inputs)
            
            y_true_val = val_labels.cpu().numpy()
            y_pred_val = nn.Softmax(dim=1)(val_outputs)[:, 1].cpu().detach().numpy()
            y_true_val_list.extend(y_true_val)
            y_pred_val_list.extend(y_pred_val)
            
    metrics = calculate_metrics(y_true_val_list, y_pred_val_list, threshold)
    avg_score = sum(metrics[:5]) / 5
    return avg_score

def bootstrap_evaluate(model, loader, n_bootstrap=5, dataset_name='Test Set'):
    """
    OOD 测试集专用评估函数：
    在全量预测结果上进行 Bootstrap 重采样。
    """
    model.eval()
    all_y_true, all_y_pred = [], []
    
    # 1. 运行一次全量推理
    with torch.no_grad():
        for pep_inputs, hla_inputs, labels in tqdm(loader, desc=f'Inference on {dataset_name}', colour='blue'):
            pep_inputs = pep_inputs.to(device)
            hla_inputs = hla_inputs.to(device)
            labels = labels.to(device)
            
            outputs, _ = model(pep_inputs, hla_inputs)
            
            all_y_true.extend(labels.cpu().numpy())
            all_y_pred.extend(nn.Softmax(dim=1)(outputs)[:, 1].cpu().detach().numpy())
            
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    print(f"\n>>> Running {n_bootstrap}-times Bootstrap on {dataset_name}...")
    
    metrics_list = []
    
    # 2. Bootstrap 重采样
    for i in range(n_bootstrap):
        y_true_boot, y_pred_boot = resample(all_y_true, all_y_pred, replace=True, random_state=i*seed)
        metrics = calculate_metrics(y_true_boot, y_pred_boot, threshold)
        metrics_list.append(metrics)
        
    # 3. 汇总结果
    metrics_name = ['roc_auc', 'accuracy', 'mcc', 'f1', 'aupr', 'sensitivity', 'specificity', 'precision', 'recall']
    df = pd.DataFrame(metrics_list, columns=metrics_name)
    
    print(f"\n----- {dataset_name} Results (Mean ± Std) -----")
    for name in metrics_name:
        mean_val = df[name].mean()
        std_val = df[name].std()
        print(f"{name:12s}: {mean_val:.4f} ± {std_val:.4f}")
        
    return df

def get_dataloader(path, batch_size, shuffle=False):
    print(f"Loading: {path}")
    data = pd.read_csv(path).dropna()
    pep_inputs, hla_inputs, labels = data_process_HLA(data)
    dataset = MyDataSet_HLA(pep_inputs, hla_inputs, labels)
    # num_workers=16 + pin_memory 加速数据加载
    return Data.DataLoader(dataset, 
                           batch_size=batch_size, 
                           shuffle=shuffle, 
                           num_workers=16, 
                           pin_memory=True,
                           persistent_workers=True)

# ================= 主程序流程 =================

if __name__ == '__main__':
    # 1. 数据加载
    print('\n--- 1. Initializing Data Loaders ---')
    train_loader = get_dataloader(TRAIN_PATH, batch_size, shuffle=True)
    val_loader   = get_dataloader(VAL_PATH,   batch_size, shuffle=False)
    
    # 延迟加载测试集以节省内存，或者现在加载
    ind_loader   = get_dataloader(INDEPENDENT_PATH, batch_size, shuffle=False)
    ext_loader   = get_dataloader(EXTERNAL_PATH,    batch_size, shuffle=False)

    # 2. 模型初始化
    print('\n--- 2. Initializing HLA Model ---')
    model = Mymodel_HLA().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3. 训练循环
    print('\n--- 3. Training Start ---')
    save_path = os.path.join(SAVE_DIR, 'model_HLA_best.pkl')
    # 定义 Encoder 单独保存的路径
    encoder_save_path = os.path.join(SAVE_DIR, 'encoder_P_best.pth')
    
    performance_best = 0

    for epoch in range(1, epochs + 1):
        # Train
        train_epoch(model, train_loader, optimizer, criterion, epoch, epochs)
        
        # Validate (Check Pointing)
        avg_score = validate_for_selection(model, val_loader)
        
        if avg_score > performance_best:
            performance_best = avg_score
            
            # --- 关键修改：保存完整模型 ---
            torch.save(model.state_dict(), save_path)
            
            # --- 关键修改：单独保存 Encoder 以供 TCR 任务使用 ---
            torch.save(model.encoder_P.state_dict(), encoder_save_path)
            
            print(f'  [Checkpoint] New Best Model at Epoch {epoch} (Val Score: {avg_score:.4f}) Saved.')
            print(f'               Encoder saved to: {encoder_save_path}')

    print('\n--- 4. Training Finished. Starting Final Evaluation ---')

    if os.path.exists(save_path):
        print(f"Loading best model from: {save_path}")
        model.load_state_dict(torch.load(save_path))
        
        # 4. Final Test (Bootstrap)
        df_ind = bootstrap_evaluate(model, ind_loader, n_bootstrap=n_bootstrap, dataset_name='Independent Set')
        df_ext = bootstrap_evaluate(model, ext_loader, n_bootstrap=n_bootstrap, dataset_name='External Set')
        
        print("\n\n================ FINAL REPORT (HLA) ================")
        report = pd.DataFrame({
            'Metric': df_ind.columns,
            'Ind_Mean': df_ind.mean().values,
            'Ind_Std': df_ind.std().values,
            'Ext_Mean': df_ext.mean().values,
            'Ext_Std': df_ext.std().values
        })
        
        print(report.round(4).to_string(index=False))
        print("====================================================")

    else:
        print("Error: Model file was not saved!")
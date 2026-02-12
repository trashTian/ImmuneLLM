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

# 引入 TCR 模型组件
# 务必确保 models/TCR.py 中的 Encoder_padding forward 函数已修复 batch_size 问题
from models.TCR import * 

# ================= 配置区域 =================

# 1. 数据路径 (严格隔离)
# BASE_PATH = '/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR'

# # 训练集：用于反向传播
# TRAIN_PATH = os.path.join(BASE_PATH, 'train_fold_1.csv')
# # 验证集：同源分布，仅用于 Checkpointing
# VAL_PATH   = os.path.join(BASE_PATH, 'val_fold_1.csv')

# # OOD 测试集列表：用于最终 Bootstrap 测试
# TEST_SETS = {
#     'Independent Set': os.path.join(BASE_PATH, 'independent_set.csv'),
#     'Triple Set':      os.path.join(BASE_PATH, 'triple_set.csv'),
#     'Covid Set':       os.path.join(BASE_PATH, 'covid_set.csv')
# }

# # 预训练 Peptide Encoder 路径 (来自 HLA 任务)
# # 请确保这个路径指向你上一步训练 HLA 模型生成的 encoder_P_best.pth
# PRETRAINED_HLA_ENCODER_PATH = '../trained_model/HLA_Bootstrap_stage1/encoder_P_best.pth'

# SAVE_DIR = '../trained_model/TCR_Bootstrap_stage1'


# 1. 数据路径 (严格隔离)
BASE_PATH = '/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR'

# 训练集：用于反向传播
TRAIN_PATH = "/mnt/lustre/guopeijin/Immune_LLM/code/data_prepare/stage2_data_efficiency/original_csv/train_subset_TCR_25pct.csv"
# 验证集：同源分布，仅用于 Checkpointing
VAL_PATH   = os.path.join(BASE_PATH, 'val_fold_1.csv')

# OOD 测试集列表：用于最终 Bootstrap 测试
TEST_SETS = {
    # 'Independent Set': os.path.join(BASE_PATH, 'independent_set.csv'),
    'Triple Set':      os.path.join(BASE_PATH, 'triple_set.csv'),
    # 'Covid Set':       os.path.join(BASE_PATH, 'covid_set.csv')
}

# 预训练 Peptide Encoder 路径 (来自 HLA 任务)
# 请确保这个路径指向你上一步训练 HLA 模型生成的 encoder_P_best.pth
PRETRAINED_HLA_ENCODER_PATH = '/mnt/lustre/guopeijin/Immune_LLM/code/baselines/unifyimmune/UnifyImmun-main/trained_model/ratio25/encoder_P_best.pth'

SAVE_DIR = '/mnt/lustre/guopeijin/Immune_LLM/code/baselines/unifyimmune/UnifyImmun-main/trained_model/ratio25'




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
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# 3. 超参数 (保持模型结构参数与原始代码一致，优化 batch_size 和 lr)
# 原始参数: d_model=64, n_heads=1, n_layers=1
batch_size = 81920  # 针对 A100 优化
lr = 1e-3           # 配合大 Batch 优化
epochs = 30
threshold = 0.5
n_bootstrap = 5     

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# ================= 工具函数 =================

f_mean = lambda l: sum(l) / len(l)

class FGM():
    """
    对抗训练模块
    修正点：原始 TCR 代码攻击的是 'encoder_T' 和 'encoder_P'
    """
    def __init__(self, model):
        self.model = model
        self.backup1 = {}
        self.backup2 = {}
        
    def attack(self, epsilon=1., emb_name='emb'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # 对应 models.TCR 中的命名
                if emb_name == 'encoder_T.src_emb':
                    self.backup1[name] = param.data.clone()
                if emb_name == 'encoder_P.src_emb':
                    self.backup2[name] = param.data.clone()
                
                if param.grad is not None:
                    norm = torch.norm(param.grad)
                    if norm != 0:
                        r_at = epsilon * param.grad / norm
                        param.data.add_(r_at)
                        
    def restore(self, emb_name='emb'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if emb_name == 'encoder_T.src_emb':
                    assert name in self.backup1
                    param.data = self.backup1[name]
                if emb_name == 'encoder_P.src_emb':
                    assert name in self.backup2
                    param.data = self.backup2[name]
        if emb_name == 'encoder_T.src_emb':
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
    # 注意：TCR 的 DataLoader 返回顺序通常是 pep, tcr, label
    for pep_inputs, tcr_inputs, labels in loop:
        pep_inputs = pep_inputs.to(device)
        tcr_inputs = tcr_inputs.to(device)
        labels = labels.to(device)
        
        # 1. Forward
        outputs, _ = model(pep_inputs, tcr_inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # 2. Adversarial Attack (FGM)
        # 严格对齐原始代码：先 attack encoder_T, 再 attack encoder_P
        fgm.attack(emb_name='encoder_T.src_emb')
        fgm.attack(emb_name='encoder_P.src_emb')
        
        outputs_adv, _ = model(pep_inputs, tcr_inputs)
        loss_adv = criterion(outputs_adv, labels)
        loss_adv.backward()
        
        fgm.restore(emb_name='encoder_T.src_emb')
        fgm.restore(emb_name='encoder_P.src_emb')
        
        # 3. Update
        optimizer.step()
        optimizer.zero_grad()
        
        loss_list.append(loss.item())
        loop.set_postfix(loss=loss.item())

    print(f'Train Epoch {epoch}/{epochs} Mean Loss: {f_mean(loss_list):.4f}')

def validate_for_selection(model, val_loader):
    """仅用于 Checkpointing"""
    model.eval()
    y_true_all, y_pred_all = [], []
    
    with torch.no_grad():
        for pep_inputs, tcr_inputs, labels in val_loader:
            pep_inputs = pep_inputs.to(device)
            tcr_inputs = tcr_inputs.to(device)
            
            outputs, _ = model(pep_inputs, tcr_inputs)
            
            y_true_all.extend(labels.cpu().numpy())
            y_pred_all.extend(nn.Softmax(dim=1)(outputs)[:, 1].cpu().detach().numpy())
            
    metrics = calculate_metrics(y_true_all, y_pred_all, threshold)
    avg_score = sum(metrics[:5]) / 5
    return avg_score

def bootstrap_evaluate(model, loader, n_bootstrap=5, dataset_name='Test Set'):
    """全量推理 + Bootstrap 重采样"""
    model.eval()
    all_y_true, all_y_pred = [], []
    
    # 1. Inference
    with torch.no_grad():
        for pep_inputs, tcr_inputs, labels in tqdm(loader, desc=f'Inference on {dataset_name}', colour='blue'):
            pep_inputs = pep_inputs.to(device)
            tcr_inputs = tcr_inputs.to(device)
            labels = labels.to(device)
            
            outputs, _ = model(pep_inputs, tcr_inputs)
            
            all_y_true.extend(labels.cpu().numpy())
            all_y_pred.extend(nn.Softmax(dim=1)(outputs)[:, 1].cpu().detach().numpy())
            
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    print(f"\n>>> Running {n_bootstrap}-times Bootstrap on {dataset_name}...")
    
    metrics_list = []
    
    # 2. Resampling
    for i in range(n_bootstrap):
        # random_state i*seed 保证可复现
        y_true_boot, y_pred_boot = resample(all_y_true, all_y_pred, replace=True, random_state=i*seed)
        metrics = calculate_metrics(y_true_boot, y_pred_boot, threshold)
        metrics_list.append(metrics)
        
    # 3. Report
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
    # 使用原始代码中的 data_process_tcr
    pep_inputs, tcr_inputs, labels = data_process_tcr(data)
    dataset = MyDataSet_tcr(pep_inputs, tcr_inputs, labels)
    
    return Data.DataLoader(dataset, 
                           batch_size=batch_size, 
                           shuffle=shuffle, 
                           num_workers=16, 
                           pin_memory=True,
                           persistent_workers=True)

# ================= 主程序 =================

if __name__ == '__main__':
    # 1. 加载数据
    print('\n--- 1. Initializing Data Loaders ---')
    train_loader = get_dataloader(TRAIN_PATH, batch_size, shuffle=True)
    val_loader   = get_dataloader(VAL_PATH,   batch_size, shuffle=False)
    
    test_loaders = {}
    for name, path in TEST_SETS.items():
        test_loaders[name] = get_dataloader(path, batch_size, shuffle=False)

    # 2. 初始化模型
    print('\n--- 2. Initializing TCR Model ---')
    model = Mymodel_tcr().to(device)
    
    # --- 关键修正：加载预训练 HLA Encoder ---
    # 这是原始代码中的逻辑，用于迁移学习
    if os.path.exists(PRETRAINED_HLA_ENCODER_PATH):
        print(f"Loading Pretrained HLA Encoder from: {PRETRAINED_HLA_ENCODER_PATH}")
        # 加载权重
        hla_state_dict = torch.load(PRETRAINED_HLA_ENCODER_PATH)
        # 将权重加载到 TCR 模型的 encoder_P 中
        model.encoder_P.load_state_dict(hla_state_dict)
    else:
        print(f"Warning: Pretrained HLA Encoder not found at {PRETRAINED_HLA_ENCODER_PATH}!")
        print("Training will proceed from scratch (Performance might be lower than original paper).")
    # ------------------------------------

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3. 训练流程
    print('\n--- 3. Training Start ---')
    save_path = os.path.join(SAVE_DIR, 'model_TCR_best.pkl')
    encoder_save_path = os.path.join(SAVE_DIR, 'encoder_P_best.pth')
    performance_best = 0

    for epoch in range(1, epochs + 1):
        train_epoch(model, train_loader, optimizer, criterion, epoch, epochs)
        
        # Checkpointing
        avg_score = validate_for_selection(model, val_loader)
        
        if avg_score > performance_best:
            performance_best = avg_score
            torch.save(model.state_dict(), save_path)
            print(f'  [Checkpoint] New Best Model at Epoch {epoch} (Val Score: {avg_score:.4f}) Saved.')
            torch.save(model.encoder_P.state_dict(), encoder_save_path)

    # 4. 最终评估流程
    print('\n--- 4. Training Finished. Starting Final Evaluation ---')

    if os.path.exists(save_path):
        print(f"Loading best model from: {save_path}")
        model.load_state_dict(torch.load(save_path))
        
        final_results = {}
        
        # 遍历三个测试集
        for name, loader in test_loaders.items():
            df_result = bootstrap_evaluate(model, loader, n_bootstrap=n_bootstrap, dataset_name=name)
            final_results[name] = df_result
            
        print("\n\n================ FINAL REPORT (TCR) ================")
        summary_data = {'Metric': final_results['Independent Set'].columns}
        
        for name, df in final_results.items():
            summary_data[f'{name}_Mean'] = df.mean().values
            summary_data[f'{name}_Std'] = df.std().values
            
        report = pd.DataFrame(summary_data)
        # 格式化输出
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(report.round(4))
        print("====================================================")
        
    else:
        print("Error: Model file was not saved!")
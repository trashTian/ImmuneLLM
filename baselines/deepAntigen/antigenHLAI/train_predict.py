
import os
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import argparse
from tqdm import tqdm
from rdkit import Chem
from sklearn.metrics import (roc_auc_score, accuracy_score, matthews_corrcoef, 
                             f1_score, average_precision_score, precision_score, 
                             recall_score, confusion_matrix)

# PyTorch Geometric 依赖
from torch.utils.data import DataLoader
from torch_geometric import data as DATA
from torch_geometric.data import Batch
from torch_geometric.utils.subgraph import subgraph
torch.multiprocessing.set_sharing_strategy('file_system')
# ---------------------------------------------------------
# 1. 导入自定义模块
# ---------------------------------------------------------
try:
    from load_dataset.featurizer import MolGraphConvFeaturizer
    from networks.pHLAI_seq import DeepGCN
    from utils.model_utils import AverageMeter, FocalLoss, set_optimizer, adjust_learning_rate, warmup_learning_rate
except ImportError as e:
    print("----------------------------------------------------------------")
    print(f"Error: 缺少依赖文件。请确保 load_dataset/, networks/ 和 utils/ 在当前目录下。")
    print(f"详细错误: {e}")
    print("----------------------------------------------------------------")
    exit()

# ---------------------------------------------------------
# 2. 核心：带本地缓存的高效 Dataset
# ---------------------------------------------------------
class pMHC_DataSet_Cached(DATA.InMemoryDataset):
    def __init__(self, csv_path, cache_dir='./cached_graphs', aug=False, test=False):
        super(pMHC_DataSet_Cached, self).__init__()
        self.aug = aug
        self.test = test
        self.AAstringList = list('ACDEFGHIKLMNPQRSTVWY')
        
        # 1. 准备缓存路径
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # 根据csv文件名生成唯一的缓存文件名
        file_name = os.path.basename(csv_path).split('.')[0]
        self.cache_path = os.path.join(cache_dir, f"{file_name}_graphs.pkl")

        print(f"Reading CSV: {csv_path} ...")
        self.rawdata = pd.read_csv(csv_path, header=0)

        # 2. 检查本地是否有缓存
        if os.path.exists(self.cache_path):
            print(f"Found cache! Loading graph data from: {self.cache_path}")
            try:
                with open(self.cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.peptide_cache = cache_data['peptide']
                    self.pseudo_cache = cache_data['hla']
                print("Cache loaded successfully.")
            except Exception as e:
                print(f"Failed to load cache ({e}), re-computing...")
                self.process_and_save()
        else:
            print(f"No cache found at {self.cache_path}. Pre-computing graphs...")
            self.process_and_save()

    # ================= 添加这个方法 =================
    def __len__(self):
        return len(self.rawdata)
    # ===============================================

    def check(self, seq):
        # 检查非法氨基酸
        for aa in seq:
            if aa not in self.AAstringList:
                return False
        return True

    def generateGraph(self, seq):
        # RDKit -> Graph
        featurizer = MolGraphConvFeaturizer(use_edges=True)
        seq_chem = Chem.MolFromSequence(seq)
        if seq_chem is None:
            return None
        seq_feature = featurizer._featurize(seq_chem)
        feature = seq_feature.node_features
        edge_index = seq_feature.edge_index
        edge_feature = seq_feature.edge_features
        
        # 转换为 PyG 对象 (存在 CPU 内存中)
        graph = DATA.Data(x=torch.Tensor(feature), 
                          edge_index=torch.LongTensor(edge_index), 
                          edge_attr=torch.Tensor(edge_feature))
        return graph

    def process_and_save(self):
        """一次性计算所有图并保存到硬盘"""
        unique_peptides = self.rawdata['peptide'].unique()
        # 注意：这里假设您的 CSV 列名是 HLA
        unique_hlas = self.rawdata['HLA'].unique()
        
        peptide_cache = {}
        pseudo_cache = {}

        print(f"Processing {len(unique_peptides)} unique peptides...")
        for pep in tqdm(unique_peptides, desc="Peptides"):
            if self.check(pep):
                graph = self.generateGraph(pep)
                if graph is not None:
                    peptide_cache[pep] = graph

        print(f"Processing {len(unique_hlas)} unique HLA sequences...")
        for hla in tqdm(unique_hlas, desc="HLAs"):
            # HLA 序列较长，如需校验可打开 check
            graph = self.generateGraph(hla)
            if graph is not None:
                pseudo_cache[hla] = graph
        
        self.peptide_cache = peptide_cache
        self.pseudo_cache = pseudo_cache

        print(f"Saving graphs to {self.cache_path} ...")
        with open(self.cache_path, 'wb') as f:
            pickle.dump({'peptide': peptide_cache, 'hla': pseudo_cache}, f)
        print("Cache saved!")

    def __getitem__(self, idx):
        row = self.rawdata.loc[idx]
        peptide = row['peptide']
        pseudo = row['HLA']
        label = row['label'] if 'label' in self.rawdata.columns else -1

        # 查表获取图
        # 如果数据集中有脏数据导致没有生成图，这里简单的策略是取下一条数据
        if peptide not in self.peptide_cache or pseudo not in self.pseudo_cache:
            return self.__getitem__((idx + 1) % len(self))

        peptide_graph = self.peptide_cache[peptide]
        pseudo_graph = self.pseudo_cache[pseudo]

        # 数据增强 (只在训练时开启)
        # 注意：必须在 deepcopy 的副本上操作，不能修改缓存中的原图
        if self.aug:
            peptide_graph = self.augmentation(peptide_graph)
            pseudo_graph = self.augmentation(pseudo_graph)
        
        # 直接返回对象，不需要 pickle
        return (idx, peptide, pseudo, label, peptide_graph, pseudo_graph)

    def augmentation(self, graph):
        if graph is None: return None
        aug_graph = copy.deepcopy(graph)
        prob = torch.rand(aug_graph.num_nodes)
        mask = prob > 0.05 # 随机 mask 掉 5% 的节点
        edge_index, edge_attr = subgraph(mask, aug_graph.edge_index, aug_graph.edge_attr, relabel_nodes=True)
        aug_graph.x = aug_graph.x[mask, :]
        aug_graph.edge_index = edge_index
        aug_graph.edge_attr = edge_attr
        return aug_graph

def collate(batch):
    idxs = [item[0] for item in batch]
    peptides = [item[1] for item in batch]
    alleles = [item[2] for item in batch]
    labels = [item[3] for item in batch]
    
    # 关键修改：直接获取列表，不再 pickle.loads，大幅降低 CPU 开销
    peptide_graphs = [item[4] for item in batch]
    pseudo_graphs = [item[5] for item in batch]
    
    return idxs, peptides, alleles, torch.LongTensor(labels), peptide_graphs, pseudo_graphs

# ---------------------------------------------------------
# 3. 辅助函数：指标计算与参数
# ---------------------------------------------------------
def calculate_metrics(y_true, y_pred, y_prob):
    try:
        auc = roc_auc_score(y_true, y_prob)
    except: auc = 0.0
    
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except: pr_auc = 0.0

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc": auc, "accuracy": acc, "mcc": mcc, "f1": f1,
        "pr_auc": pr_auc, "specificity": specificity,
        "precision": precision, "recall": recall
    }

def get_args():
    return {
        'batchsize': 512,      # 建议 256 或 512，不要 10000+
        'epochs': 50,
        'lr': 0.01,
        'hidden_size': 64,    # 需匹配 DeepGCN 定义
        'depth': 3,
        'k': 20,
        'heads': 4,
        'dropout': 0.1,
        'num_process': 8,      # 建议设置为 CPU 核心数的一半
        'save_dir': './checkpoints/',
        'cache_dir': './cached_graphs', # 缓存文件存放位置
        'lr_decay_epochs': [30, 40],
        'lr_decay_rate': 0.1,
        'weight_decay': 1e-4,
        'momentum': 0.9,
        'print_freq': 1,
        'warm': False,
        'cosine': False,
        'optim':'Adam'
    }

# ---------------------------------------------------------
# 4. 训练与验证流程
# ---------------------------------------------------------
def train_one_epoch(train_loader, model, criterion, optimizer, epoch, device, args):
    model.train()
    losses = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training", unit="batch")
    
    for idx, (_, peptides, alleles, labels, peptide_graphs, pseudo_graphs) in enumerate(pbar):
        # 使用 PyG 的 Batch 将列表拼接成大图
        peptide_graphs = Batch.from_data_list(peptide_graphs).to(device)
        pseudo_graphs = Batch.from_data_list(pseudo_graphs).to(device)
        labels = labels.to(device)

        logits = model(peptide_graphs, pseudo_graphs)
        loss = criterion(logits, labels)

        losses.update(loss.item(), labels.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.get('warm', False):
             warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)
        
        pbar.set_postfix({'loss': f'{losses.avg:.4f}'})

    return losses.avg

def evaluate(val_loader, model, criterion, device, is_test=False):
    model.eval()
    losses = AverageMeter()
    val_preds = []
    val_trues = []
    val_scores = []
    
    stage_name = "Testing" if is_test else "Validating"
    pbar = tqdm(val_loader, desc=f"{stage_name}", unit="batch")

    with torch.no_grad():
        for idx, (_, peptides, alleles, labels, peptide_graphs, pseudo_graphs) in enumerate(pbar):
            peptide_graphs = Batch.from_data_list(peptide_graphs).to(device)
            pseudo_graphs = Batch.from_data_list(pseudo_graphs).to(device)
            labels = labels.to(device)

            logits = model(peptide_graphs, pseudo_graphs)
            loss = criterion(logits, labels)
            
            losses.update(loss.item(), labels.size(0))
            
            # 获取预测结果
            preds = logits.argmax(dim=1)
            scores = logits[:, 1] # 取正类概率 (index 1)
            
            val_preds.extend(preds.cpu().numpy())
            val_scores.extend(scores.cpu().numpy())
            val_trues.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{losses.avg:.4f}'})

    # 计算 8 项指标
    m = calculate_metrics(val_trues, val_preds, val_scores)
    
    print("\n" + "="*105)
    print(f"{stage_name} Results:")
    print(f"{'AUC':<10} {'ACC':<10} {'MCC':<10} {'F1':<10} {'PR_AUC':<10} {'Spec':<10} {'Prec':<10} {'Recall':<10}")
    print(f"{m['auc']:<10.4f} {m['accuracy']:<10.4f} {m['mcc']:<10.4f} "
          f"{m['f1']:<10.4f} {m['pr_auc']:<10.4f} {m['specificity']:<10.4f} "
          f"{m['precision']:<10.4f} {m['recall']:<10.4f}")
    print("="*105 + "\n")
    
    return m['auc']

# ---------------------------------------------------------
# 5. 主函数
# ---------------------------------------------------------
def main():
    # --- 路径设置 ---
    # 请确认这些路径是正确的
    train_csv = '/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/train_fold_1.csv'
    val_csv = '/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/val_fold_1.csv'
    test_csv = '/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/independent_set.csv'
    
    args = get_args()
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Batch Size: {args['batchsize']} | Workers: {args['num_process']}")

    # --- 数据集加载 (带缓存) ---
    print("\n[Step 1] Initializing Datasets (Graphs will be cached to disk)...")
    # aug=True 表示训练集开启数据增强
    train_dataset = pMHC_DataSet_Cached(train_csv, cache_dir=args['cache_dir'], aug=True, test=False)
    val_dataset = pMHC_DataSet_Cached(val_csv, cache_dir=args['cache_dir'], aug=False, test=False)
    test_dataset = pMHC_DataSet_Cached(test_csv, cache_dir=args['cache_dir'], aug=False, test=True)

    # --- DataLoader ---
    # 训练集必须加 drop_last=True，防止最后一个 batch 大小为 1 导致 BN 报错
    train_loader = DataLoader(train_dataset, batch_size=args['batchsize'], shuffle=True, 
                              collate_fn=collate, num_workers=args['num_process'], 
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args['batchsize'], shuffle=False, 
                            collate_fn=collate, num_workers=args['num_process'], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args['batchsize'], shuffle=False, 
                             collate_fn=collate, num_workers=args['num_process'], pin_memory=True)

    # --- 模型初始化 ---
    print("\n[Step 2] Initializing Model...")
    model = DeepGCN(args).to(device)
    criterion = FocalLoss(reduction='sum').to(device)
    optimizer = set_optimizer(model, args)
    
    best_auroc = 0.0
    
    # --- 训练循环 ---
    print(f"\n[Step 3] Start Training for {args['epochs']} epochs...")
    for epoch in range(1, args['epochs'] + 1):
        adjust_learning_rate(args, optimizer, epoch)
        
        # 1. Train
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, device, args)
        
        # 2. Validation
        if epoch % args['print_freq'] == 0:
            val_auroc = evaluate(val_loader, model, criterion, device, is_test=False)
            
            # Save Best Model
            if val_auroc > best_auroc:
                best_auroc = val_auroc
                if not os.path.exists(args['save_dir']):
                    os.makedirs(args['save_dir'])
                save_path = os.path.join(args['save_dir'], 'best_model.pt')
                torch.save({'model': model.state_dict(), 'args': args}, save_path)
                print(f"*** New Best Model Saved (AUROC: {best_auroc:.4f}) ***")

    # --- 最终测试 ---
    print("\n" + "#"*50)
    print("Training Finished. Loading best model for testing...")
    print("#"*50)
    
    checkpoint = torch.load(os.path.join(args['save_dir'], 'best_model.pt'))
    model.load_state_dict(checkpoint['model'])
    
    print("Evaluating on Test Set...")
    evaluate(test_loader, model, criterion, device, is_test=True)

if __name__ == '__main__':
    main()
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import argparse
import torch.distributed as dist
from tqdm import tqdm
from rdkit import Chem
from sklearn.metrics import (roc_auc_score, accuracy_score, matthews_corrcoef, 
                             f1_score, average_precision_score, precision_score, 
                             recall_score, confusion_matrix)

# PyTorch Geometric 依赖
from torch.utils.data import DataLoader, DistributedSampler
from torch_geometric import data as DATA
from torch_geometric.data import Batch
from torch_geometric.utils.subgraph import subgraph
from torch_geometric.data import Batch
# DDP 依赖
from torch.nn.parallel import DistributedDataParallel as DDP

torch.multiprocessing.set_sharing_strategy('file_system')

# ---------------------------------------------------------
# 1. 导入自定义模块
# ---------------------------------------------------------
try:
    from load_dataset.featurizer import MolGraphConvFeaturizer
    from networks.pHLAI_seq import DeepGCN
    from utils.model_utils import AverageMeter, FocalLoss, set_optimizer, adjust_learning_rate, warmup_learning_rate
except ImportError as e:
    print(f"Error: 缺少依赖文件。请确保 load_dataset/, networks/ 和 utils/ 在当前目录下。")
    exit()

# ---------------------------------------------------------
# 2. 核心：带本地缓存的高效 Dataset (保持不变)
# ---------------------------------------------------------
class pMHC_DataSet_Cached(DATA.InMemoryDataset):
    def __init__(self, csv_path, cache_dir='./cached_graphs', aug=False, test=False):
        super(pMHC_DataSet_Cached, self).__init__()
        self.aug = aug
        self.test = test
        self.AAstringList = list('ACDEFGHIKLMNPQRSTVWY')
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True) # 防止多进程竞争报错
        
        file_name = os.path.basename(csv_path).split('.')[0]
        self.cache_path = os.path.join(cache_dir, f"{file_name}_graphs.pkl")

        # 为了避免多进程同时读取CSV打印多次，只让主进程打印
        if is_main_process():
            print(f"Reading CSV: {csv_path} ...")
        self.rawdata = pd.read_csv(csv_path, header=0)

        if os.path.exists(self.cache_path):
            if is_main_process():
                print(f"Found cache! Loading graph data from: {self.cache_path}")
            try:
                with open(self.cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.peptide_cache = cache_data['peptide']
                    self.pseudo_cache = cache_data['hla']
            except Exception as e:
                if is_main_process():
                    print(f"Failed to load cache, re-computing...")
                self.process_and_save()
        else:
            if is_main_process():
                print(f"No cache found. Pre-computing graphs...")
            self.process_and_save()
        
        # 增加 barrier 确保所有进程等待主进程处理完数据（如果是第一次生成缓存）
        if dist.is_initialized():
            dist.barrier()

    def __len__(self):
        return len(self.rawdata)

    def check(self, seq):
        for aa in seq:
            if aa not in self.AAstringList:
                return False
        return True

    def generateGraph(self, seq):
        featurizer = MolGraphConvFeaturizer(use_edges=True)
        seq_chem = Chem.MolFromSequence(seq)
        if seq_chem is None:
            return None
        seq_feature = featurizer._featurize(seq_chem)
        graph = DATA.Data(x=torch.Tensor(seq_feature.node_features), 
                          edge_index=torch.LongTensor(seq_feature.edge_index), 
                          edge_attr=torch.Tensor(seq_feature.edge_features))
        return graph

    def process_and_save(self):
        # 只有主进程负责生成和保存，避免写冲突
        if not is_main_process():
            return

        unique_peptides = self.rawdata['peptide'].unique()
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

        if peptide not in self.peptide_cache or pseudo not in self.pseudo_cache:
            return self.__getitem__((idx + 1) % len(self))

        peptide_graph = self.peptide_cache[peptide]
        pseudo_graph = self.pseudo_cache[pseudo]

        if self.aug:
            peptide_graph = self.augmentation(peptide_graph)
            pseudo_graph = self.augmentation(pseudo_graph)
        
        return (idx, peptide, pseudo, label, peptide_graph, pseudo_graph)

    def augmentation(self, graph):
        if graph is None: return None
        aug_graph = copy.deepcopy(graph)
        prob = torch.rand(aug_graph.num_nodes)
        mask = prob > 0.05
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
    
    # --- 修改开始 ---
    # 在子进程中就完成图的打包，大大减少进程间通信的文件句柄消耗
    peptide_graphs_list = [item[4] for item in batch]
    pseudo_graphs_list = [item[5] for item in batch]
    
    # 将 list 转换为 PyG 的 Batch 对象
    batched_peptide = Batch.from_data_list(peptide_graphs_list)
    batched_pseudo = Batch.from_data_list(pseudo_graphs_list)
    # --- 修改结束 ---
    
    return idxs, peptides, alleles, torch.LongTensor(labels), batched_peptide, batched_pseudo

# ---------------------------------------------------------
# 3. DDP 辅助函数
# ---------------------------------------------------------
def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(gpu)
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        dist.barrier()
        print(f"Distributed init: Rank {rank}, GPU {gpu}, World Size {world_size}")
    else:
        print('Not using distributed mode')
        return False, 0
    return True, gpu

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def calculate_metrics(y_true, y_pred, y_prob):
    # 指标计算保持不变
    try: auc = roc_auc_score(y_true, y_prob)
    except: auc = 0.0
    try: pr_auc = average_precision_score(y_true, y_prob)
    except: pr_auc = 0.0
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc": auc, "accuracy": acc, "mcc": mcc, "f1": f1,
        "pr_auc": pr_auc, "specificity": specificity,
        "precision": precision, "recall": recall
    }

def get_args():
    return {
        'batchsize': 1024,       # 每个GPU的batchsize，总batchsize = 512 * 5 = 2560
        'epochs': 50,
        'lr': 0.0001,         # DDP 技巧：总batch变大，学习率通常需要线性缩放
        'hidden_size': 64,
        'depth': 5,
        'k': 20,
        'heads': 4,
        'dropout': 0,
        'num_process': 4,       # 每个进程的 DataLoader worker 数
        'save_dir': './checkpoints/',
        'cache_dir': './cached_graphs',
        'lr_decay_epochs': 150,
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
    
    # DDP 环境下只在主进程显示进度条
    if is_main_process():
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training", unit="batch")
    else:
        pbar = train_loader
    
    for idx, (_, peptides, alleles, labels, peptide_graphs, pseudo_graphs) in enumerate(pbar):
        peptide_graphs = peptide_graphs.to(device)
        pseudo_graphs = pseudo_graphs.to(device)
        labels = labels.to(device)

        logits = model(peptide_graphs, pseudo_graphs)
        loss = criterion(logits, labels)

        losses.update(loss.item(), labels.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.get('warm', False):
             warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)
        
        if is_main_process():
            pbar.set_postfix({'loss': f'{losses.avg:.4f}'})

    return losses.avg

def evaluate(val_loader, model, criterion, device, is_test=False):
    model.eval()
    losses = AverageMeter()
    
    # 局部结果容器
    local_preds = []
    local_trues = []
    local_scores = []
    
    stage_name = "Testing" if is_test else "Validating"
    if is_main_process():
        pbar = tqdm(val_loader, desc=f"{stage_name}", unit="batch")
    else:
        pbar = val_loader

    with torch.no_grad():
        for idx, (_, peptides, alleles, labels, peptide_graphs, pseudo_graphs) in enumerate(pbar):
            peptide_graphs = peptide_graphs.to(device)
            pseudo_graphs = pseudo_graphs.to(device)
            labels = labels.to(device)

            logits = model(peptide_graphs, pseudo_graphs)
            loss = criterion(logits, labels)
            
            losses.update(loss.item(), labels.size(0))
            
            preds = logits.argmax(dim=1)
            scores = logits[:, 1]
            
            local_preds.extend(preds.cpu().tolist())
            local_scores.extend(scores.cpu().tolist())
            local_trues.extend(labels.cpu().tolist())
            
            if is_main_process():
                pbar.set_postfix({'loss': f'{losses.avg:.4f}'})

    # --- DDP 关键：汇聚所有 GPU 的预测结果 ---
    if dist.is_initialized():
        # 创建列表来收集所有进程的结果
        gathered_preds = [None for _ in range(dist.get_world_size())]
        gathered_scores = [None for _ in range(dist.get_world_size())]
        gathered_trues = [None for _ in range(dist.get_world_size())]
        
        # 使用 all_gather_object (可以处理不同长度的 list)
        dist.all_gather_object(gathered_preds, local_preds)
        dist.all_gather_object(gathered_scores, local_scores)
        dist.all_gather_object(gathered_trues, local_trues)
        
        # 展平列表
        final_preds = [item for sublist in gathered_preds for item in sublist]
        final_scores = [item for sublist in gathered_scores for item in sublist]
        final_trues = [item for sublist in gathered_trues for item in sublist]
    else:
        final_preds = local_preds
        final_scores = local_scores
        final_trues = local_trues

    # 只在主进程计算和打印指标
    if is_main_process():
        m = calculate_metrics(final_trues, final_preds, final_scores)
        print("\n" + "="*105)
        print(f"{stage_name} Results (All GPUs):")
        print(f"{'AUC':<10} {'ACC':<10} {'MCC':<10} {'F1':<10} {'PR_AUC':<10} {'Spec':<10} {'Prec':<10} {'Recall':<10}")
        print(f"{m['auc']:<10.4f} {m['accuracy']:<10.4f} {m['mcc']:<10.4f} "
              f"{m['f1']:<10.4f} {m['pr_auc']:<10.4f} {m['specificity']:<10.4f} "
              f"{m['precision']:<10.4f} {m['recall']:<10.4f}")
        print("="*105 + "\n")
        return m['auc']
    else:
        return 0.0

# ---------------------------------------------------------
# 5. 主函数
# ---------------------------------------------------------
def main():
    # 初始化 DDP
    is_distributed, device_id = init_distributed_mode()
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    train_csv = '/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/train_fold_1.csv'
    val_csv = '/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/val_fold_1.csv'
    test_csv = '/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/independent_set.csv'
    
    args = get_args()
    
    if is_main_process():
        print(f"Using device: {device} | Local Batch Size: {args['batchsize']} | Total Batch Size: {args['batchsize'] * dist.get_world_size() if is_distributed else args['batchsize']}")

    # --- 数据集加载 ---
    if is_main_process():
        print("\n[Step 1] Initializing Datasets...")
    
    # 训练和验证集
    train_dataset = pMHC_DataSet_Cached(train_csv, cache_dir=args['cache_dir'], aug=True, test=False)
    val_dataset = pMHC_DataSet_Cached(val_csv, cache_dir=args['cache_dir'], aug=False, test=False)
    test_dataset = pMHC_DataSet_Cached(test_csv, cache_dir=args['cache_dir'], aug=False, test=True)

    # --- Sampler ---
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    # --- DataLoader ---
    # shuffle 必须在 Sampler 里设置，这里 DataLoader 的 shuffle 设为 False
    train_loader = DataLoader(train_dataset, batch_size=args['batchsize'], 
                              shuffle=(train_sampler is None), 
                              sampler=train_sampler,
                              collate_fn=collate, num_workers=args['num_process'], 
                              pin_memory=True, drop_last=True)
                              
    val_loader = DataLoader(val_dataset, batch_size=args['batchsize'], 
                            shuffle=False, sampler=val_sampler,
                            collate_fn=collate, num_workers=args['num_process'], pin_memory=True)
                            
    test_loader = DataLoader(test_dataset, batch_size=args['batchsize'], 
                             shuffle=False, sampler=test_sampler,
                             collate_fn=collate, num_workers=args['num_process'], pin_memory=True)

    # --- 模型初始化 ---
    if is_main_process():
        print("\n[Step 2] Initializing Model...")
    
    model = DeepGCN(args).to(device)
    
    # 转换为 SyncBatchNorm (推荐在 DDP 中使用)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if is_distributed:
        # device_ids 必须指定，否则会变慢
        model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    criterion = FocalLoss(reduction='sum').to(device)
    optimizer = set_optimizer(model, args)
    
    best_auroc = 0.0
    
    # --- 训练循环 ---
    if is_main_process():
        print(f"\n[Step 3] Start Training for {args['epochs']} epochs...")
        
    for epoch in range(1, args['epochs'] + 1):
        # DDP 必须步骤：每个 epoch 都要设置 sampler 的 epoch，以保证 shuffle 随机性
        if is_distributed:
            train_sampler.set_epoch(epoch)
            
        adjust_learning_rate(args, optimizer, epoch)
        
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, device, args)
        
        if epoch % args['print_freq'] == 0:
            val_auroc = evaluate(val_loader, model, criterion, device, is_test=False)
            
            # 只在主进程保存模型
            if is_main_process():
                if val_auroc > best_auroc:
                    best_auroc = val_auroc
                    if not os.path.exists(args['save_dir']):
                        os.makedirs(args['save_dir'])
                    # 保存 DDP 模型时要取 .module
                    save_state = model.module.state_dict() if is_distributed else model.state_dict()
                    torch.save({'model': save_state, 'args': args}, 
                               os.path.join(args['save_dir'], 'best_model.pt'))
                    print(f"*** New Best Model Saved (AUROC: {best_auroc:.4f}) ***")

    # --- 最终测试 ---
    if is_main_process():
        print("\n" + "#"*50)
        print("Training Finished. Loading best model for testing...")
        print("#"*50)
        
        # 重新加载模型 (注意这里要在 wrapper 之前加载，或者小心处理 module 前缀)
        # 简单起见，我们在主进程加载，DDP 里的权重是同步的，但最好是所有进程都加载
        checkpoint = torch.load(os.path.join(args['save_dir'], 'best_model.pt'), map_location=device)
        
        # 如果模型现在包裹了 DDP，checkpoint 是没有 module. 的（因为上面保存时去掉了），
        # 但现在的 model 实例有 module. 前缀。
        if is_distributed:
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])

    # 确保所有进程同步
    if is_distributed:
        dist.barrier()
    
    if is_main_process():
        print("Evaluating on Test Set...")
    evaluate(test_loader, model, criterion, device, is_test=True)
    
    # 清理分布式进程
    if is_distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()

    """
    # --nproc_per_node=5 表示使用5张卡
    # CUDA_VISIBLE_DEVICES=2,3,4,5,6 将物理显卡映射为程序的 0-4 号设备

    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=6 --master_port=29500 train_ddp.py
    """
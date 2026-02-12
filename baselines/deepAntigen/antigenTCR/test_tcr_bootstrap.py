# import os
# # 强制只使用一张显卡，防止模型结构自动变异
# # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# # 防止文件句柄限制
# import torch
# torch.multiprocessing.set_sharing_strategy('file_system')

# import copy
# import pandas as pd
# import numpy as np
# import pickle
# import argparse
# from tqdm import tqdm
# from rdkit import Chem
# from sklearn.metrics import (roc_auc_score, accuracy_score, matthews_corrcoef, 
#                              f1_score, average_precision_score, precision_score, 
#                              recall_score, confusion_matrix)
# from torch.utils.data import DataLoader
# from torch_geometric import data as DATA
# from torch_geometric.data import Batch
# from torch_geometric.utils.subgraph import subgraph
# from sklearn.utils import resample

# # ---------------------------------------------------------
# # 1. 导入依赖
# # ---------------------------------------------------------
# try:
#     from load_dataset.featurizer import MolGraphConvFeaturizer
#     # 【修改】导入 TCR 对应的网络
#     from networks.pTCR_seq import DeepGCN
# except ImportError as e:
#     print(f"Error: 缺少依赖文件: {e}")
#     exit()

# # ---------------------------------------------------------
# # 2. Dataset 类 (TCR 推理专用)
# # ---------------------------------------------------------
# class pTCR_DataSet_Inference(DATA.InMemoryDataset):
#     def __init__(self, csv_path, cache_dir='./cached_graphs_tcr'):
#         super(pTCR_DataSet_Inference, self).__init__()
#         self.AAstringList = list('ACDEFGHIKLMNPQRSTVWY')
        
#         if not os.path.exists(cache_dir):
#             os.makedirs(cache_dir, exist_ok=True)
        
#         file_name = os.path.basename(csv_path).split('.')[0]
#         self.cache_path = os.path.join(cache_dir, f"{file_name}_graphs.pkl")

#         print(f"Reading CSV: {csv_path} ...")
#         self.rawdata = pd.read_csv(csv_path, header=0)

#         if os.path.exists(self.cache_path):
#             print(f"Loading cache from: {self.cache_path}")
#             try:
#                 with open(self.cache_path, 'rb') as f:
#                     cache_data = pickle.load(f)
#                     self.peptide_cache = cache_data['peptide']
#                     self.cdr3_cache = cache_data['cdr3'] # 【修改】读取 cdr3 缓存
#             except Exception as e:
#                 print(f"Cache broken, re-computing...")
#                 self.process_and_save()
#         else:
#             print("No cache found. Computing graphs...")
#             self.process_and_save()

#     def __len__(self):
#         return len(self.rawdata)

#     def check(self, seq):
#         for aa in seq:
#             if aa not in self.AAstringList: return False
#         return True

#     def generateGraph(self, seq):
#         featurizer = MolGraphConvFeaturizer(use_edges=True)
#         seq_chem = Chem.MolFromSequence(seq)
#         if seq_chem is None: return None
#         seq_feature = featurizer._featurize(seq_chem)
#         graph = DATA.Data(x=torch.Tensor(seq_feature.node_features), 
#                           edge_index=torch.LongTensor(seq_feature.edge_index), 
#                           edge_attr=torch.Tensor(seq_feature.edge_features))
#         return graph

#     def process_and_save(self):
#         unique_peptides = self.rawdata['peptide'].unique()
#         # 【修改】读取 TCR 列
#         unique_cdr3s = self.rawdata['tcr'].unique()
        
#         peptide_cache = {}
#         cdr3_cache = {}

#         for pep in tqdm(unique_peptides, desc="Processing Peptides"):
#             if self.check(pep):
#                 g = self.generateGraph(pep)
#                 if g: peptide_cache[pep] = g

#         for cdr3 in tqdm(unique_cdr3s, desc="Processing TCRs"):
#             if self.check(cdr3):
#                 g = self.generateGraph(cdr3)
#                 if g: cdr3_cache[cdr3] = g
        
#         self.peptide_cache = peptide_cache
#         self.cdr3_cache = cdr3_cache

#     def __getitem__(self, idx):
#         row = self.rawdata.loc[idx]
#         peptide = row['peptide']
#         cdr3 = row['tcr'] # 【修改】读取 TCR
#         label = row['label'] if 'label' in self.rawdata.columns else -1

#         if peptide not in self.peptide_cache or cdr3 not in self.cdr3_cache:
#             return self.__getitem__(0)

#         peptide_graph = self.peptide_cache[peptide]
#         cdr3_graph = self.cdr3_cache[cdr3]
        
#         return (idx, peptide, cdr3, label, peptide_graph, cdr3_graph)

# def collate(batch):
#     idxs = [item[0] for item in batch]
#     peptides = [item[1] for item in batch]
#     tcrs = [item[2] for item in batch]
#     labels = [item[3] for item in batch]
#     peptide_graphs = Batch.from_data_list([item[4] for item in batch])
#     cdr3_graphs = Batch.from_data_list([item[5] for item in batch])
#     return idxs, peptides, tcrs, torch.LongTensor(labels), peptide_graphs, cdr3_graphs

# # ---------------------------------------------------------
# # 3. 指标计算函数
# # ---------------------------------------------------------
# def compute_metrics_dict(y_true, y_pred, y_prob):
#     try: auc = roc_auc_score(y_true, y_prob)
#     except: auc = 0.0
#     try: pr_auc = average_precision_score(y_true, y_prob)
#     except: pr_auc = 0.0

#     acc = accuracy_score(y_true, y_pred)
#     mcc = matthews_corrcoef(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred, zero_division=0)
#     precision = precision_score(y_true, y_pred, zero_division=0)
#     recall = recall_score(y_true, y_pred, zero_division=0)
    
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

#     return {"AUC": auc, "ACC": acc, "MCC": mcc, "F1": f1, "PR_AUC": pr_auc, "Specificity": specificity, "Precision": precision, "Recall": recall}

# # ---------------------------------------------------------
# # 4. 主程序
# # ---------------------------------------------------------
# def main():
#     # ================= 配置区域 (请根据实际情况修改) =================
#     # 你的 TCR 模型路径
#     model_path = './checkpoints_tcr/best_tcr_model.pt'
#     # 你的测试集路径
#     # test_csv = '/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/independent_set.csv'
#     test_csv = '/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/triple_set.csv'
#     # 你的缓存路径
#     cache_dir = './cached_graphs_tcr'
#     n_bootstrap = 5
#     batch_size = 1024
#     # =============================================================

#     device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     if not os.path.exists(model_path):
#         print(f"Error: Model not found at {model_path}")
#         return

#     print(f"Loading model checkpoint from {model_path}...")
#     checkpoint = torch.load(model_path, map_location=device)
    
#     args = checkpoint['args']
#     args['batchsize'] = batch_size 
    
#     model = DeepGCN(args).to(device)
#     model.eval()

#     # --- 智能权重加载逻辑 ---
#     print("Adapting state_dict keys...")
#     model_keys = model.state_dict().keys()
#     saved_state = checkpoint['model']
#     saved_state_clean = {k.replace("module.", ""): v for k, v in saved_state.items()}
    
#     new_state_dict = {}
#     for key in model_keys:
#         if key in saved_state_clean:
#             new_state_dict[key] = saved_state_clean[key]
#         else:
#             key_no_module = key.replace(".module.", ".")
#             if key_no_module in saved_state_clean:
#                 new_state_dict[key] = saved_state_clean[key_no_module]
#             else:
#                 print(f"Warning: Key {key} missing in checkpoint.")
    
#     model.load_state_dict(new_state_dict, strict=False)
#     print("Model loaded successfully.")

#     # --- 加载数据 ---
#     print(f"Loading Test Dataset: {test_csv}")
#     dataset = pTCR_DataSet_Inference(test_csv, cache_dir=cache_dir)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
#                         collate_fn=collate, num_workers=4, pin_memory=True)

#     # --- 执行推理 ---
#     all_trues = []
#     all_probs = []
#     all_preds = []

#     print("Running Inference...")
#     with torch.no_grad():
#         for _, _, _, labels, pep_graphs, cdr3_graphs in tqdm(loader, desc="Inference"):
#             pep_graphs = pep_graphs.to(device)
#             cdr3_graphs = cdr3_graphs.to(device)
            
#             # 【修改】传入 TCR 图
#             logits = model(pep_graphs, cdr3_graphs)
#             probs = logits[:, 1]
#             preds = logits.argmax(dim=1)
            
#             all_trues.extend(labels.numpy())
#             all_probs.extend(probs.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())

#     y_true = np.array(all_trues)
#     y_prob = np.array(all_probs)
#     y_pred = np.array(all_preds)

#     print(f"\nInference finished. Total samples: {len(y_true)}")

#     # --- Bootstrap 采样 ---
#     print(f"\nPerforming Bootstrap Analysis ({n_bootstrap} rounds)...")
    
#     bootstrap_results = []
#     np.random.seed(42)

#     for i in range(n_bootstrap):
#         indices = resample(np.arange(len(y_true)), replace=True, n_samples=len(y_true))
#         metrics = compute_metrics_dict(y_true[indices], y_pred[indices], y_prob[indices])
#         bootstrap_results.append(metrics)
#         print(f"Round {i+1}: AUC={metrics['AUC']:.4f} ACC={metrics['ACC']:.4f} F1={metrics['F1']:.4f}")

#     # --- 统计结果 ---
#     df = pd.DataFrame(bootstrap_results)
#     mean_metrics = df.mean()
#     std_metrics = df.std()

#     print("\n" + "="*60)
#     print(f"{'Metric':<15} | {'Mean':<10} | {'Std':<10}")
#     print("-" * 60)
#     for metric in mean_metrics.index:
#         print(f"{metric:<15} | {mean_metrics[metric]:.4f}     | {std_metrics[metric]:.4f}")
#     print("="*60)

#     output_file = "bootstrap_results_tcr.csv"
#     df.loc['Mean'] = mean_metrics
#     df.loc['Std'] = std_metrics
#     df.to_csv(output_file)
#     print(f"\nDetailed results saved to {output_file}")

# if __name__ == '__main__':
#     main()

import os
# 强制只使用一张显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # 根据你的报错截图，你似乎想用 cuda:2，这里可以指定
import torch
# 防止文件句柄限制
torch.multiprocessing.set_sharing_strategy('file_system')

import copy
import pandas as pd
import numpy as np
import pickle
import argparse
from tqdm import tqdm
from rdkit import Chem
from sklearn.metrics import (roc_auc_score, accuracy_score, matthews_corrcoef, 
                             f1_score, average_precision_score, precision_score, 
                             recall_score, confusion_matrix)
from torch.utils.data import DataLoader
from torch_geometric import data as DATA
from torch_geometric.data import Batch
from torch_geometric.utils.subgraph import subgraph
from sklearn.utils import resample

# ---------------------------------------------------------
# 1. 导入依赖
# ---------------------------------------------------------
try:
    from load_dataset.featurizer import MolGraphConvFeaturizer
    from networks.pTCR_seq import DeepGCN
except ImportError as e:
    print(f"Error: 缺少依赖文件: {e}")
    exit()

# ---------------------------------------------------------
# 2. Dataset 类 (修复了节点数不足导致的崩溃)
# ---------------------------------------------------------
class pTCR_DataSet_Inference(DATA.InMemoryDataset):
    def __init__(self, csv_path, cache_dir='./cached_graphs_tcr'):
        super(pTCR_DataSet_Inference, self).__init__()
        self.AAstringList = list('ACDEFGHIKLMNPQRSTVWY')
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        
        file_name = os.path.basename(csv_path).split('.')[0]
        self.cache_path = os.path.join(cache_dir, f"{file_name}_graphs.pkl")

        print(f"Reading CSV: {csv_path} ...")
        self.rawdata = pd.read_csv(csv_path, header=0)

        if os.path.exists(self.cache_path):
            print(f"Loading cache from: {self.cache_path}")
            try:
                with open(self.cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.peptide_cache = cache_data['peptide']
                    self.cdr3_cache = cache_data['cdr3']
            except Exception as e:
                print(f"Cache broken, re-computing...")
                self.process_and_save()
        else:
            print("No cache found. Computing graphs...")
            self.process_and_save()

    def __len__(self):
        return len(self.rawdata)

    def check(self, seq):
        if not isinstance(seq, str): return False
        for aa in seq:
            if aa not in self.AAstringList: return False
        return True

    def generateGraph(self, seq):
        featurizer = MolGraphConvFeaturizer(use_edges=True)
        seq_chem = Chem.MolFromSequence(seq)
        if seq_chem is None: return None
        seq_feature = featurizer._featurize(seq_chem)
        graph = DATA.Data(x=torch.Tensor(seq_feature.node_features), 
                          edge_index=torch.LongTensor(seq_feature.edge_index), 
                          edge_attr=torch.Tensor(seq_feature.edge_features))
        return graph

    def process_and_save(self):
        unique_peptides = self.rawdata['peptide'].unique()
        unique_cdr3s = self.rawdata['tcr'].unique()
        
        peptide_cache = {}
        cdr3_cache = {}

        for pep in tqdm(unique_peptides, desc="Processing Peptides"):
            if self.check(pep):
                g = self.generateGraph(pep)
                if g: peptide_cache[pep] = g

        for cdr3 in tqdm(unique_cdr3s, desc="Processing TCRs"):
            if self.check(cdr3):
                g = self.generateGraph(cdr3)
                if g: cdr3_cache[cdr3] = g
        
        self.peptide_cache = peptide_cache
        self.cdr3_cache = cdr3_cache

    def pad_graph(self, graph, min_nodes=20):
        """
        关键修复：如果图节点数少于20，填充虚拟节点，防止模型 Top-K Pooling 崩溃
        """
        num_nodes = graph.num_nodes
        if num_nodes < min_nodes:
            pad_n = min_nodes - num_nodes
            # 创建全0特征的虚拟节点
            pad_x = torch.zeros((pad_n, graph.x.size(1)), dtype=graph.x.dtype)
            # 拼接到原特征矩阵
            graph.x = torch.cat([graph.x, pad_x], dim=0)
            # 不需要添加边 (edge_index)，孤立节点不影响图卷积，只为了凑数
            # 注意：PyG 的 Batch 机制会自动处理 batch 索引
        return graph

    def __getitem__(self, idx):
        row = self.rawdata.loc[idx]
        peptide = row['peptide']
        cdr3 = row['tcr']
        
        # 测试集可能有 label 也可能没有
        label = row['label'] if 'label' in self.rawdata.columns else -1

        if peptide not in self.peptide_cache or cdr3 not in self.cdr3_cache:
            return self.__getitem__(0) # 容错处理

        # 获取图的深拷贝，避免原地修改缓存
        peptide_graph = copy.deepcopy(self.peptide_cache[peptide])
        cdr3_graph = copy.deepcopy(self.cdr3_cache[cdr3])
        
        # 【关键修复】执行填充
        peptide_graph = self.pad_graph(peptide_graph, min_nodes=20)
        cdr3_graph = self.pad_graph(cdr3_graph, min_nodes=20)
        
        return (idx, peptide, cdr3, label, peptide_graph, cdr3_graph)

def collate(batch):
    idxs = [item[0] for item in batch]
    peptides = [item[1] for item in batch]
    tcrs = [item[2] for item in batch]
    labels = [item[3] for item in batch]
    peptide_graphs = Batch.from_data_list([item[4] for item in batch])
    cdr3_graphs = Batch.from_data_list([item[5] for item in batch])
    return idxs, peptides, tcrs, torch.LongTensor(labels), peptide_graphs, cdr3_graphs

# ---------------------------------------------------------
# 3. 指标计算
# ---------------------------------------------------------
def compute_metrics_dict(y_true, y_pred, y_prob):
    try: auc = roc_auc_score(y_true, y_prob)
    except: auc = 0.0
    try: pr_auc = average_precision_score(y_true, y_prob)
    except: pr_auc = 0.0

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except:
        specificity = 0.0

    return {"AUC": auc, "ACC": acc, "MCC": mcc, "F1": f1, "PR_AUC": pr_auc, "Specificity": specificity, "Precision": precision, "Recall": recall}

# ---------------------------------------------------------
# 4. 主程序
# ---------------------------------------------------------
def main():
    # ================= 配置区域 =================
    model_path = './checkpoints_tcr/best_tcr_model.pt'
    # 你刚才报错的文件
    test_csv = '/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/covid_set.csv'
    cache_dir = './cached_graphs_tcr'
    n_bootstrap = 5
    batch_size = 1024
    # ===========================================

    # 自动获取空闲显卡或使用指定显卡
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    args = checkpoint['args']
    args['batchsize'] = batch_size 
    
    model = DeepGCN(args).to(device)
    model.eval()

    print("Adapting state_dict keys...")
    model_keys = model.state_dict().keys()
    saved_state = checkpoint['model']
    saved_state_clean = {k.replace("module.", ""): v for k, v in saved_state.items()}
    
    new_state_dict = {}
    for key in model_keys:
        if key in saved_state_clean:
            new_state_dict[key] = saved_state_clean[key]
        else:
            key_no_module = key.replace(".module.", ".")
            if key_no_module in saved_state_clean:
                new_state_dict[key] = saved_state_clean[key_no_module]
    
    model.load_state_dict(new_state_dict, strict=False)
    print("Model loaded successfully.")

    # --- 加载数据 ---
    print(f"Loading Test Dataset: {test_csv}")
    dataset = pTCR_DataSet_Inference(test_csv, cache_dir=cache_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                        collate_fn=collate, num_workers=4, pin_memory=True)

    # --- 执行推理 ---
    all_trues = []
    all_probs = []
    all_preds = []

    print("Running Inference...")
    with torch.no_grad():
        for _, _, _, labels, pep_graphs, cdr3_graphs in tqdm(loader, desc="Inference"):
            pep_graphs = pep_graphs.to(device)
            cdr3_graphs = cdr3_graphs.to(device)
            
            logits = model(pep_graphs, cdr3_graphs)
            probs = logits[:, 1]
            preds = logits.argmax(dim=1)
            
            all_trues.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    y_true = np.array(all_trues)
    y_prob = np.array(all_probs)
    y_pred = np.array(all_preds)

    print(f"\nInference finished. Total samples: {len(y_true)}")

    # --- Bootstrap 采样 ---
    print(f"\nPerforming Bootstrap Analysis ({n_bootstrap} rounds)...")
    
    bootstrap_results = []
    np.random.seed(42)

    for i in range(n_bootstrap):
        indices = resample(np.arange(len(y_true)), replace=True, n_samples=len(y_true))
        metrics = compute_metrics_dict(y_true[indices], y_pred[indices], y_prob[indices])
        bootstrap_results.append(metrics)
        print(f"Round {i+1}: AUC={metrics['AUC']:.4f} ACC={metrics['ACC']:.4f}")

    # --- 统计结果 ---
    df = pd.DataFrame(bootstrap_results)
    mean_metrics = df.mean()
    std_metrics = df.std()

    print("\n" + "="*60)
    print(f"{'Metric':<15} | {'Mean':<10} | {'Std':<10}")
    print("-" * 60)
    for metric in mean_metrics.index:
        print(f"{metric:<15} | {mean_metrics[metric]:.4f}     | {std_metrics[metric]:.4f}")
    print("="*60)

    # 根据数据集名称保存结果
    dataset_name = os.path.basename(test_csv).split('.')[0]
    output_file = f"bootstrap_results_{dataset_name}.csv"
    df.loc['Mean'] = mean_metrics
    df.loc['Std'] = std_metrics
    df.to_csv(output_file)
    print(f"\nDetailed results saved to {output_file}")

if __name__ == '__main__':
    main()
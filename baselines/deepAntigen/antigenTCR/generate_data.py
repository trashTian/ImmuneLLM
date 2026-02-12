# import os
# import pandas as pd
# import torch
# import pickle
# import copy
# from tqdm import tqdm
# from rdkit import Chem
# from multiprocessing import Pool, cpu_count
# from torch_geometric import data as DATA

# # ================= 配置区域 =================
# CACHE_DIR = '/mnt/lustre/guopeijin/Immune_LLM/code/baselines/deepAntigen/deepAntigen-main/deepAntigen/antigenTCR/generate_data.py'
# BASE_PATH = '/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR'

# # 定义所有需要生成的文件路径及其配置
# DATASETS = [
#     {"name": "train_fold_1",    "path": os.path.join(BASE_PATH, "train_fold_1.csv")},
#     {"name": "val_fold_1",      "path": os.path.join(BASE_PATH, "val_fold_1.csv")},
#     {"name": "independent_set", "path": os.path.join(BASE_PATH, "independent_set.csv")},
#     {"name": "triple_set",      "path": os.path.join(BASE_PATH, "triple_set.csv")},
#     {"name": "covid_set",       "path": os.path.join(BASE_PATH, "covid_set.csv")},
# ]
# # ===========================================

# # 必须导入 featurizer，否则无法生成图
# try:
#     from load_dataset.featurizer import MolGraphConvFeaturizer
# except ImportError as e:
#     print("Error: 找不到 featurizer.py，请确保该脚本在 load_dataset 文件夹的父级目录运行。")
#     exit()

# def check_seq(seq, aa_list):
#     """检查序列合法性"""
#     for aa in seq:
#         if aa not in aa_list:
#             return False
#     return True

# def generate_graph(seq):
#     """RDKit 转 PyG Graph (CPU bound)"""
#     featurizer = MolGraphConvFeaturizer(use_edges=True)
#     seq_chem = Chem.MolFromSequence(seq)
#     if seq_chem is None:
#         return None
#     seq_feature = featurizer._featurize(seq_chem)
    
#     # 只保存 Tensor 数据以减小体积
#     graph = DATA.Data(x=torch.Tensor(seq_feature.node_features), 
#                       edge_index=torch.LongTensor(seq_feature.edge_index), 
#                       edge_attr=torch.Tensor(seq_feature.edge_features))
#     return graph

# def process_single_dataset(config):
#     """处理单个 CSV 文件的主函数"""
#     csv_path = config["path"]
#     dataset_name = config["name"]
#     cache_path = os.path.join(CACHE_DIR, f"{dataset_name}_graphs.pkl")
    
#     print(f"[{dataset_name}] Starting processing...")
    
#     # 1. 检查是否已存在
#     if os.path.exists(cache_path):
#         print(f"[{dataset_name}] Cache already exists at {cache_path}. Skipping.")
#         return

#     # 2. 读取数据
#     if not os.path.exists(csv_path):
#         print(f"[{dataset_name}] Error: File not found: {csv_path}")
#         return
        
#     df = pd.read_csv(csv_path)
#     unique_peptides = df['peptide'].unique()
#     unique_tcrs = df['tcr'].unique()
    
#     aa_list = list('ACDEFGHIKLMNPQRSTVWY')
#     peptide_cache = {}
#     cdr3_cache = {}

#     # 3. 处理 Peptides
#     for pep in tqdm(unique_peptides, desc=f"[{dataset_name}] Peptides", leave=False):
#         if check_seq(pep, aa_list):
#             g = generate_graph(pep)
#             if g: peptide_cache[pep] = g

#     # 4. 处理 TCRs
#     for tcr in tqdm(unique_tcrs, desc=f"[{dataset_name}] TCRs", leave=False):
#         if check_seq(tcr, aa_list):
#             g = generate_graph(tcr)
#             if g: cdr3_cache[tcr] = g
            
#     # 5. 保存缓存
#     print(f"[{dataset_name}] Saving {len(peptide_cache)} peps and {len(cdr3_cache)} tcrs to disk...")
#     os.makedirs(CACHE_DIR, exist_ok=True)
#     with open(cache_path, 'wb') as f:
#         pickle.dump({'peptide': peptide_cache, 'cdr3': cdr3_cache}, f)
    
#     print(f"[{dataset_name}] Done!")

# def main():
#     if not os.path.exists(CACHE_DIR):
#         os.makedirs(CACHE_DIR)
        
#     # 根据数据集数量设定进程数 (最多5个并发，因为只有5个文件)
#     num_processes = min(len(DATASETS), 5)
#     print(f"Starting parallel processing with {num_processes} processes...")
    
#     with Pool(processes=num_processes) as pool:
#         pool.map(process_single_dataset, DATASETS)
    
#     print("\nAll datasets processed successfully!")

# if __name__ == '__main__':
#     main()

import os
# 限制底层线程，防止CPU过载
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import torch
import pickle
from rdkit import Chem, rdBase
import torch.multiprocessing as mp
from torch_geometric import data as DATA
from tqdm import tqdm

# 关闭 RDKit 警告
rdBase.DisableLog('rdApp.*')

# ================= 配置区域 =================
CACHE_DIR = './cached_graphs_tcr'
BASE_PATH = '/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR'

DATASETS = [
    {"name": "train_fold_1",    "path": os.path.join(BASE_PATH, "train_fold_1.csv")},
    {"name": "val_fold_1",      "path": os.path.join(BASE_PATH, "val_fold_1.csv")},
    {"name": "independent_set", "path": os.path.join(BASE_PATH, "independent_set.csv")},
    {"name": "triple_set",      "path": os.path.join(BASE_PATH, "triple_set.csv")},
    {"name": "covid_set",       "path": os.path.join(BASE_PATH, "covid_set.csv")},
]
# ===========================================

try:
    from load_dataset.featurizer import MolGraphConvFeaturizer
except ImportError as e:
    print("Error: 找不到 featurizer.py")
    exit()

def process_seq(seq):
    """单个序列处理函数"""
    if not isinstance(seq, str): return None
    
    aa_list = set('ACDEFGHIKLMNPQRSTVWY')
    for aa in seq:
        if aa not in aa_list:
            return None
            
    try:
        featurizer = MolGraphConvFeaturizer(use_edges=True)
        seq_chem = Chem.MolFromSequence(seq)
        if seq_chem is None:
            return None
        seq_feature = featurizer._featurize(seq_chem)
        
        # 生成图对象
        graph = DATA.Data(x=torch.Tensor(seq_feature.node_features), 
                          edge_index=torch.LongTensor(seq_feature.edge_index), 
                          edge_attr=torch.Tensor(seq_feature.edge_features))
        
        # 【核心修改】在此处直接序列化为 bytes！
        # 这样返回给主进程的是普通字节流，不占用 PyTorch 的共享内存句柄
        return pickle.dumps(graph)
    except:
        return None

def main():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    # 1. 收集序列
    print("Step 1: Loading CSVs and collecting unique sequences...")
    all_peptides = set()
    all_tcrs = set()
    loaded_dfs = {} 

    for config in DATASETS:
        path = config['path']
        if os.path.exists(path):
            df = pd.read_csv(path)
            loaded_dfs[config['name']] = df
            all_peptides.update(df['peptide'].dropna().unique())
            all_tcrs.update(df['tcr'].dropna().unique())
        else:
            print(f"Warning: File not found {path}")

    unique_peptides = list(all_peptides)
    unique_tcrs = list(all_tcrs)
    
    print(f"Total Unique Peptides: {len(unique_peptides)}")
    print(f"Total Unique TCRs:     {len(unique_tcrs)}")

    # 2. 并行计算
    # 【修改】大幅降低 Worker 数量，防止 FD 耗尽。建议 16-32 即可。
    num_workers = 20 
    print(f"\nStep 2: Computing graphs using {num_workers} workers (Spawn + Serialize Mode)...")

    global_pep_graph = {}
    global_tcr_graph = {}

    ctx = mp.get_context('spawn')

    # 处理 Peptide
    with ctx.Pool(num_workers) as p:
        # imap_unordered 稍微快一点
        results = list(tqdm(p.imap(process_seq, unique_peptides, chunksize=100), 
                           total=len(unique_peptides), desc="Processing Peptides"))
    
    for seq, serialized_data in zip(unique_peptides, results):
        if serialized_data is not None:
            # 【核心修改】主进程负责反序列化
            global_pep_graph[seq] = pickle.loads(serialized_data)

    # 处理 TCR
    with ctx.Pool(num_workers) as p:
        results = list(tqdm(p.imap(process_seq, unique_tcrs, chunksize=100), 
                           total=len(unique_tcrs), desc="Processing TCRs"))
            
    for seq, serialized_data in zip(unique_tcrs, results):
        if serialized_data is not None:
            # 【核心修改】主进程负责反序列化
            global_tcr_graph[seq] = pickle.loads(serialized_data)

    print(f"\nGraph computation finished. Valid Peptides: {len(global_pep_graph)}, Valid TCRs: {len(global_tcr_graph)}")

    # 3. 保存
    print("\nStep 3: Saving individual cache files...")
    for config in DATASETS:
        name = config['name']
        if name not in loaded_dfs: continue
            
        df = loaded_dfs[name]
        current_peps = df['peptide'].unique()
        current_tcrs = df['tcr'].unique()
        
        local_pep_cache = {}
        local_tcr_cache = {}
        
        for p in current_peps:
            if p in global_pep_graph: local_pep_cache[p] = global_pep_graph[p]
        for t in current_tcrs:
            if t in global_tcr_graph: local_tcr_cache[t] = global_tcr_graph[t]
        
        save_path = os.path.join(CACHE_DIR, f"{name}_graphs.pkl")
        print(f"-> Saving {name} to {save_path}")
        
        with open(save_path, 'wb') as f:
            pickle.dump({'peptide': local_pep_cache, 'cdr3': local_tcr_cache}, f)

    print("\nAll done!")

if __name__ == '__main__':
    main()
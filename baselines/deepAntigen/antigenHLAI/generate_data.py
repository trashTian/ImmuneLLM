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
CACHE_DIR = './cached_graphs'
BASE_PATH = '/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA'

# 定义需要处理的数据集文件
DATASETS = [
    # {"name": "train_fold_1",    "path": os.path.join(BASE_PATH, "train_fold_1.csv")},
    # {"name": "val_fold_1",      "path": os.path.join(BASE_PATH, "val_fold_1.csv")},
    # {"name": "independent_set", "path": os.path.join(BASE_PATH, "independent_set.csv")},
    # 如果有 external_set 也可以加在这里
    {"name": "external_set",    "path": os.path.join(BASE_PATH, "external_set.csv")},
]
# ===========================================

try:
    from load_dataset.featurizer import MolGraphConvFeaturizer
except ImportError as e:
    print("Error: 找不到 featurizer.py，请确保在正确目录下运行。")
    exit()

def process_seq(seq):
    """单个序列处理函数"""
    if not isinstance(seq, str): return None
    
    # 简单的合法性检查
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
        
        # 【核心】序列化为字节流返回，避开共享内存限制
        return pickle.dumps(graph)
    except:
        return None

def main():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    # 1. 收集所有文件的序列 (全局去重)
    print("Step 1: Loading CSVs and collecting unique sequences...")
    all_peptides = set()
    all_hlas = set()
    loaded_dfs = {} 

    for config in DATASETS:
        path = config['path']
        if os.path.exists(path):
            df = pd.read_csv(path)
            loaded_dfs[config['name']] = df
            # HLA 数据集的列名通常是 'peptide' 和 'HLA'
            if 'peptide' in df.columns:
                all_peptides.update(df['peptide'].dropna().unique())
            if 'HLA' in df.columns:
                all_hlas.update(df['HLA'].dropna().unique())
        else:
            print(f"Warning: File not found {path}")

    unique_peptides = list(all_peptides)
    unique_hlas = list(all_hlas)
    
    print(f"Total Unique Peptides: {len(unique_peptides)}")
    print(f"Total Unique HLA Seqs: {len(unique_hlas)}")

    # 2. 并行计算
    # HLA 序列通常比 TCR 短一点，但数量也不少，设置 32 个 Worker 比较合适
    num_workers = 32
    print(f"\nStep 2: Computing graphs using {num_workers} workers (Spawn Mode)...")

    global_pep_graph = {}
    global_hla_graph = {}

    # 使用 spawn 模式防止死锁
    ctx = mp.get_context('spawn')

    # --- 处理 Peptide ---
    with ctx.Pool(num_workers) as p:
        results = list(tqdm(p.imap(process_seq, unique_peptides, chunksize=200), 
                           total=len(unique_peptides), desc="Processing Peptides"))
    
    for seq, serialized_data in zip(unique_peptides, results):
        if serialized_data is not None:
            global_pep_graph[seq] = pickle.loads(serialized_data)

    # --- 处理 HLA ---
    with ctx.Pool(num_workers) as p:
        results = list(tqdm(p.imap(process_seq, unique_hlas, chunksize=50), 
                           total=len(unique_hlas), desc="Processing HLAs"))
            
    for seq, serialized_data in zip(unique_hlas, results):
        if serialized_data is not None:
            global_hla_graph[seq] = pickle.loads(serialized_data)

    print(f"\nGraph computation finished. Valid Peptides: {len(global_pep_graph)}, Valid HLAs: {len(global_hla_graph)}")

    # 3. 分发并保存
    print("\nStep 3: Saving individual cache files...")
    for config in DATASETS:
        name = config['name']
        if name not in loaded_dfs: continue
            
        df = loaded_dfs[name]
        current_peps = df['peptide'].unique()
        current_hlas = df['HLA'].unique()
        
        local_pep_cache = {}
        local_hla_cache = {}
        
        for p in current_peps:
            if p in global_pep_graph: local_pep_cache[p] = global_pep_graph[p]
        for h in current_hlas:
            if h in global_hla_graph: local_hla_cache[h] = global_hla_graph[h]
        
        save_path = os.path.join(CACHE_DIR, f"{name}_graphs.pkl")
        print(f"-> Saving {name} to {save_path}")
        
        with open(save_path, 'wb') as f:
            pickle.dump({'peptide': local_pep_cache, 'hla': local_hla_cache}, f)

    print("\nAll done! You can now run train_hla_ddp.py.")

if __name__ == '__main__':
    main()
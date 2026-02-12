import os
# 针对 4090 的稳定性环境变量
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, TensorDataset
from Bio.Align import substitution_matrices
from sklearn.metrics import (roc_auc_score, accuracy_score, matthews_corrcoef, 
                             f1_score, average_precision_score, precision_score, 
                             recall_score, confusion_matrix)
import shutil
import warnings

# 4090 稳定性终极设置
# 使用 highest 精度以获得最大数值稳定性
torch.set_float32_matmul_precision('highest') 
torch.backends.cudnn.enabled = False 
warnings.filterwarnings("ignore")

from scripts.model_raw import TEIM
from utils.misc import load_config, calc_auc_aupr
from utils.dataset import SeqLevelDataset

# ================= 路径与配置 (保持不变) =================
PATHS = {
    "train": "/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/train_fold_1.csv",
    "val": "/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/val_fold_1.csv",
    "tests": {
        "independent": "/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/independent_set.csv",
        "triple": "/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/triple_set.csv",
        "covid": "/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/covid_set.csv"
    },
    "config": "configs/seqlevel_all.yml",
    "ae_scratch_ckpt": "./results_teim_scratch/epi_ae_scratch.ckpt",
    "save_dir": "./results_teim_scratch",
}

# ================= 评价指标与数据清洗 (保持 12 位逻辑) =================

def compute_all_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_prob),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred)
    }

# ... (Tokenizer, AutoEncoder, GetBlosumMat 等代码保持上一版本的 Len=12 逻辑) ...
# 注意：确保 encoding_epi_fixed 里的 max_len=12

class Tokenizer:
    def __init__(self):
        self.res_all = list("GAVLI FYW DN EKQM STCP HR".replace(" ",""))
        self.tokens = ['-'] + self.res_all
    def id_list(self, seq):
        return [self.tokens.index(s.upper()) if s.upper() in self.tokens else 0 for s in seq]
    def embedding_mat(self):
        blosum62 = substitution_matrices.load('BLOSUM62')
        mat = np.eye(len(self.tokens))
        for i, res_i in enumerate(self.res_all):
            for j, res_j in enumerate(self.res_all):
                if (res_i, res_j) in blosum62:
                    mat[i+1, j+1] = blosum62[(res_i, res_j)]
        return mat

class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
    def forward(self, x): return x.view(x.shape[0], *self.shape)

class AutoEncoder(nn.Module):
    def __init__(self, tokenizer, dim_hid=32, len_seq=12):
        super().__init__()
        emb_mat = tokenizer.embedding_mat()
        self.embedding_module = nn.Embedding.from_pretrained(torch.FloatTensor(emb_mat), padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Conv1d(emb_mat.shape[1], dim_hid, 3, padding=1), nn.BatchNorm1d(dim_hid), nn.ReLU(),
            nn.Conv1d(dim_hid, dim_hid, 3, padding=1), nn.BatchNorm1d(dim_hid), nn.ReLU()
        )
        self.seq2vec = nn.Sequential(nn.Flatten(), nn.Linear(len_seq * dim_hid, dim_hid), nn.ReLU())
        self.vec2seq = nn.Sequential(nn.Linear(dim_hid, len_seq * dim_hid), nn.ReLU(), View(dim_hid, len_seq))
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(dim_hid, dim_hid, 3, padding=1), nn.BatchNorm1d(dim_hid), nn.ReLU(),
            nn.ConvTranspose1d(dim_hid, dim_hid, 3, padding=1), nn.BatchNorm1d(dim_hid), nn.ReLU()
        )
        self.out_layer = nn.Linear(dim_hid, emb_mat.shape[0])

    def forward(self, inputs):
        x = self.embedding_module(inputs).transpose(1, 2)
        v = self.seq2vec(self.encoder(x))
        out = self.out_layer(self.decoder(self.vec2seq(v)).transpose(1, 2))
        return out

def encoding_epi_fixed(seqs, max_len=12):
    tokenizer = Tokenizer()
    encoding = np.zeros([len(seqs), max_len], dtype='long')
    for i, seq in enumerate(seqs):
        l = len(seq)
        if l in [8, 9]: encoding[i, 1:l+1] = tokenizer.id_list(seq)
        elif l in [10, 11]: encoding[i, 0:l] = tokenizer.id_list(seq)
        else: encoding[i, :min(l, max_len)] = tokenizer.id_list(seq)[:max_len]
    return encoding

# def prepare_data(input_path, output_name):
#     df = pd.read_csv(input_path)
#     mapping = {'tcr': 'cdr3', 'peptide': 'epi', 'label': 'y_true'}
#     df = df.rename(columns=mapping)
#     initial_count = len(df)
#     standard_aas = set("ACDEFGHIKLMNPQRSTVWY")
#     def clean_seq(seq):
#         if not isinstance(seq, str): return None
#         s = seq.strip().upper()
#         if not s.isalpha() or not all(c in standard_aas for c in s): return None
#         return s
#     df['cdr3'] = df['cdr3'].apply(clean_seq); df['epi'] = df['epi'].apply(clean_seq)
#     df = df.dropna(subset=['cdr3', 'epi'])
#     df = df[df['cdr3'].str.len().between(7, 20)]
#     df = df[df['epi'].str.len().between(8, 11)]
#     df = df[['cdr3', 'epi', 'y_true']]
#     os.makedirs("./temp_scratch_data", exist_ok=True)
#     save_path = f"./temp_scratch_data/{output_name}.csv"
#     df.to_csv(save_path, index=False)
#     return "./temp_scratch_data/", output_name

# ================= 核心：数据截断与替换逻辑 (不删除任何行) =================

def prepare_data(input_path, output_name):
    """
    通过截断和字符替换处理数据，确保输出行数与输入完全一致。
    """
    df = pd.read_csv(input_path)
    mapping = {'tcr': 'cdr3', 'peptide': 'epi', 'label': 'y_true'}
    df = df.rename(columns=mapping)
    
    standard_aas = set("ACDEFGHIKLMNPQRSTVWY")
    
    def process_seq(seq, max_len):
        if not isinstance(seq, str):
            return "G" * 8 # 缺失值填充
        
        # 1. 清洗与大写
        s = seq.strip().upper()
        
        # 2. 非法字符替换为 'G' (甘氨酸) 而不是删除样本
        s_clean = "".join([c if c in standard_aas else "G" for c in s])
        
        # 3. 截断 (TEIM 架构硬限制)
        return s_clean[:max_len]

    # 执行处理：CDR3B 截断至 20, Peptide 截断至 11
    df['cdr3'] = df['cdr3'].apply(lambda x: process_seq(x, 20))
    df['epi'] = df['epi'].apply(lambda x: process_seq(x, 11))
    
    # 补齐：如果序列太短（极少见），至少保证有 1 位
    df['cdr3'] = df['cdr3'].apply(lambda x: x if len(x) > 0 else "G")
    df['epi'] = df['epi'].apply(lambda x: x if len(x) > 0 else "G")

    print(f"--- [PROCESSED] {output_name}: Final rows = {len(df)} (Aligned with input) ---")
    
    df = df[['cdr3', 'epi', 'y_true']]
    os.makedirs("./temp_aligned_data", exist_ok=True)
    save_path = f"./temp_aligned_data/{output_name}.csv"
    df.to_csv(save_path, index=False)
    return "./temp_aligned_data/", output_name

# ================= TEIM 系统 (增强防御版) =================

class TEIMSystemScratch(pl.LightningModule):
    def __init__(self, config, ae_ckpt_path):
        super().__init__()
        self.save_hyperparameters(ignore=['ae_ckpt_path'])
        self.teim_model = TEIM(config.model)
        self.lr = float(config.training.lr)
        self.validation_step_outputs = []
        sd = torch.load(ae_ckpt_path, map_location='cpu')
        new_sd = {k.replace('model.', ''): v for k, v in sd.items()}
        self.teim_model.ae_encoder.load_state_dict(new_sd)

    def forward(self, x): 
        # 防御措施 1：强制钳位索引，防止 Embedding 越界触发 CUDA illegal access
        # TEIM 内部索引范围是 0-20
        cdr3, epi = x[0], x[1]
        cdr3 = torch.clamp(cdr3, 0, 20)
        epi = torch.clamp(epi, 0, 20)
        return self.teim_model([cdr3, epi])['seqlevel_out']

    def training_step(self, batch, batch_idx):
        pred = self([batch['cdr3'], batch['epi']])
        
        # 防御措施 2：数值稳定性钳位，防止 BCE 出现 NaN 触发 cublasGEMM 失败
        pred = torch.clamp(pred, 1e-7, 1.0 - 1e-7)
        
        loss = torch.nn.functional.binary_cross_entropy(pred.view(-1), batch['labels'].float())
        
        # 实时监控 NaN
        if torch.isnan(loss):
            print(f"NaN detected in Loss at batch {batch_idx}, skipping...")
            return None
            
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self([batch['cdr3'], batch['epi']])
        out = {'y_true': batch['labels'].detach().cpu(), 'y_pred': pred.detach().cpu()}
        self.validation_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs: return
        y_t = torch.cat([x['y_true'] for x in self.validation_step_outputs]).numpy()
        y_p = torch.cat([x['y_pred'] for x in self.validation_step_outputs]).numpy()
        if len(np.unique(y_t)) > 1:
            self.log('val_auc', roc_auc_score(y_t, y_p), prog_bar=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self): return optim.Adam(self.parameters(), lr=self.lr)

# ================= 主流程: Bootstrap 推理 (保持不变) =================

def run_scratch_experiment():
    os.makedirs(PATHS["save_dir"], exist_ok=True)
    pl.seed_everything(0)

    # -------- Phase 1: AE Training --------
    print("\n>>>> PHASE 1: AE Training (Len=12)")
    train_df = pd.read_csv(PATHS["train"])
    unique_epis = [e for e in train_df['peptide'].unique().tolist() if isinstance(e, str) and len(e) <= 12]
    ae_inputs = torch.tensor(encoding_epi_fixed(unique_epis), dtype=torch.long)
    ae_loader = DataLoader(TensorDataset(ae_inputs, ae_inputs), batch_size=1024, shuffle=True)
    ae_model = AutoEncoder(Tokenizer(), len_seq=12).cuda()
    ae_opt = optim.Adam(ae_model.parameters(), lr=1e-5)
    ae_crit = nn.CrossEntropyLoss(ignore_index=0)
    for epoch in range(10):
        ae_model.train()
        for b_in, b_tg in ae_loader:
            b_in, b_tg = b_in.cuda(), b_tg.cuda()
            ae_opt.zero_grad()
            loss = ae_crit(ae_model(b_in).view(-1, 21), b_tg.view(-1))
            loss.backward(); ae_opt.step()
    torch.save(ae_model.state_dict(), PATHS["ae_scratch_ckpt"])

    # -------- Phase 2: TEIM Training --------
    print("\n>>>> PHASE 2: TEIM Training")
    config = load_config(PATHS["config"])
    tp, tf = prepare_data(PATHS["train"], "train_scratch")
    vp, vf = prepare_data(PATHS["val"], "val_scratch")
    class DataConfig:
        def __init__(self, p, f): self.path, self.file_list = p, [f]
    # train_loader = DataLoader(SeqLevelDataset(DataConfig(tp, tf)), batch_size=config.training.batch_size, shuffle=True, num_workers=0, drop_last=True)
    # val_loader = DataLoader(SeqLevelDataset(DataConfig(vp, vf)), batch_size=config.training.batch_size, shuffle=False, num_workers=0)
    train_loader = DataLoader(SeqLevelDataset(DataConfig(tp, tf)), batch_size=1024, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(SeqLevelDataset(DataConfig(vp, vf)), batch_size=1024, shuffle=False, num_workers=4)
    model = TEIMSystemScratch(config, PATHS["ae_scratch_ckpt"])
    
    ckpt_cb = ModelCheckpoint(monitor='val_auc', mode='max', save_top_k=1, filename='best-scratch-teim')
    
    # 增加 EarlyStopping 防止后期模型震荡
    es_cb = EarlyStopping(monitor='val_auc', patience=1, mode='max')

    trainer = pl.Trainer(
        accelerator="gpu", 
        devices=1, 
        max_epochs=config.training.epochs, 
        callbacks=[ckpt_cb, es_cb], 
        default_root_dir=PATHS["save_dir"], 
        # 防御措施 3：收紧梯度剪裁，防止数值爆炸触发 GEMM 失败
        gradient_clip_val=0.2
    )
    
    trainer.fit(model, train_loader, val_loader)

    # -------- Phase 3: Bootstrap Testing --------
    print("\n>>>> PHASE 3: Bootstrap Testing (5 iterations)")
    best_model = TEIMSystemScratch.load_from_checkpoint(ckpt_cb.best_model_path, config=config, ae_ckpt_path=PATHS["ae_scratch_ckpt"])
    best_model.eval().cuda()

    summary_results = []

    for name, path in PATHS["tests"].items():
        print(f"Testing on {name}...")
        tp, tf = prepare_data(path, f"test_{name}_scratch")
        loader = DataLoader(SeqLevelDataset(DataConfig(tp, tf)), batch_size=config.training.batch_size, num_workers=0)

        raw_preds, raw_trues = [], []
        with torch.no_grad():
            for b in loader:
                p = best_model([b['cdr3'].cuda(), b['epi'].cuda()])
                raw_preds.extend(p.detach().cpu().numpy().flatten())
                raw_trues.extend(b['labels'].numpy().flatten())
        
        raw_preds = np.array(raw_preds)
        raw_trues = np.array(raw_trues)

        # Bootstrap 采样 5 次
        boot_metrics = []
        for i in range(5):
            np.random.seed(i) # 确保可复现
            indices = np.random.choice(len(raw_trues), len(raw_trues), replace=True)
            res = compute_all_metrics(raw_trues[indices], raw_preds[indices])
            boot_metrics.append(res)
        
        # 计算统计量
        df_boot = pd.DataFrame(boot_metrics)
        means = df_boot.mean()
        stds = df_boot.std()

        print(f"\nFinal Stats for {name}:")
        for m in means.index:
            print(f"  {m:12s}: {means[m]:.4f} ± {stds[m]:.4f}")
            summary_results.append({
                "Dataset": name,
                "Metric": m,
                "Mean": means[m],
                "Std": stds[m]
            })
        
        # 保存该测试集的原始预测
        res_df = pd.read_csv(os.path.join(tp, f"{tf}.csv"))
        res_df['y_prob'] = raw_preds
        res_df.to_csv(os.path.join(PATHS["save_dir"], f"results_scratch_{name}.csv"), index=False)

    # 保存汇总指标
    pd.DataFrame(summary_results).to_csv(os.path.join(PATHS["save_dir"], "bootstrap_metrics_summary.csv"), index=False)
    shutil.rmtree("./temp_scratch_data")

if __name__ == "__main__":
    run_scratch_experiment()
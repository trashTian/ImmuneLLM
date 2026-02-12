import os

# 必须在 import torch 之前强制设置环境变量
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
import shutil
import warnings

# ================= 4090 稳定性终极设置 =================
# 1. 设置矩阵乘法精度
torch.set_float32_matmul_precision('high')

# 2. 【核心修复】完全禁用 CuDNN
# 很多 4090 的 illegal memory access 报错是由 CuDNN 的优化内核在特定形状输入下引起的。
# 禁用它会强制使用原生算子，速度略慢（约10-20%），但能彻底消除该报错。
torch.backends.cudnn.enabled = False 

warnings.filterwarnings("ignore")

from scripts.model_raw import TEIM
from utils.misc import load_config, calc_auc_aupr
from utils.dataset import SeqLevelDataset

# ================= 路径定义 =================
PATHS = {
    "ae_ckpt": "/mnt/lustre/guopeijin/Immune_LLM/code/baselines/TEIM/ckpt/epi_ae.ckpt",
    "train": "/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/train_fold_1.csv",
    "val": "/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/val_fold_1.csv",
    "tests": {
        "independent": "/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/independent_set.csv",
        "triple": "/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/triple_set.csv",
        "covid": "/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/covid_set.csv"
    },
    "config": "configs/seqlevel_all.yml",
    "save_dir": "./results_teim_pretrained",
}

# ================= 极致数据清洗 =================
def prepare_data(input_path, output_name):
    df = pd.read_csv(input_path)
    # 适配你的新列名 mapping
    mapping = {'tcr': 'cdr3', 'peptide': 'epi', 'label': 'y_true'}
    df = df.rename(columns=mapping)
    initial_count = len(df)
    
    # 仅允许标准 20 种氨基酸
    standard_aas = set("ACDEFGHIKLMNPQRSTVWY")
    
    def clean_seq(seq):
        if not isinstance(seq, str): return None
        s = seq.strip().upper()
        if len(s) == 0: return None
        if not all(c in standard_aas for c in s): return None
        return s

    df['cdr3'] = df['cdr3'].apply(clean_seq)
    df['epi'] = df['epi'].apply(clean_seq)
    df = df.dropna(subset=['cdr3', 'epi'])
    
    # 严格遵循 TEIM 索引范围
    df = df[df['cdr3'].str.len().between(7, 20)]
    df = df[df['epi'].str.len().between(8, 11)]
    
    filtered_count = len(df)
    print(f"--- [CLEANING] {output_name}: {initial_count} -> {filtered_count} ---")
    
    df = df[['cdr3', 'epi', 'y_true']]
    os.makedirs("./temp_data", exist_ok=True)
    save_path = f"./temp_data/{output_name}.csv"
    df.to_csv(save_path, index=False)
    return "./temp_data/", output_name

# ================= 模型系统 =================
class TEIMSystem(pl.LightningModule):
    def __init__(self, config, ae_ckpt_path=None):
        super().__init__()
        self.save_hyperparameters(ignore=['ae_ckpt_path'])
        self.config = config
        self.teim_model = TEIM(config.model)
        self.lr = float(config.training.lr)
        self.validation_step_outputs = []

        if ae_ckpt_path and os.path.exists(ae_ckpt_path):
            print(f"--- Loading AE weights from {ae_ckpt_path} ---")
            sd = torch.load(ae_ckpt_path, map_location='cpu')
            new_sd = {k.replace('model.', ''): v for k, v in sd.items()}
            try:
                self.teim_model.ae_encoder.load_state_dict(new_sd)
                print("AE weights loaded successfully.")
            except Exception as e:
                print(f"AE loading failed: {e}")

    def forward(self, x):
        return self.teim_model(x)['seqlevel_out']

    def training_step(self, batch, batch_idx):
        cdr3, epi, labels = batch['cdr3'], batch['epi'], batch['labels']
        
        # 实时索引越界检查 (预防性)
        if (cdr3 >= 21).any() or (epi >= 21).any():
            print(f"Index out of bounds detected in batch {batch_idx}")
            return None
            
        pred = self([cdr3, epi])
        loss = torch.nn.functional.binary_cross_entropy(pred.view(-1), labels.float())
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        cdr3, epi, labels = batch['cdr3'], batch['epi'], batch['labels']
        pred = self([cdr3, epi])
        out = {'y_true': labels.detach().cpu(), 'y_pred': pred.detach().cpu()}
        self.validation_step_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs: return
        y_true = torch.cat([x['y_true'] for x in self.validation_step_outputs]).numpy()
        y_pred = torch.cat([x['y_pred'] for x in self.validation_step_outputs]).numpy()
        if len(np.unique(y_true)) > 1:
            auc, aupr = calc_auc_aupr(y_true, y_pred)
            self.log('val_auc', auc, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# ================= 运行逻辑 =================
def run_experiment():
    config = load_config(PATHS["config"])
    pl.seed_everything(42)

    # 数据准备
    train_path, train_file = prepare_data(PATHS["train"], "train")
    val_path, val_file = prepare_data(PATHS["val"], "val")
    
    class DataConfig:
        def __init__(self, p, f): self.path, self.file_list = p, [f]

    train_ds = SeqLevelDataset(DataConfig(train_path, train_file))
    val_ds = SeqLevelDataset(DataConfig(val_path, val_file))
    
    # 将 num_workers 改为 0 是最稳妥的，防止多线程与显存冲突
    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, 
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, 
                            shuffle=False, num_workers=0, drop_last=False)

    model = TEIMSystem(config, PATHS["ae_ckpt"])
    
    checkpoint_callback = ModelCheckpoint(monitor='val_auc', mode='max', save_top_k=1, filename='best-teim')
    early_stop_callback = EarlyStopping(monitor='val_auc', patience=15, mode='max')

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config.training.epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        default_root_dir=PATHS["save_dir"],
        # 梯度剪裁，防止反向传播数值溢出
        gradient_clip_val=0.5,
        precision=32
    )

    print("--- Training Started (CuDNN Disabled for 4090 Stability) ---")
    trainer.fit(model, train_loader, val_loader)

    print("\n--- Testing Started ---")
    best_model = TEIMSystem.load_from_checkpoint(checkpoint_callback.best_model_path, config=config)
    best_model.eval()
    best_model.cuda()

    for name, path in PATHS["tests"].items():
        t_path, t_file = prepare_data(path, f"test_{name}")
        test_ds = SeqLevelDataset(DataConfig(t_path, t_file))
        test_loader = DataLoader(test_ds, batch_size=config.training.batch_size, shuffle=False, num_workers=0)

        preds, trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                p = best_model([batch['cdr3'].cuda(), batch['epi'].cuda()])
                preds.extend(p.detach().cpu().numpy().flatten())
                trues.extend(batch['labels'].numpy().flatten())

        auc, aupr = calc_auc_aupr(np.array(trues), np.array(preds))
        print(f"RESULT [{name}]: AUC = {auc:.4f}")
        
        res_df = pd.read_csv(os.path.join(t_path, f"{t_file}.csv"))
        res_df['y_prob'] = preds
        res_df.to_csv(os.path.join(PATHS["save_dir"], f"results_{name}.csv"), index=False)

    shutil.rmtree("./temp_data")

if __name__ == "__main__":
    run_experiment()
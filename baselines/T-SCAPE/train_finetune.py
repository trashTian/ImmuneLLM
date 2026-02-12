import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
from transformers import get_linear_schedule_with_warmup
# ================= 导入 T-SCAPE 模块 =================
# 假设 io_utils_fused 里有 CSVDataset (训练用) 和 Collater
# 如果报错 ImportError，请检查 io_utils_fused.py 里具体的类名
# 我们这里尝试复用 CSVDataset_test，因为它已经能返回 label (frac)
from src.io_utils_fused import CSVDataset_test, Collater_test
from src.constants import PAD, PROTEIN_ALPHABET

# 导入模型定义
from src.model_fused import task3, task9

# ================= 自定义配置 =================
class Config:
    def __init__(self, args):
        self.d_model = 280
        self.embedding_dim = 280
        self.n_tokens = 29
        self.kernel_size = 1
        self.n_layers = 6
        self.r = 1
        self.lr = 1e-2  # 论文微调阶段推荐的学习率
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 辅助函数 =================
def load_pretrained_weights(model, checkpoint_path, device):
    print(f"Loading pretrained weights from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    model_dict = model.state_dict()
    new_state_dict = {}
    
    loaded_count = 0
    
    print("🔍 Filtering weights: ONLY loading 'shared_encoder'...")
    
    for k, v in state_dict.items():
        k_clean = k.replace('module.', '')
        
        # === 核心修改：只加载 Shared Encoder ===
        # 我们跳过所有 task9_encoder, task9_decoder, task3... 的权重
        # 让它们保持随机初始化
        if 'shared_encoder' not in k_clean and 'shared_encoder' not in k:
            continue
        # ===================================

        # 匹配逻辑
        if k_clean in model_dict and v.shape == model_dict[k_clean].shape:
            new_state_dict[k_clean] = v
            loaded_count += 1
        else:
            # 模糊匹配
            for model_k in model_dict.keys():
                if (k_clean in model_k or model_k in k_clean) and 'shared_encoder' in model_k:
                    if model_dict[model_k].shape == v.shape:
                        new_state_dict[model_k] = v
                        loaded_count += 1
                        break

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    
    print(f"✅ Successfully loaded {loaded_count} layers (Shared Encoder ONLY).")
    print(f"   Total model layers: {len(model_dict)}")
    print("   Task specific layers are now Randomly Initialized.")

# ================= 验证函数 =================
def validate(model, val_loader, device, task_id):
    model.eval()
    preds = []
    targets = []
    
    with torch.no_grad():
        for i, (src, m1, m2, tcr, frac, p_lens, mhcs) in enumerate(val_loader):
            src, m1, m2, tcr = src.to(device), m1.to(device), m2.to(device), tcr.to(device)
            
            # [修正] frac 是 list，不需要 .to(device) 因为我们这里只用来计算 AUC
            # 如果你要计算验证集的 loss，才需要转成 tensor
            
            output = model(src, m1, m2, tcr=tcr, task=[task_id])
            logits = output[-1]
            probs = torch.sigmoid(logits)
            
            preds.extend(probs.cpu().numpy())
            
            # [修正] 直接 extend list，不需要 .cpu().numpy()
            targets.extend(frac)
            
    try:
        auc = roc_auc_score(targets, preds)
    except:
        auc = 0.5
    return auc

# ================= 训练函数 =================
def train(args):
    config = Config(args)
    # [建议] 强制把 Batch Size 限制在合理范围，防止梯度稀释
    real_batch_size = min(config.batch_size, 2048) 
    print(f"🚀 Training with: Device={config.device} | BS={real_batch_size} | LR={config.lr}")
    
    # 1. 准备数据
    print("Preparing Data...")
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    
    # 预处理
    for df in [train_df, val_df]:
        df['peptide'] = df['peptide'].fillna("").astype(str)
        df['pseudo'] = df.get('pseudo', pd.Series([""]*len(df))).fillna("").astype(str)
        df['CDR3b'] = df.get('CDR3b', pd.Series([""]*len(df))).fillna("").astype(str)
        df['task'] = 1
        df['mhc'] = ""
        # 长度过滤
        df.drop(df[df['peptide'].apply(len) > 20].index, inplace=True)
        if args.task_type == 'ptcr_ba':
            df.drop(df[df['CDR3b'].apply(len) > 20].index, inplace=True)
        df.reset_index(drop=True, inplace=True)

    train_dataset = CSVDataset_test(train_df)
    val_dataset = CSVDataset_test(val_df)
    collater = Collater_test(alphabet=PROTEIN_ALPHABET, pad=True, backwards=False, pad_token=PAD)
    
    # 使用调整后的 Batch Size
    train_loader = DataLoader(train_dataset, batch_size=real_batch_size, shuffle=True, collate_fn=collater)
    val_loader = DataLoader(val_dataset, batch_size=real_batch_size, shuffle=False, collate_fn=collater)
    
    # 2. 初始化模型
    print(f"Initializing Model for task: {args.task_type}")
    if args.task_type == 'pmhc_ba_I':
        model = task3(d_model=config.d_model, n_tokens=config.n_tokens, kernel_size=config.kernel_size,
                      n_layers=config.n_layers, d_embedding=config.embedding_dim, r=config.r, mask_condition=False)
        task_id = 3
    elif args.task_type == 'ptcr_ba':
        model = task9(d_model=config.d_model, n_tokens=config.n_tokens, kernel_size=config.kernel_size,
                      n_layers=config.n_layers, d_embedding=config.embedding_dim, r=config.r, mask_condition=False)
        task_id = 9
    else:
        raise ValueError("Unsupported task type")
        
    model.to(config.device)
    
    # 3. 加载预训练权重
    if args.pretrained_path:
        load_pretrained_weights(model, args.pretrained_path, config.device)
    
    # ================= 核心修改 START =================
    # A. 冻结 Shared Encoder
    print("❄️  Freezing Shared Encoder parameters...")
    for param in model.shared_encoder.parameters():
        param.requires_grad = False
        
    # B. 设置优化器 (只定义一次！)
    # 过滤出 requires_grad=True 的参数 (即 Task Encoder/Decoder)
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    print(f"🔥 Optimizing {len(trainable_params)} tensor groups (Task Head Only). LR={config.lr}")
    
    optimizer = optim.Adam(trainable_params, lr=config.lr)
    
    # C. 添加 Scheduler (关键：防止大LR导致震荡，同时保证后期收敛)
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * 0.1) # 10% Warmup
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    # ================= 核心修改 END =================
    
    criterion = nn.BCEWithLogitsLoss() 
    # [已删除] 重复定义的 optimizer
    
    # 5. 训练循环
    best_auc = 0.0
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for i, (src, m1, m2, tcr, frac, p_lens, mhcs) in enumerate(pbar):
            src, m1, m2, tcr = src.to(config.device), m1.to(config.device), m2.to(config.device), tcr.to(config.device)
            labels = torch.tensor(frac).float().to(config.device).unsqueeze(1) 
            
            optimizer.zero_grad()
            
            output = model(src, m1, m2, tcr=tcr, task=[task_id])
            logits = output[-1]
            
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step() # 更新 LR
            
            total_loss += loss.item()
            
            # 显示当前 LR 和 Loss
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.5f}"})
            
        # Validation
        val_auc = validate(model, val_loader, config.device, task_id)
        print(f"Epoch {epoch+1} finished. Train Loss: {total_loss/len(train_loader):.4f}, Val AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            save_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"✅ New best model saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--val_csv', type=str, required=True, help='Path to validation CSV')
    parser.add_argument('--task_type', type=str, required=True, choices=['pmhc_ba_I', 'ptcr_ba'], help='Task type')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained .pt checkpoint')
    parser.add_argument('--output_dir', type=str, default='./trained_models', help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=10000)
    
    args = parser.parse_args() # 4096
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    train(args)


    """
    # hla_i
    python train_finetune.py \
  --train_csv /mnt/lustre/guopeijin/Immune_LLM/code/baselines/titanian/T-SCAPE-main/train_datas/hal_i/train_fold_1.csv \
  --val_csv /mnt/lustre/guopeijin/Immune_LLM/code/baselines/titanian/T-SCAPE-main/train_datas/hal_i/val_fold_1.csv \
  --task_type pmhc_ba_I \
  --pretrained_path /mnt/lustre/guopeijin/Immune_LLM/code/baselines/titanian/T-SCAPE-main/checkpoints/models/Small_OAS_el-fused_ADV1.0_60.pt \
  --epochs 10 \
  --output_dir ./trained_models/pmhc_ba_I


    # tcr
    python train_finetune.py \
  --train_csv /mnt/lustre/guopeijin/Immune_LLM/code/baselines/titanian/T-SCAPE-main/train_datas/tcr/train_fold_1.csv \
  --val_csv /mnt/lustre/guopeijin/Immune_LLM/code/baselines/titanian/T-SCAPE-main/train_datas/tcr/val_fold_1.csv \
  --task_type ptcr_ba \
  --pretrained_path /mnt/lustre/guopeijin/Immune_LLM/code/baselines/titanian/T-SCAPE-main/checkpoints/models/Uni_el-fused_ADV1.0_0.pt \
  --epochs 10 \
  --output_dir ./trained_models/ptcr_ba
    """

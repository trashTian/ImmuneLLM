import os
import time
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from collections import Counter
from sklearn import metrics
from tqdm import tqdm

# ==========================================
# 1. 配置与工具 (Configuration & Utils)
# ==========================================

class Config:
    """超参数配置类"""
    def __init__(self):
        # 路径配置 (请根据实际情况修改)
        self.vocab_path = 'Transformer_vocab_dict.npy'
        self.data_dir = '/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA'
        self.save_dir = './model/pHLAIformer/'
        
        # 数据参数
        self.pep_max_len = 15
        self.hla_max_len = 34
        self.vocab_size = 23  # 默认值，加载字典后会更新
        
        # 模型参数
        self.d_model = 64
        self.d_ff = 512
        self.n_heads = 1     # 原代码循环里的变量，建议固定或通过参数传入
        self.n_layers = 1
        self.dropout = 0.1
        self.use_mask_in_interaction = False
        
        # 训练参数
        self.seed = 19961231
        self.batch_size = 1024
        self.epochs = 50
        self.learning_rate = 1e-3
        self.threshold = 0.5
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __str__(self):
        return str(self.__dict__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_metrics(y_true, y_prob, threshold=0.5):
    """计算评估指标"""
    y_pred = [1 if p > threshold else 0 for p in y_prob]
    
    # 基础指标
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    
    # 防止除零错误
    denom = (tp+fn)*(tn+fp)*(tp+fp)*(tn+fn)
    mcc = ((tp*tn) - (fn*fp)) / np.sqrt(denom) if denom > 0 else 0.0
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    auc = metrics.roc_auc_score(y_true, y_prob)
    prec_curve, rec_curve, _ = metrics.precision_recall_curve(y_true, y_prob)
    aupr = metrics.auc(rec_curve, prec_curve)
    
    return {
        "acc": acc, "mcc": mcc, "auc": auc, "aupr": aupr,
        "f1": f1, "recall": recall, "precision": precision,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }

# ==========================================
# 2. 数据处理 (Data Handling)
# ==========================================

class Tokenizer:
    """处理序列到ID的映射"""
    def __init__(self, vocab_path=None):
        self.vocab = None
        
        # 尝试加载外部文件
        if vocab_path and os.path.exists(vocab_path):
            try:
                # 尝试加载
                self.vocab = np.load(vocab_path, allow_pickle=True).item()
                print(f"Successfully loaded vocab from {vocab_path}")
            except Exception as e:
                # 捕获 numpy 版本不兼容导致的加载错误
                print(f"Warning: Failed to load {vocab_path} due to error: {e}")
                print("Switching to default vocabulary.")
                self.vocab = None

        # 如果加载失败或文件不存在，构建默认字典
        if self.vocab is None:
            # 标准氨基酸字典 + Padding
            # 注意：必须确保 '-' 映射为 0，这与原来的逻辑一致
            chars = ['-'] + list("ACDEFGHIKLMNPQRSTVWY")
            self.vocab = {c: i for i, c in enumerate(chars)}
            
            # 为了兼容性，也可以添加一些未知字符的映射（可选）
            # self.vocab['X'] = 0 
            
        self.pad_token = self.vocab.get('-', 0)

    def encode(self, seq, max_len):
        # 1. 先截断：防止序列过长 (Truncate if longer than max_len)
        seq = seq[:max_len]
        
        # 2. 再填充：防止序列过短 (Pad if shorter than max_len)
        seq = seq.ljust(max_len, '-')
        
        # 3. 映射为ID
        return [self.vocab.get(c, 0) for c in seq] 
    
    def __len__(self):
        return len(self.vocab)

class PHLADataset(Data.Dataset):
    def __init__(self, df, tokenizer, config):
        self.pep_inputs = []
        self.hla_inputs = []
        self.labels = []
        
        # 【修改点】处理列名兼容性
        # 如果数据里是 'HLA' 就用 'HLA'，如果是 'HLA_sequence' 就用 'HLA_sequence'
        if 'HLA' in df.columns:
            hla_col = 'HLA'
        elif 'HLA_sequence' in df.columns:
            hla_col = 'HLA_sequence'
        else:
            raise ValueError(f"CSV must contain 'HLA' or 'HLA_sequence' column. Found: {df.columns}")

        # 确保 peptide 列存在
        if 'peptide' not in df.columns:
             # 有可能 CSV 带表头但因为空格等问题没读对，打印出来检查
            raise ValueError(f"CSV must contain 'peptide' column. Found: {df.columns}")

        # 使用正确的列名进行迭代
        for pep, hla, label in zip(df['peptide'], df[hla_col], df['label']):
            self.pep_inputs.append(tokenizer.encode(str(pep), config.pep_max_len))
            self.hla_inputs.append(tokenizer.encode(str(hla), config.hla_max_len))
            self.labels.append(label)
            
        self.pep_inputs = torch.LongTensor(self.pep_inputs)
        self.hla_inputs = torch.LongTensor(self.hla_inputs)
        self.labels = torch.LongTensor(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.pep_inputs[idx], self.hla_inputs[idx], self.labels[idx]

# ==========================================
# 3. 模型组件 (Model Components)
# ==========================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model] -> transpose to [seq_len, batch, d_model] for PE addition
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x).transpose(0, 1) # Return [batch, seq, d_model]

def get_attn_pad_mask(seq_q, seq_k, pad_idx=0):
    """
    seq_q: [batch_size, len_q]
    seq_k: [batch_size, len_k]
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(pad_idx) is True where padding exists. We want mask=True to indicate "MASKED OUT" (filled with -inf)
    # But PyTorch MultiheadAttention usually takes mask where True is "ignore".
    # Manual implementation often fills where Mask is True.
    pad_attn_mask = seq_k.data.eq(pad_idx).unsqueeze(1) # [batch, 1, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k) # [batch, len_q, len_k]

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        # Q: [batch, n_heads, len_q, d_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.attn_layer = ScaledDotProductAttention(self.d_k)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        
        # Linear projections & split heads
        # [batch, seq, d_model] -> [batch, seq, heads, d_k] -> [batch, heads, seq, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if attn_mask is not None:
            # [batch, seq, seq] -> [batch, 1, seq, seq] -> broadcast to heads
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        context, attn = self.attn_layer(Q, K, V, attn_mask)
        
        # Concat heads
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        
        return self.layer_norm(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return self.layer_norm(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class Encoder(nn.Module):
    """标准的Transformer Encoder"""
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff):
        super().__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

    def forward(self, enc_inputs):
        # enc_inputs: [batch, seq_len]
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        
        attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            attns.append(enc_self_attn)
        return enc_outputs, attns

class InteractionLayer(nn.Module):
    """
    原名为 Decoder，实为拼接后的交互层。
    接收拼接后的 Peptied+HLA Embedding，再次进行 Self-Attention。
    """
    def __init__(self, d_model, n_layers, n_heads, d_ff, use_mask=True):
        super().__init__()
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.use_mask = use_mask

    def forward(self, inputs, inputs_idx=None):
        # inputs: [batch, total_len, d_model] (Concatenated)
        # inputs_idx: [batch, total_len] (Concatenated indices for masking)
        
        # 再次添加位置编码，因为拼接破坏了原始位置信息
        dec_outputs = self.pos_emb(inputs)
        
        if self.use_mask and inputs_idx is not None:
            dec_self_attn_pad_mask = get_attn_pad_mask(inputs_idx, inputs_idx)
        else:
            # 原代码逻辑：全为 False (不Mask)
            b, l, _ = inputs.shape
            dec_self_attn_pad_mask = torch.zeros((b, l, l), dtype=torch.bool, device=inputs.device)

        attns = []
        for layer in self.layers:
            dec_outputs, attn = layer(dec_outputs, dec_self_attn_pad_mask)
            attns.append(attn)
            
        return dec_outputs, attns

# ==========================================
# 4. 主模型 (TransPHLA)
# ==========================================

class TransPHLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pep_encoder = Encoder(config.vocab_size, config.d_model, config.n_layers, config.n_heads, config.d_ff)
        self.hla_encoder = Encoder(config.vocab_size, config.d_model, config.n_layers, config.n_heads, config.d_ff)
        
        # 交互层 (原 Decoder)
        self.interaction_layer = InteractionLayer(config.d_model, config.n_layers, config.n_heads, config.d_ff, 
                                                  use_mask=config.use_mask_in_interaction)
        
        total_len = config.pep_max_len + config.hla_max_len
        self.projection = nn.Sequential(
            nn.Linear(total_len * config.d_model, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 2)
        )

    def forward(self, pep_inputs, hla_inputs):
        # 1. 独立编码
        pep_enc_outputs, _ = self.pep_encoder(pep_inputs)
        hla_enc_outputs, _ = self.hla_encoder(hla_inputs)
        
        # 2. 拼接
        enc_outputs = torch.cat((pep_enc_outputs, hla_enc_outputs), 1)
        # 同时也拼接原始索引，用于生成 Mask (如果开启的话)
        inputs_idx = torch.cat((pep_inputs, hla_inputs), 1)
        
        # 3. 交互 (Cross-molecular interaction simulation)
        dec_outputs, dec_attns = self.interaction_layer(enc_outputs, inputs_idx)
        
        # 4. 展平与分类
        flat_outputs = dec_outputs.view(dec_outputs.shape[0], -1)
        logits = self.projection(flat_outputs)
        
        return logits, dec_attns

# ==========================================
# 5. 训练器 (Trainer)
# ==========================================

class Trainer:
    def __init__(self, model, config, optimizer, criterion):
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = optimizer
        self.criterion = criterion
    
    def run_epoch(self, loader, is_train=True):
        if is_train:
            self.model.train()
        else:
            self.model.eval()
            
        loss_list = []
        y_true_list = []
        y_prob_list = []
        
        with torch.set_grad_enabled(is_train):
            for pep, hla, label in tqdm(loader, leave=False):
                pep = pep.to(self.config.device)
                hla = hla.to(self.config.device)
                label = label.to(self.config.device)
                
                outputs, _ = self.model(pep, hla)
                loss = self.criterion(outputs, label)
                
                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                loss_list.append(loss.item())
                y_true_list.extend(label.cpu().numpy())
                y_prob_list.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy())
                
        metrics_dict = get_metrics(y_true_list, y_prob_list, self.config.threshold)
        avg_loss = np.mean(loss_list)
        return avg_loss, metrics_dict

    def train(self, train_loader, val_loader):
        best_metric = 0
        best_epoch = -1
        
        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)
            
        print(f"Start training on {self.config.device}...")
        
        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            train_loss, train_metrics = self.run_epoch(train_loader, is_train=True)
            val_loss, val_metrics = self.run_epoch(val_loader, is_train=False)
            dt = time.time() - t0
            
            # 使用 AUC 和 Accuracy 的均值作为 Model Selection 标准
            current_metric = (val_metrics['auc'] + val_metrics['acc']) / 2
            
            print(f"Epoch {epoch}/{self.config.epochs} | Time: {dt:.1f}s | "
                  f"Train Loss: {train_loss:.4f} AUC: {train_metrics['auc']:.4f} | "
                  f"Val Loss: {val_loss:.4f} AUC: {val_metrics['auc']:.4f}")
            
            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                save_path = os.path.join(self.config.save_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), save_path)
                print(f"  >>> Best model saved at epoch {best_epoch} (Metric: {best_metric:.4f})")
                
        print("Training Finished.")
        return best_epoch

# ==========================================
# 6. 主程序 (Main)
# ==========================================

def load_data_from_csv(csv_path, tokenizer, config):
    if not os.path.exists(csv_path):
        # 假数据生成逻辑保持不变...
        print(f"File {csv_path} not found. Generating dummy data...")
        # ... (省略假数据生成代码)
        # df = pd.DataFrame(data)
    else:
        # 【修改点】去掉 index_col=0，或者设为 False/None
        # 这样 peptide 就会被正确识别为一列，而不是索引
        df = pd.read_csv(csv_path, index_col=False) 
        
    return PHLADataset(df, tokenizer, config)

def main():
    # 1. 初始化
    config = Config()
    set_seed(config.seed)
    
    # 2. 准备数据
    tokenizer = Tokenizer(config.vocab_path)
    config.vocab_size = len(tokenizer) # 更新配置中的字典大小
    
    # 这里假设你是做 5-fold CV 的某一个 fold
    train_file = os.path.join(config.data_dir, f'train_fold_1.csv')
    val_file = os.path.join(config.data_dir, f'val_fold_1.csv')
    
    train_dataset = load_data_from_csv(train_file, tokenizer, config)
    val_dataset = load_data_from_csv(val_file, tokenizer, config)
    
    train_loader = Data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = Data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    print(f"Vocab Size: {config.vocab_size}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 3. 构建模型
    model = TransPHLA(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 4. 训练
    trainer = Trainer(model, config, optimizer, criterion)
    trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main()

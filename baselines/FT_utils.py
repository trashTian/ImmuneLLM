import sys
import re 
from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import esm

class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)  # 输出到控制台
        self.log.write(message)       # 输出到文件
        self.log.flush()              # 立即写入，避免缓冲

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()




def custom_collate_fn(batch):
    """自定义collate_fn：强制分离(samples列表)和(labels列表)，彻底避免格式混乱"""
    samples = []  # 存储 (name, seq) 元组的列表
    labels = []   # 存储标签的列表
    for item in batch:
        # 每个item是 ((name, seq), label)，手动分离
        sample, label = item
        samples.append(sample)
        labels.append(label)
    return samples, labels

class TCRBindingDataset(Dataset):
    """修复数据集：添加序列校验，确保名称和序列完全分离，支持自定义序列列名"""
    def __init__(self, df, col1, col2, col3, max_length=1024):
        """
        初始化数据集
        Args:
            df (pd.DataFrame): 包含序列和标签的数据框
            col1 (str): 第一个序列列的列名
            col2 (str): 第二个序列列的列名
            max_length (int): 序列最大长度，超过将被截断
        """
        self.df = df
        self.max_length = max_length
        self.aa_pattern = re.compile(r'^[A-Z]+$')  # 标准氨基酸仅含A-Z
        self.col1 = col1
        self.col2 = col2
        self.col3 = col3
        
        # 检查指定的列是否存在
        if col1 not in self.df.columns:
            raise ValueError(f"数据框中不存在列名 '{col1}'")
        if col2 not in self.df.columns:
            raise ValueError(f"数据框中不存在列名 '{col2}'")
        
        # 使用指定的两列组合序列
        self.df = self.df[self.df[col1].notna() & self.df[col2].notna()].copy()
        self.df['combined_seq'] = self.df[col1] + self.df[col2]
        self.df['combined_seq'] = self.df['combined_seq'].apply(self._clean_and_validate_seq)
        
        # 再次过滤校验失败的序列
        self.df = self.df[self.df['combined_seq'].notna()]
        
        # 截断过长序列
        self.df['combined_seq'] = self.df['combined_seq'].apply(lambda x: x[:self.max_length])
        
        print(f"Loaded {len(self.df)} valid samples（已过滤非氨基酸字符）")
    
    def _clean_and_validate_seq(self, seq):
        """清理并校验序列：仅保留A-Z字符，无效序列返回None"""
        if pd.isna(seq):
            return None
        # 移除所有非A-Z的字符（如数字、符号、小写字母）
        cleaned_seq = re.sub(r'[^A-Z]', '', seq.upper())
        # 校验清理后的序列是否非空且仅含A-Z
        if self.aa_pattern.match(cleaned_seq) and len(cleaned_seq) > 0:
            return cleaned_seq
        else:
            print(f"Warning: 无效序列已过滤（含非氨基酸字符）: {seq[:20]}...")
            return None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # 严格返回：(名称, 序列)元组 + 标签（两个独立元素）
        protein_name = f"protein_{idx}"  # 名称仅用于标识，与序列完全分离
        protein_seq = row['combined_seq']
        label = row[self.col3]
        return (protein_name, protein_seq), label  # 格式：((name, seq), label)
    


def extract_esm2_features(seqs, model, alphabet, n_layer, batch_size=1, device=torch.device("cuda:6" if torch.cuda.is_available() else "cpu")):
    """
    Extracts ESM-2 features for a list of sequences from specified model layers.
    Args:
        seqs (list of str): Protein sequences.
        model: ESM-2 model.
        alphabet: ESM-2 alphabet object.
        layers (list of int): Layer indices to extract representations from.
        batch_size (int): Number of sequences per batch.
    Returns:
        torch.Tensor: Representations of shape (num_layers, num_seqs, feature_dim)
    """
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    # Prepare a list for each layer to collect representations
    layers = list(range(n_layer))
    layer_reps = [[] for _ in layers]

    # Process sequences in batches
    for start in range(0, len(seqs), batch_size):
        seq_batch = seqs[start:start + batch_size]
        x_batch = [("protein", seq) for seq in seq_batch]
        _, _, batch_tokens = batch_converter(x_batch)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results = model(batch_tokens.to(device), repr_layers=layers)

        # Collect mean representations for each sequence and layer
        for layer_idx, layer in enumerate(layers):
            token_reps = results["representations"][layer]
            for seq_idx, seq_len in enumerate(batch_lens):
                # Exclude BOS/EOS tokens
                rep = token_reps[seq_idx, 1:seq_len-1].mean(0).cpu()
                layer_reps[layer_idx].append(rep)

    # Stack representations: (num_layers, num_seqs, feature_dim)
    stacked = torch.stack([torch.stack(reps) for reps in layer_reps])
    return stacked


def generate_steering_vector(
        device=torch.device("cuda:6" if torch.cuda.is_available() else "cpu"),
        theshold_pos=1, 
        theshold_neg=0, 
        property='specificity',
        num_data=None, 
        save_folder="/mnt/lustre/guopeijin/data/single_air/steering_vectors/binding_specificity/TCR",
        name_mark='RAKFKQLL',
        data_path="/mnt/lustre/guopeijin/data/single_air/AIR_dataset/AIR-binding-specificity/TCR/B0801_RAKFKQLL_BZLF1_EBV_binder_train.csv"
        ):

    os.makedirs(save_folder, exist_ok=True)
    print(f"Steering vectors will be saved to: {save_folder}")

    if theshold_pos <= theshold_neg:
        raise ValueError("Threshold for positive data must be greater than threshold for negative data.")

    df = pd.read_csv(data_path)
    df['sequence'] = df['b_aaseq'] + df['a_aaseq']
    pos_seqs = df['sequence'][df['labels']>=theshold_pos].to_list()
    neg_seqs = df['sequence'][df['labels']<=theshold_neg].to_list()
    
    if num_data is not None:
        pos_seqs = pos_seqs[:num_data]
        neg_seqs = neg_seqs[:num_data]

    model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t36_3B_UR50D")
    n_layers, _ = (36, 2560)

    pos_seq_repr_mat = extract_esm2_features(pos_seqs, model, alphabet, n_layers, device=device)
    neg_seq_repr_mat = extract_esm2_features(neg_seqs, model, alphabet, n_layers, device=device)

    pos_steering_vectors, neg_steering_vectors = [], []

    for i in range(n_layers):
        pos_steering_vectors.append(pos_seq_repr_mat[i].mean(dim=0))
        neg_steering_vectors.append(neg_seq_repr_mat[i].mean(dim=0))

    pos_steering_vectors = torch.stack(pos_steering_vectors).detach().cpu()
    neg_steering_vectors = torch.stack(neg_steering_vectors).detach().cpu()

    # 从模型名称中提取关键部分用于文件名
    model_name = "3B" 
    sv_path = f"{save_folder}/{model_name}_{property}_{name_mark}_steering_vectors.pt"
    torch.save((pos_steering_vectors, neg_steering_vectors), sv_path)

    # 返回生成的向量路径
    return sv_path


def steering_forward(self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False, steering_vectors=None):            
    if return_contacts:
        need_head_weights = True

    assert tokens.ndim == 2
    padding_mask = tokens.eq(self.padding_idx)  # B, T

    x = self.embed_scale * self.embed_tokens(tokens)

    if padding_mask is not None:
        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

    repr_layers = set(repr_layers)
    hidden_representations = {}
    if 0 in repr_layers:
        hidden_representations[0] = x

    if need_head_weights:
        attn_weights = []

    # (B, T, E) => (T, B, E)
    x = x.transpose(0, 1)

    if not padding_mask.any():
        padding_mask = None

    # target_layers = [25,26,27,28,29, 30, 31, 32, 33, 34, 35]  # 可根据需求修改
        


    for layer_idx, layer in enumerate(self.layers):
        x, attn = layer(
            x,
            self_attn_padding_mask=padding_mask,
            need_head_weights=need_head_weights,
        )
        if steering_vectors is not None:
            add_x = steering_vectors[layer_idx]
            new_x = x + add_x
            new_x_norm = torch.norm(new_x, p=2, dim=-1, keepdim=True).detach()
            x_norm = torch.norm(x, p=2, dim=-1, keepdim=True).detach()
            x = new_x * (x_norm / new_x_norm)

        # if steering_vectors is not None and layer_idx in target_layers:  # 仅对目标层应用
        #     add_x = steering_vectors[layer_idx]
        #     new_x = x + add_x
        #     new_x_norm = torch.norm(new_x, p=2, dim=-1, keepdim=True).detach()
        #     x_norm = torch.norm(x, p=2, dim=-1, keepdim=True).detach()
        #     x = new_x * (x_norm / new_x_norm)

        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x.transpose(0, 1)
        if need_head_weights:
            # (H, B, T, T) => (B, H, T, T)
            attn_weights.append(attn.transpose(1, 0))

    x = self.emb_layer_norm_after(x)
    x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)

    # last hidden representation should have layer norm applied
    if (layer_idx + 1) in repr_layers:
        hidden_representations[layer_idx + 1] = x
    x = self.lm_head(x)

    result = {"logits": x, "representations": hidden_representations}
    if need_head_weights:
        # attentions: B x L x H x T x T
        attentions = torch.stack(attn_weights, 1)
        if padding_mask is not None:
            attention_mask = 1 - padding_mask.type_as(attentions)
            attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
            attentions = attentions * attention_mask[:, None, None, :, :]
        result["attentions"] = attentions
        if return_contacts:
            contacts = self.contact_head(tokens, attentions)
            result["contacts"] = contacts

    return result

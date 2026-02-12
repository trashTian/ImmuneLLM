import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, accuracy_score, matthews_corrcoef, f1_score,
    average_precision_score, confusion_matrix, precision_score, recall_score,
    precision_recall_curve
)
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
import warnings
from modules import QFormer 

warnings.filterwarnings("ignore")

# =========================================================================
# 0. 配置区域
# =========================================================================
# 路径配置 (请根据实际情况修改)
QWEN_PATH = "Qwen3-4B-Instruct-2507"
ESM_PATH = "models--facebook--esm2_t36_3B_UR50D/snapshots/476b639933c8baad5ad09a60ac1a87f987b656fc"

# 评测数据集列表
BENCHMARK_DATASETS = [
    ("TCR_Independent", "unifyimmune_data/data_TCR/independent_set.csv"),
    ("TCR_Triple",      "unifyimmune_data/data_TCR/triple_set.csv"),
    ("TCR_Covid",       "unifyimmune_data/data_TCR/covid_set.csv"),
    ("HLA_External",    "unifyimmune_data/data_HLA/external_set.csv"),
    ("HLA_Independent", "unifyimmune_data/data_HLA/independent_set.csv")
]

PER_DEVICE_BATCH_SIZE = 256
BOOTSTRAP_ROUNDS = 5 

# ================= 参数解析 =================
def parse_args():
    parser = argparse.ArgumentParser(description="Qwen SFT Inference Benchmark")
    
    # 策略参数 (必须与 SFT 训练时完全一致)
    parser.add_argument("--esm_strategy", type=str, required=True, choices=["frozen", "unfreeze_12", "lora", "direct_lora"])
    parser.add_argument("--qwen_strategy", type=str, required=True, choices=["frozen", "lora", "full", "direct_lora"])
    
    # 权重路径 (SFT 训练产出的 pytorch_model.bin)
    parser.add_argument("--checkpoint_path", default="esm_lora_qwen_lora/pytorch_model.bin", type=str, required=True, help="Path to pytorch_model.bin")
    parser.add_argument("--num_queries", type=int, default=64)
    
    return parser.parse_args()

# ================= 模型定义 (ESM + Q-Former + Qwen) =================
class ESMToQwenSFTInferenceModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._keys_to_ignore_on_save = None 

        # 1. 加载基础模型
        print(f"Loading ESM-2 Base...")
        self.esm = AutoModel.from_pretrained(ESM_PATH, dtype=torch.bfloat16)
        self.esm.config.use_cache = False
        print(f"Loading Qwen Base...")
        self.qwen = AutoModelForCausalLM.from_pretrained(
            QWEN_PATH, 
            dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        self.qwen.config.use_cache = False 
        
        print(f"Initializing Q-Former (Q={args.num_queries})...")
        self.projector = QFormer(
            esm_dim=self.esm.config.hidden_size, 
            llm_dim=self.qwen.config.hidden_size, 
            num_queries=args.num_queries, 
            num_hidden_layers=6
        ).to(dtype=torch.bfloat16)

        # 2. 重建训练时的架构 (LoRA)
        self._configure_esm()
        self._configure_qwen()

    def _configure_esm(self):
        strategy = self.args.esm_strategy
        print(f"❄️ ESM Strategy: {strategy}")

        if strategy == "frozen":
            for p in self.esm.parameters(): p.requires_grad = False
            
        elif strategy == "unfreeze_12":
            for p in self.esm.parameters(): p.requires_grad = False
            for layer in self.esm.encoder.layer[-12:]:
                for p in layer.parameters(): p.requires_grad = True
            if hasattr(self.esm.encoder, 'emb_layer_norm_after'):
                for p in self.esm.encoder.emb_layer_norm_after.parameters(): p.requires_grad = True
                
        elif strategy == "lora" or strategy == "direct_lora":
            print("   -> Applying LoRA to ESM (Rank=16, Regex Match)")
            peft_config = LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.1, bias='none',
                target_modules=r".*\.(query|key|value|dense)" 
            )
            self.esm = get_peft_model(self.esm, peft_config)
            
        if hasattr(self.esm, 'pooler') and self.esm.pooler:
            for p in self.esm.pooler.parameters(): p.requires_grad = False
            
    def _configure_qwen(self):
        strategy = self.args.qwen_strategy
        print(f"❄️ Qwen Strategy: {strategy}")
        
        if strategy == "frozen":
            for p in self.qwen.parameters(): p.requires_grad = False
            
        elif strategy == "full":
            print("   -> Unfreezing ALL Qwen parameters")
            for p in self.qwen.parameters(): p.requires_grad = True
            
        elif strategy == "lora" or strategy == "direct_lora":
            print("   -> Applying LoRA to Qwen (Rank=64, Regex Match)")
            # Qwen2 的 Linear 层命名通常与 Llama 一致 (q_proj, k_proj...)
            # 如果是 Qwen1.5 或更早，可能是 c_attn 等，这里的 regex 覆盖了常见情况 
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False,
                r=32, lora_alpha=64, lora_dropout=0.1, bias="none",
                target_modules=r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
            )
            self.qwen = get_peft_model(self.qwen, peft_config)
            self.qwen.print_trainable_parameters()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None: kwargs = {"use_reentrant": False}
        else: kwargs = gradient_checkpointing_kwargs
        kwargs["use_reentrant"] = False
        
        self.qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
        if self.args.qwen_strategy == "lora":
            if hasattr(self.qwen, "enable_input_require_grads"): 
                self.qwen.enable_input_require_grads()
            
        if self.args.esm_strategy != "frozen":
            self.esm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)

    def forward(self, ids_a, ids_mid, ids_b, rec_ids, rec_mask, lig_ids, lig_mask):
        # 1. ESM Encoding
        with torch.no_grad():
            rec_out = self.esm(input_ids=rec_ids, attention_mask=rec_mask).last_hidden_state
            lig_out = self.esm(input_ids=lig_ids, attention_mask=lig_mask).last_hidden_state
        
        # 2. Q-Former Projection
        rec_embeds = self.projector(rec_out, rec_mask)
        lig_embeds = self.projector(lig_out, lig_mask)
        
        # 3. Qwen Embedding
        if hasattr(self.qwen, "get_input_embeddings"): embed = self.qwen.get_input_embeddings()
        else: embed = self.qwen.model.embed_tokens
        
        emb_a = embed(ids_a)
        emb_mid = embed(ids_mid)
        emb_b = embed(ids_b)
        
        # 4. Concat Inputs
        # 结构: [A] + [Rec] + [Mid] + [Lig] + [B]
        inputs_embeds = torch.cat([emb_a, rec_embeds, emb_mid, lig_embeds, emb_b], dim=1)
        
        return self.qwen(inputs_embeds=inputs_embeds)

# ================= Dataset (Qwen ChatML 适配版) =================
class QwenSFTInferenceDataset(Dataset):
    def __init__(self, dataframe, qwen_tokenizer, esm_tokenizer):
        self.df = dataframe.reset_index(drop=True)
        self.qwen_tokenizer = qwen_tokenizer
        self.esm_tokenizer = esm_tokenizer
        
        # 自动列名适配
        if 'HLA' in self.df.columns: self.rec_col = 'HLA'
        elif 'tcr' in self.df.columns: self.rec_col = 'tcr'
        else: self.rec_col = 'receptor_seq' # Fallback

        # Qwen Special Tokens (ChatML)
        self.im_start_id = qwen_tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = qwen_tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.nl_ids = qwen_tokenizer.encode("\n", add_special_tokens=False)
        self.user_ids = qwen_tokenizer.encode("user", add_special_tokens=False)
        self.asst_ids = qwen_tokenizer.encode("assistant", add_special_tokens=False)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        rec_seq = row[self.rec_col]
        if isinstance(rec_seq, list): rec_seq = rec_seq[0]
        lig_seq = row['peptide'] if 'peptide' in row else row['ligand_seq']
        
        # 构造 Prompt (必须与训练时完全一致！)
        text_pre = "Analyze the structural compatibility between "
        text_mid = " and "
        text_post = ".\nPrediction of stable complex formation (Yes/No):"

        ids_pre = self.qwen_tokenizer(text_pre, add_special_tokens=False)['input_ids']
        ids_mid = self.qwen_tokenizer(text_mid, add_special_tokens=False)['input_ids']
        ids_post = self.qwen_tokenizer(text_post, add_special_tokens=False)['input_ids']

        # Part A: <|im_start|>user\n{text_pre}
        ids_a = [self.im_start_id] + self.user_ids + self.nl_ids + ids_pre
        
        # Part Mid: 纯文本
        # ids_mid 保持原样
        
        # Part B: {text_post}<|im_end|>\n<|im_start|>assistant\n
        # 注意：这里不需要加 Answer Token，因为我们要预测它
        ids_b = ids_post + [self.im_end_id] + self.nl_ids + [self.im_start_id] + self.asst_ids + self.nl_ids

        input_ids_a = torch.tensor(ids_a, dtype=torch.long)
        input_ids_mid = torch.tensor(ids_mid, dtype=torch.long)
        input_ids_b = torch.tensor(ids_b, dtype=torch.long)

        # ESM Tokenization
        esm_rec = self.esm_tokenizer(rec_seq, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        esm_lig = self.esm_tokenizer(lig_seq, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

        return {
            "label": int(row['label']),
            "ids_a": input_ids_a, "ids_mid": input_ids_mid, "ids_b": input_ids_b,
            "esm_rec_ids": esm_rec['input_ids'][0], "esm_rec_mask": esm_rec['attention_mask'][0],
            "esm_lig_ids": esm_lig['input_ids'][0], "esm_lig_mask": esm_lig['attention_mask'][0]
        }

def collate_fn(batch, pad_id):
    ids_a = torch.nn.utils.rnn.pad_sequence([x['ids_a'] for x in batch], batch_first=True, padding_value=pad_id)
    ids_mid = torch.nn.utils.rnn.pad_sequence([x['ids_mid'] for x in batch], batch_first=True, padding_value=pad_id)
    ids_b = torch.nn.utils.rnn.pad_sequence([x['ids_b'] for x in batch], batch_first=True, padding_value=pad_id)
    return {
        "labels": torch.tensor([x['label'] for x in batch]),
        "ids_a": ids_a, "ids_mid": ids_mid, "ids_b": ids_b,
        "rec_ids": torch.stack([x['esm_rec_ids'] for x in batch]), "rec_mask": torch.stack([x['esm_rec_mask'] for x in batch]),
        "lig_ids": torch.stack([x['esm_lig_ids'] for x in batch]), "lig_mask": torch.stack([x['esm_lig_mask'] for x in batch])
    }

# ================= 工具函数 =================
def load_sft_weights(model, checkpoint_path):
    if not os.path.exists(checkpoint_path): raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"📥 Overlaying SFT weights from {checkpoint_path} ...")
    
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    print(f"✅ Weights loaded.")
    print(f"   - Missing keys: {len(missing)} (Base params usually)") 
    print(f"   - Unexpected keys: {len(unexpected)}")

def calculate_metrics(np_labels, np_scores):
    try: auc = roc_auc_score(np_labels, np_scores)
    except: auc = 0.5
    pr_auc = average_precision_score(np_labels, np_scores)
    
    precision, recall, thresholds = precision_recall_curve(np_labels, np_scores)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    
    tn, fp, fn, tp = confusion_matrix(np_labels, np_scores).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "AUC": auc, "Accuracy": accuracy_score(np_labels, np_preds), "MCC": matthews_corrcoef(np_labels, np_preds),
        "F1": f1_score(np_labels, np_preds), "PR_AUC": pr_auc, "Specificity": spec,
        "Precision": precision_score(np_labels, np_preds, zero_division=0), "Recall": recall_score(np_labels, np_preds, zero_division=0)
    }

# ================= 主流程 =================
def main():
    args = parse_args()
    accelerator = Accelerator()
    if accelerator.is_main_process: 
        print(f"🚀 Qwen SFT Inference Start")
        print(f"   - ESM Strategy: {args.esm_strategy}")
        print(f"   - Qwen Strategy: {args.qwen_strategy}")

    # 1. Load Tokenizers
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
    if qwen_tokenizer.pad_token is None: qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
    esm_tokenizer = AutoTokenizer.from_pretrained(ESM_PATH)
    
    # 2. Build Model
    model = ESMToQwenSFTInferenceModel(args)
    if accelerator.is_main_process: load_sft_weights(model, args.checkpoint_path)
    model = accelerator.prepare(model)
    model.eval()

    # 3. Define Target Tokens
    # Qwen 的 token ID 可能与 Llama 不同
    yes_tokens = ["Yes", "yes"]
    no_tokens = ["No", "no"]
    yes_ids, no_ids = [], []
    for w in yes_tokens: yes_ids.extend(qwen_tokenizer.encode(w, add_special_tokens=False))
    for w in no_tokens: no_ids.extend(qwen_tokenizer.encode(w, add_special_tokens=False))
    # 去重
    yes_ids = list(set(yes_ids))
    no_ids = list(set(no_ids))
    if accelerator.is_main_process:
        print(f"Target Yes IDs: {yes_ids}")
        print(f"Target No IDs: {no_ids}")

    METRIC_ORDER = ["AUC", "Accuracy", "MCC", "F1", "PR_AUC", "Specificity", "Precision", "Recall"]
    summary_results = []

    # 4. Iterate Datasets
    for ds_name, ds_path in BENCHMARK_DATASETS:
        if accelerator.is_main_process: 
            print("\n" + "-"*40)
            print(f"🧪 Testing: {ds_name}")
        
        if not os.path.exists(ds_path): continue
        df = pd.read_csv(ds_path)
        
        # 4.1 Single Pass Inference
        ds = QwenSFTInferenceDataset(df, qwen_tokenizer, esm_tokenizer)
        dl = DataLoader(ds, batch_size=PER_DEVICE_BATCH_SIZE, shuffle=False, 
                        collate_fn=lambda x: collate_fn(x, qwen_tokenizer.pad_token_id), num_workers=4)
        dl = accelerator.prepare(dl)
        
        all_preds = []
        all_labels = []
        
        for batch in tqdm(dl, disable=not accelerator.is_main_process, leave=False):
            with torch.no_grad():
                out = model(
                    batch['ids_a'], batch['ids_mid'], batch['ids_b'],
                    batch['rec_ids'], batch['rec_mask'], batch['lig_ids'], batch['lig_mask']
                )
                logits = out.logits[:, -1, :] # Last token logits
                
                # Probability Calculation
                s_yes, _ = torch.max(logits[:, yes_ids], dim=1)
                s_no, _ = torch.max(logits[:, no_ids], dim=1)
                probs = torch.softmax(torch.stack([s_no, s_yes], dim=1), dim=-1)[:, 1]
                
                all_preds.extend(accelerator.gather_for_metrics(probs).cpu().tolist())
                all_labels.extend(accelerator.gather_for_metrics(batch['labels']).cpu().tolist())
        
        # 4.2 Efficient Bootstrap in Memory
        if accelerator.is_main_process:
            preds_np = np.array(all_preds)
            labels_np = np.array(all_labels)
            
            ds_metrics = []
            for r in range(BOOTSTRAP_ROUNDS):
                indices = np.random.choice(len(preds_np), size=len(preds_np), replace=True)
                m = calculate_metrics(labels_np[indices], preds_np[indices])
                ds_metrics.append(m)

            df_res = pd.DataFrame(ds_metrics)[METRIC_ORDER]
            mean_vals = df_res.mean()
            std_vals = df_res.std()
            for col in METRIC_ORDER:
                print(f"   {col:<15}: {mean_vals[col]:.4f} ± {std_vals[col]:.4f}")
            
            summary_entry = {"Dataset": ds_name}
            for col in METRIC_ORDER:
                summary_entry[f"{col}_Mean"] = mean_vals[col]
                summary_entry[f"{col}_Std"] = std_vals[col]
            summary_results.append(summary_entry)

    if accelerator.is_main_process:
        print("\n" + "="*80)
        df_summary = pd.DataFrame(summary_results)
        print(df_summary.to_string(index=False))
        # df_summary.to_csv("qwen_benchmark_summary.csv", index=False)

if __name__ == "__main__":
    main()

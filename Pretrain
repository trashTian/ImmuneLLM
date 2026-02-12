import os
# [关键] 禁止 Tokenizer 并行，防止死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
)
from transformers import AutoConfig
from transformers.trainer_utils import get_last_checkpoint
from modules import QFormer

# ==========================================
# 0. 配置区域 [修改]
# ==========================================
# 修改为 Qwen 的路径
LLM_LOCAL_PATH = "Qwen3-4B-Instruct-2507"  # 请替换为你实际的 Qwen 模型路径
ESM2_LOCAL_PATH = "models--facebook--esm2_t36_3B_UR50D/snapshots/476b639933c8baad5ad09a60ac1a87f987b656fc"
DATASET_PATH = "dataset/pretrain/uniprot"

OUTPUT_DIR = "/trained_models/pretrain_EsmQformerQwen_v1"


# ==========================================
# 2. Dataset 和 Collator [重点修改：适配 Qwen ChatML] 
# ==========================================
class ProteinAlignmentDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, llm_tokenizer, esm_tokenizer, max_len_esm=1024, max_len_llm=1024):
        self.dataset = hf_dataset
        self.llm_tokenizer = llm_tokenizer
        self.esm_tokenizer = esm_tokenizer
        self.max_len_esm = max_len_esm
        self.max_len_llm = max_len_llm
        
        # =========================================================
        # [核心修复] 预先获取特殊 Token 的 ID
        # 直接操作 ID 是防止 Tokenizer 错误切分特殊字符串的唯一 100% 安全的方法
        # =========================================================
        self.im_start_id = llm_tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = llm_tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.nl_id = llm_tokenizer.encode("\n", add_special_tokens=False)[0] # 换行符
        
        # 预先编码角色名 (避免在 getitem 里重复计算)
        self.user_role_ids = llm_tokenizer.encode("user", add_special_tokens=False)
        self.asst_role_ids = llm_tokenizer.encode("assistant", add_special_tokens=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item['question']
        answer = item['answer']
        protein_seq = item['proteins'][0] if len(item['proteins']) > 0 else ""

        # 1. 文本切分 (Input Text Splitting)
        # 逻辑：User Prompt 被 <|proteinHere|> 分为前后两部分
        if "<|proteinHere|>" in question:
            parts = question.split("<|proteinHere|>")
            text_pre = parts[0]
            text_post = parts[1]
        else:
            text_pre = question
            text_post = ""

        # 2. 编码文本内容 (Text Tokenization)
        # 注意：add_special_tokens=False，我们要手动控制结构
        ids_pre = self.llm_tokenizer(text_pre, add_special_tokens=False)['input_ids']
        ids_post = self.llm_tokenizer(text_post, add_special_tokens=False)['input_ids']
        ids_answer = self.llm_tokenizer(answer, add_special_tokens=False)['input_ids']

        # =========================================================
        # 3. 构造 Part A (User Header + Pre-text)
        # 格式: <|im_start|>user\n{Pre}
        # =========================================================
        # 拼接: [start] + [user] + [\n] + [pre]
        part_a_ids = [self.im_start_id] + self.user_role_ids + [self.nl_id] + ids_pre
        
        # 转为 Tensor 并截断 (Part A 通常较短，截断风险小，但为了保险)
        if len(part_a_ids) > self.max_len_llm:
             part_a_ids = part_a_ids[-self.max_len_llm:] # 保留后半部分(靠近蛋白的)
             
        input_ids_a = torch.tensor(part_a_ids, dtype=torch.long)

        # =========================================================
        # 4. 构造 Part B (Post-text + End User + Assistant + Answer)
        # 格式: {Post}<|im_end|>\n<|im_start|>assistant\n{Answer}<|im_end|>
        # =========================================================
        
        # --- 前缀部分 (Input, 不计算 Loss) ---
        # 拼接: [post] + [end] + [\n] + [start] + [assistant] + [\n]
        part_b_prefix = (
            ids_post + 
            [self.im_end_id, self.nl_id] + 
            [self.im_start_id] + self.asst_role_ids + [self.nl_id]
        )
        
        # --- 目标部分 (Label, 计算 Loss) ---
        # 拼接: [answer] + [end]
        part_b_target = ids_answer + [self.im_end_id]
        
        # --- 合并 ---
        full_b_ids = part_b_prefix + part_b_target
        
        # --- 截断处理 ---
        # 如果总长度超过限制，从后面截断 (保留 Prompt 和 Answer 的开头)
        if len(full_b_ids) > self.max_len_llm:
            full_b_ids = full_b_ids[:self.max_len_llm]
            
        input_ids_b = torch.tensor(full_b_ids, dtype=torch.long)

        # =========================================================
        # 5. 构建 Labels (Masking)
        # =========================================================
        labels_b = input_ids_b.clone()
        
        # 计算前缀长度 (注意截断可能导致 prefix 变短，取最小值)
        prefix_len = min(len(part_b_prefix), len(labels_b))
        
        # 将前缀部分设为 -100 (Ignored Index)
        labels_b[:prefix_len] = -100 

        # =========================================================
        # 6. ESM 编码
        # =========================================================
        esm_inputs = self.esm_tokenizer(
            protein_seq, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_len_esm, 
            return_tensors='pt'
        )

        return {
            "input_ids_a": input_ids_a,
            "input_ids_b": input_ids_b,
            "labels_b": labels_b,
            "esm_input_ids": esm_inputs['input_ids'][0],
            "esm_attention_mask": esm_inputs['attention_mask'][0]
        }
    
    
class ProteinDataCollator:
    def __init__(self, llm_pad_token_id):
        self.pad_id = llm_pad_token_id

    def __call__(self, batch):
        input_ids_a = [x['input_ids_a'] for x in batch]
        input_ids_b = [x['input_ids_b'] for x in batch]
        labels_b = [x['labels_b'] for x in batch]
        esm_ids = [x['esm_input_ids'] for x in batch]
        esm_mask = [x['esm_attention_mask'] for x in batch]

        input_ids_a_padded = torch.nn.utils.rnn.pad_sequence(input_ids_a, batch_first=True, padding_value=self.pad_id)
        input_ids_b_padded = torch.nn.utils.rnn.pad_sequence(input_ids_b, batch_first=True, padding_value=self.pad_id)
        labels_b_padded = torch.nn.utils.rnn.pad_sequence(labels_b, batch_first=True, padding_value=-100)
        
        esm_ids_stacked = torch.stack(esm_ids)
        esm_mask_stacked = torch.stack(esm_mask)

        attn_mask_a = (input_ids_a_padded != self.pad_id).long()
        attn_mask_b = (input_ids_b_padded != self.pad_id).long()

        return {
            "input_ids_a": input_ids_a_padded,
            "attention_mask_a": attn_mask_a,
            "input_ids_b": input_ids_b_padded,
            "attention_mask_b": attn_mask_b,
            "labels_b": labels_b_padded,
            "esm_input_ids": esm_ids_stacked,
            "esm_attention_mask": esm_mask_stacked
        }

# ==========================================
# 3. 模型类 [修改：适配 Qwen 架构]
# ==========================================
class ESMToQwenModel(nn.Module):
    def __init__(self, esm_path, llm_path, freeze_esm=True, freeze_llm=True):
        super().__init__()
        
        print("Loading ESM-2...")
        self.esm = AutoModel.from_pretrained(esm_path, dtype=torch.bfloat16)
        if freeze_esm:
            self.esm.eval()
            for param in self.esm.parameters():
                param.requires_grad = False
        
        print(f"Loading LLM (Qwen) from {llm_path}...")

        
        self.llm = AutoModelForCausalLM.from_pretrained(
                llm_path, 
                dtype=torch.bfloat16,
                # trust_remote_code=True,  # 必须加，以防它包含自定义代码
                # low_cpu_mem_usage=True   # 防止 8 卡加载爆内存
            )
        
        self.llm.config.use_cache = False 

        if freeze_llm:
            self.llm.eval()
            for param in self.llm.parameters():
                param.requires_grad = False

        esm_dim = self.esm.config.hidden_size
        llm_dim = self.llm.config.hidden_size
        
        print(f"Initializing Q-Former (ESM dim: {esm_dim} -> LLM dim: {llm_dim})...")
        self.projector = QFormer(
            esm_dim=esm_dim, 
            llm_dim=llm_dim, 
            num_queries=64, 
            num_hidden_layers=6
        ).to(dtype=torch.bfloat16)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def forward(self, input_ids_a, attention_mask_a, input_ids_b, attention_mask_b, labels_b, esm_input_ids, esm_attention_mask):
        # 1. ESM
        with torch.no_grad():
            esm_out = self.esm(input_ids=esm_input_ids, attention_mask=esm_attention_mask)
            protein_embeds = esm_out.last_hidden_state
        
        # 2. Q-Former
        projected_protein_embeds = self.projector(protein_embeds, esm_attention_mask)

        # 3. LLM Embedding 拼接
        # [注意] Qwen2 的 embedding 层通常也在 model.embed_tokens
        # 使用 get_input_embeddings() 是最通用的方法，兼容 Llama 和 Qwen
        embed_tokens = self.llm.get_input_embeddings()
        
        embeds_a = embed_tokens(input_ids_a)
        embeds_b = embed_tokens(input_ids_b)

        inputs_embeds = torch.cat([embeds_a, projected_protein_embeds, embeds_b], dim=1)

        # 4. Mask 拼接
        batch_size = input_ids_a.shape[0]
        num_queries = self.projector.num_queries
        prot_mask = torch.ones((batch_size, num_queries), device=input_ids_a.device, dtype=attention_mask_a.dtype)
        final_attention_mask = torch.cat([attention_mask_a, prot_mask, attention_mask_b], dim=1)

        # 5. Label 拼接
        labels_a = torch.full((batch_size, input_ids_a.shape[1]), -100, device=input_ids_a.device)
        labels_prot = torch.full((batch_size, num_queries), -100, device=input_ids_a.device)
        final_labels = torch.cat([labels_a, labels_prot, labels_b], dim=1)

        outputs = self.llm(
            inputs_embeds=inputs_embeds, 
            attention_mask=final_attention_mask, 
            labels=final_labels
        )
        return outputs

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# ==========================================
# 4. 主函数
# ==========================================
def main():
    print("Loading Tokenizers...")
    # [修改] Qwen Tokenizer
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_LOCAL_PATH, trust_remote_code=True)
    # Qwen 通常没有默认的 pad_token，手动指定为 eos_token
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
        print(f"Set pad_token to eos_token: {llm_tokenizer.pad_token}")
        
    esm_tokenizer = AutoTokenizer.from_pretrained(ESM2_LOCAL_PATH)

    print("Loading Dataset from Disk...")
    dataset_dict = load_from_disk(DATASET_PATH)
    
    if "train" in dataset_dict:
        train_raw = dataset_dict["train"].shuffle(seed=42)
    else:
        raise ValueError("Dataset missing 'train' split.")
        
    if "test" in dataset_dict:
        eval_raw = dataset_dict["test"]
    elif "validation" in dataset_dict:
        eval_raw = dataset_dict["validation"]
    else:
        eval_raw = None

    if eval_raw is not None:
        eval_raw = eval_raw.select(range(min(2000, len(eval_raw)))) 

    print("Processing Datasets...")
    train_dataset = ProteinAlignmentDataset(train_raw, llm_tokenizer, esm_tokenizer)
    eval_dataset = ProteinAlignmentDataset(eval_raw, llm_tokenizer, esm_tokenizer) if eval_raw else None

    data_collator = ProteinDataCollator(llm_pad_token_id=llm_tokenizer.pad_token_id)

    print("Initializing Model...")
    model = ESMToQwenModel(
        esm_path=ESM2_LOCAL_PATH, 
        llm_path=LLM_LOCAL_PATH,
        freeze_esm=True,    
        freeze_llm=True 
    )
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        
        # 1. 显存优化与吞吐量
        per_device_train_batch_size=16,  # 尝试调大 Batch Size (如果 OOM 改回 12)
        gradient_accumulation_steps=8,   # 16 * 4 * 8 GPUs = 512 Total Batch Size (更稳定)
        
        # 2. 学习率调度 (Standard Pretraining Practices)
        learning_rate=4e-4,              # Q-Former 初始训练常用 1e-4
        lr_scheduler_type="cosine",      # [关键] 使用余弦退火，收敛效果通常优于 linear
        warmup_ratio=0.03,               # [关键] 3% 预热，防止初始化 Q-Former 梯度爆炸
        weight_decay=0.05,               # 加入正则化防止过拟合
        
        # 3. 检查点与恢复
        resume_from_checkpoint=True,
        save_total_limit=3,
        
        # 4. 评估策略
        eval_strategy="steps" if eval_dataset else "no", 
        eval_steps=200,                
        save_steps=200,                
        per_device_eval_batch_size=16, 
        load_best_model_at_end=True if eval_dataset else False, 
        metric_for_best_model="eval_loss",

        logging_steps=10,
        num_train_epochs=3,
        bf16=True,                      
        gradient_checkpointing=True,    
        save_safetensors=False,
        remove_unused_columns=False,
        report_to="none",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=16
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, 
        data_collator=data_collator,
    )

    print("Checking for existing checkpoints...")
    # [修复] 先判断目录是否存在，避免 FileNotFoundError
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    else:
        last_checkpoint = None
    
    if last_checkpoint is not None:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("No checkpoint found. Starting fresh training...")
        trainer.train()

    if trainer.args.process_index == 0:
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.projector.state_dict(), os.path.join(OUTPUT_DIR, "stage1_only_pf_qformer_qwen.pt"))
        print("Model saved successfully.")

if __name__ == "__main__":
    main()

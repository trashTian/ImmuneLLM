import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW
from modules import QFormer

# ==========================================
# 1. 参数解析
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 Mix SFT Trainer (Qwen Backbone)")

    # 策略配置 (必须与上一阶段 Checkpoint 保持一致)
    parser.add_argument("--esm_strategy", type=str, default="lora", 
                        choices=["frozen", "unfreeze_12", "lora", "direct_lora"])
    parser.add_argument("--qwen_strategy", type=str, default="lora",
                        choices=["frozen", "lora", "full", "direct_lora"])
    
    # 路径配置
    parser.add_argument("--qwen_path", type=str, default="/mnt/lustre/guopeijin/model/models/LLM-Research/Qwen3-4B-Instruct-2507")
    parser.add_argument("--esm_path", type=str, default="/mnt/lustre/guopeijin/model/hub/models--facebook--esm2_t36_3B_UR50D/snapshots/476b639933c8baad5ad09a60ac1a87f987b656fc")
    
    # [关键] 这里指向 Raw SFT 训练好的 checkpoint (pytorch_model.bin)
    parser.add_argument("--pretrained_ckpt", type=str, 
                        default="/mnt/lustre/guopeijin/Immune_LLM/code/trained_models/sft_qwen/esm_lora_qwen_lora/pytorch_model.bin",
                        help="Path to the Stage 2 (Raw SFT) checkpoint")
    
    # 数据集路径 (Mix Data)
    parser.add_argument("--data_path", type=str, default="/mnt/lustre/guopeijin/Immune_LLM/code/data_prepare/stage2_sft_mixed_answer_only")
    parser.add_argument("--output_base_dir", type=str, default="/mnt/lustre/guopeijin/Immune_LLM/code/trained_models/sft_qwen_mixed_v4")
    
    # 超参 (Mix SFT 通常 LR 较小)
    parser.add_argument("--lr", type=float, default=2e-5) 
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num_queries", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=2048)
    
    return parser.parse_args()

# ==========================================
# 2. 模型定义 (ESM + Q-Former + Qwen)
# ==========================================
class ESMToQwenModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._keys_to_ignore_on_save = None 
        
        # 1. 加载基础模型
        print(f"Loading ESM-2 from: {args.esm_path}")
        self.esm = AutoModel.from_pretrained(args.esm_path, dtype=torch.bfloat16)
        self.esm.config.use_cache = False 
        
        print(f"Loading Qwen from: {args.qwen_path}")
        # Qwen 通常建议开启 trust_remote_code
        self.qwen = AutoModelForCausalLM.from_pretrained(
            args.qwen_path, 
            dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        self.qwen.config.use_cache = False 
        
        print(f"Initializing Q-Former (Queries={args.num_queries})...")
        # ⚠️ 注意：self.qwen.config.hidden_size 必须与 Stage 1 训练时一致
        self.projector = QFormer(
            esm_dim=self.esm.config.hidden_size, 
            llm_dim=self.qwen.config.hidden_size, 
            num_queries=args.num_queries, 
            num_hidden_layers=6
        ).to(dtype=torch.bfloat16)
        
         # =================================================================
        # 🚨 步骤 2：必须先应用 LoRA！(把这两行代码移到加载权重之前)
        # =================================================================
        self._configure_esm()   # <--- 先建房
        self._configure_qwen()  # <--- 先建房
        
        # =================================================================
        # 🚨 步骤 3：架构准备好后，再搬家具 (加载权重)
        # =================================================================
        if args.pretrained_ckpt and os.path.exists(args.pretrained_ckpt):
            print(f"📥 Overlaying SFT weights from {args.pretrained_ckpt} ...")
            state_dict = torch.load(args.pretrained_ckpt, map_location="cpu")
            
            # 此时模型里已经有 lora_A/B 层了，所以 state_dict 里的 lora 参数会被正确吸入
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            
            print(f"✅ Weights loaded.")
            print(f"   - Missing keys: {len(missing)} (Should be mostly Base Params)") 
            
            # 【关键指标】: 这里必须接近 0！
            # 如果这里还是 1956，那就还是错的。如果是 0，就对了！
            print(f"   - Unexpected keys: {len(unexpected)}") 
            
            # 双重保险检查
            if len(unexpected) > 100:
                print(f"❌ DANGER: Still dumping too many keys! Sample: {unexpected[:3]}")
                # 这种情况下通常需要抛出异常停止
                raise ValueError("LoRA weights were not loaded correctly (Unexpected keys count too high).")
        else:
            raise ValueError(f"Checkpoint file not found at: {args.pretrained_ckpt}")
        
        self._set_gradients()
    
    def _set_gradients(self):
        # Q-Former 始终训练
        for p in self.projector.parameters(): p.requires_grad = True
        
        # 打印可训练参数量
        trainable_params = 0
        all_params = 0
        for p in self.parameters():
            all_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()
        print(f"📊 Trainable params: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")


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
                r=64, lora_alpha=128, lora_dropout=0.1, bias="none",
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

    def forward(self, input_ids_a, attention_mask_a, input_ids_mid, attention_mask_mid, input_ids_b, attention_mask_b, labels_b, esm_rec_ids, esm_rec_mask, esm_lig_ids, esm_lig_mask):
        # 1. ESM Encoding
        with torch.set_grad_enabled(self.args.esm_strategy != "frozen"):
            rec_feats = self.esm(input_ids=esm_rec_ids, attention_mask=esm_rec_mask).last_hidden_state
            lig_feats = self.esm(input_ids=esm_lig_ids, attention_mask=esm_lig_mask).last_hidden_state
        
        rec_embeds = self.projector(rec_feats, esm_rec_mask) 
        lig_embeds = self.projector(lig_feats, esm_lig_mask) 

        # 2. Get Qwen Embeddings
        # Qwen 通常可以通过 get_input_embeddings 获取
        if hasattr(self.qwen, "get_input_embeddings"): embed = self.qwen.get_input_embeddings()
        else: embed = self.qwen.model.embed_tokens
        
        emb_a = embed(input_ids_a)
        emb_mid = embed(input_ids_mid)
        emb_b = embed(input_ids_b)

        # 3. Concat Inputs
        inputs_embeds = torch.cat([emb_a, rec_embeds, emb_mid, lig_embeds, emb_b], dim=1)

        # 4. Construct Masks & Position IDs
        B, _ = input_ids_a.shape
        Q = self.projector.num_queries
        p_mask = torch.ones((B, Q), device=input_ids_a.device, dtype=attention_mask_a.dtype)
        final_mask = torch.cat([attention_mask_a, p_mask, attention_mask_mid, p_mask, attention_mask_b], dim=1)
        
        long_mask = final_mask.long()
        position_ids = long_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(long_mask == 0, 0)

        # 5. Construct Labels (Masking everything except part B output)
        l_a = torch.full((B, input_ids_a.shape[1]), -100, device=input_ids_a.device)
        l_mid = torch.full((B, input_ids_mid.shape[1]), -100, device=input_ids_a.device)
        l_p = torch.full((B, Q), -100, device=input_ids_a.device)
        final_labels = torch.cat([l_a, l_p, l_mid, l_p, labels_b], dim=1)

        return self.qwen(
            inputs_embeds=inputs_embeds, 
            attention_mask=final_mask, 
            position_ids=position_ids,
            labels=final_labels
        )

# ==========================================
# 3. 数据集与 Collator (Qwen ChatML Logic)
# ==========================================
class QwenSFTDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, qwen_tokenizer, esm_tokenizer, max_length=2048):
        self.dataset = hf_dataset
        self.qwen_tokenizer = qwen_tokenizer
        self.esm_tokenizer = esm_tokenizer
        self.max_length = max_length
        
        # Qwen ChatML Special Tokens
        self.im_start_id = qwen_tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = qwen_tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.nl_ids = qwen_tokenizer.encode("\n", add_special_tokens=False)
        self.user_ids = qwen_tokenizer.encode("user", add_special_tokens=False)
        self.asst_ids = qwen_tokenizer.encode("assistant", add_special_tokens=False)
        
        self.token_70 = "<|reserved_special_token_70|>"
        self.token_71 = "<|reserved_special_token_71|>"

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        rec_seq = item['receptor_seq']
        lig_seq = item['ligand_seq']
        user_input = item['input']
        assistant_output = item['output']

        ids_ans = self.qwen_tokenizer(assistant_output, add_special_tokens=False)['input_ids']

        # 分支逻辑：Raw SFT vs CoT
        if self.token_70 in user_input and self.token_71 in user_input:
            # Case A: Raw SFT (填空)
            try:
                part1, temp = user_input.split(self.token_70)
                part2, part3 = temp.split(self.token_71)
            except ValueError:
                part1, part2, part3 = "Analyze compatibility: ", " and ", "."
            
            ids_p1 = self.qwen_tokenizer(part1, add_special_tokens=False)['input_ids']
            ids_p2 = self.qwen_tokenizer(part2, add_special_tokens=False)['input_ids']
            ids_p3 = self.qwen_tokenizer(part3, add_special_tokens=False)['input_ids']
            
            # 构造 ChatML 格式:
            # Part A: <|im_start|>user\n{p1}
            ids_a_list = [self.im_start_id] + self.user_ids + self.nl_ids + ids_p1
            # Part Mid: {p2}
            ids_mid_list = ids_p2
            # Part B: {p3}<|im_end|>\n<|im_start|>assistant\n{Answer}<|im_end|>\n
            ids_b_list = (
                ids_p3 + 
                [self.im_end_id] + self.nl_ids + 
                [self.im_start_id] + self.asst_ids + self.nl_ids + 
                ids_ans + 
                [self.im_end_id] + self.nl_ids
            )
            
            input_ids_a = torch.tensor(ids_a_list, dtype=torch.long)
            input_ids_mid = torch.tensor(ids_mid_list, dtype=torch.long)
            input_ids_b = torch.tensor(ids_b_list, dtype=torch.long)
            
            labels_b = input_ids_b.clone()
            # Mask prefix
            prefix_len = len(ids_p3) + 1 + len(self.nl_ids) + 1 + len(self.asst_ids) + len(self.nl_ids)
            labels_b[:prefix_len] = -100

        else:
            # Case B: CoT (前置注入)
            # 需手动构造 "Here is..." 文本以包裹 Protein Embedding
            prefix_text = "Here is the receptor sequence representation: "
            mid_text = " and the ligand sequence representation: "
            suffix_text = ".\n\n"
            
            ids_pre = self.qwen_tokenizer(prefix_text, add_special_tokens=False)['input_ids']
            ids_mid_text = self.qwen_tokenizer(mid_text, add_special_tokens=False)['input_ids']
            ids_suf = self.qwen_tokenizer(suffix_text, add_special_tokens=False)['input_ids']
            ids_user_content = self.qwen_tokenizer(user_input, add_special_tokens=False)['input_ids']
            
            # Part A: <|im_start|>user\n{prefix}
            ids_a_list = [self.im_start_id] + self.user_ids + self.nl_ids + ids_pre
            # Part Mid: {mid}
            ids_mid_list = ids_mid_text
            # Part B: {suffix}{UserContent}<|im_end|>\n<|im_start|>assistant\n{Answer}<|im_end|>\n
            ids_b_list = (
                ids_suf + ids_user_content + 
                [self.im_end_id] + self.nl_ids + 
                [self.im_start_id] + self.asst_ids + self.nl_ids + 
                ids_ans + 
                [self.im_end_id] + self.nl_ids
            )
            
            # 简单的长度截断保护
            overhead = len(ids_a_list) + 128 + len(ids_mid_list) + len(ids_b_list)
            if overhead > self.max_length:
                 ids_user_content = ids_user_content[:-(overhead - self.max_length)]
                 # Rebuild B
                 ids_b_list = (
                    ids_suf + ids_user_content + 
                    [self.im_end_id] + self.nl_ids + 
                    [self.im_start_id] + self.asst_ids + self.nl_ids + 
                    ids_ans + 
                    [self.im_end_id] + self.nl_ids
                )
            
            input_ids_a = torch.tensor(ids_a_list, dtype=torch.long)
            input_ids_mid = torch.tensor(ids_mid_list, dtype=torch.long)
            input_ids_b = torch.tensor(ids_b_list, dtype=torch.long)
            
            labels_b = input_ids_b.clone()
            # Mask prefix
            mask_len = len(ids_suf) + len(ids_user_content) + 1 + len(self.nl_ids) + 1 + len(self.asst_ids) + len(self.nl_ids)
            labels_b[:mask_len] = -100

        esm_rec = self.esm_tokenizer(rec_seq, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        esm_lig = self.esm_tokenizer(lig_seq, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

        return {
            "input_ids_a": input_ids_a, "input_ids_mid": input_ids_mid, "input_ids_b": input_ids_b, 
            "labels_b": labels_b,
            "esm_rec_ids": esm_rec['input_ids'][0], "esm_rec_mask": esm_rec['attention_mask'][0],
            "esm_lig_ids": esm_lig['input_ids'][0], "esm_lig_mask": esm_lig['attention_mask'][0],
            "input_ids": input_ids_b 
        }

class QwenProteinCollator:
    def __init__(self, pad_id): self.pad_id = pad_id
    def __call__(self, batch):
        ids_a = torch.nn.utils.rnn.pad_sequence([x['input_ids_a'] for x in batch], batch_first=True, padding_value=self.pad_id)
        ids_mid = torch.nn.utils.rnn.pad_sequence([x['input_ids_mid'] for x in batch], batch_first=True, padding_value=self.pad_id)
        ids_b = torch.nn.utils.rnn.pad_sequence([x['input_ids_b'] for x in batch], batch_first=True, padding_value=self.pad_id)
        labels_b = torch.nn.utils.rnn.pad_sequence([x['labels_b'] for x in batch], batch_first=True, padding_value=-100)
        
        return {
            "input_ids_a": ids_a, "attention_mask_a": (ids_a != self.pad_id).long(),
            "input_ids_mid": ids_mid, "attention_mask_mid": (ids_mid != self.pad_id).long(),
            "input_ids_b": ids_b, "attention_mask_b": (ids_b != self.pad_id).long(), "labels_b": labels_b,
            "esm_rec_ids": torch.stack([x['esm_rec_ids'] for x in batch]), "esm_rec_mask": torch.stack([x['esm_rec_mask'] for x in batch]),
            "esm_lig_ids": torch.stack([x['esm_lig_ids'] for x in batch]), "esm_lig_mask": torch.stack([x['esm_lig_mask'] for x in batch])
        }

def get_train_config(args, model):
    params = []
    
    # 1. Q-Former
    qformer_lr = 5e-5
    params.append({
        "params": [p for p in model.projector.parameters() if p.requires_grad], 
        "lr": qformer_lr
    })
    
    # 2. ESM
    esm_params = [p for p in model.esm.parameters() if p.requires_grad]
    if esm_params:
        lr = 1e-4 if args.esm_strategy == "lora" else 1e-5
        params.append({"params": esm_params, "lr": lr})
        
    # 3. Qwen
    qwen_params = [p for p in model.qwen.parameters() if p.requires_grad]
    if qwen_params:
        lr = 2e-4 if args.qwen_strategy == "lora" else 5e-6
        params.append({"params": qwen_params, "lr": lr})

    return params

# ==========================================
# 4. 主程序
# ==========================================
def main():
    args = parse_args()
    
    run_name = f"stage2_sft_{args.esm_strategy}_{args.qwen_strategy}_mixed"
    output_dir = f"{args.output_base_dir}/{run_name}"
    
    # Tokenizer
    qwen_tokenizer = AutoTokenizer.from_pretrained(args.qwen_path, trust_remote_code=True)
    if qwen_tokenizer.pad_token is None: qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
    esm_tokenizer = AutoTokenizer.from_pretrained(args.esm_path)
    
    print(f"Loading Mixed Dataset: {args.data_path}")
    ds_dict = load_from_disk(args.data_path)
    train_ds = QwenSFTDataset(ds_dict['train'], qwen_tokenizer, esm_tokenizer, max_length=args.max_length)
    if 'validation' in ds_dict:
        eval_ds = QwenSFTDataset(ds_dict['validation'], qwen_tokenizer, esm_tokenizer, max_length=args.max_length)
    else:
        eval_ds = None

    model = ESMToQwenModel(args)
    optimizer = AdamW(get_train_config(args, model), weight_decay=0.01)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, 
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        num_train_epochs=args.epochs,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=10,
        load_best_model_at_end=True if eval_ds else False,
        metric_for_best_model="eval_loss",
        
        remove_unused_columns=False,
        report_to="none",
        ddp_find_unused_parameters=False,
        save_safetensors=False ,
        dataloader_num_workers=16,
        per_device_eval_batch_size=args.batch_size,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=QwenProteinCollator(qwen_tokenizer.pad_token_id),
        optimizers=(optimizer, None)
    )
    
    # 自动处理 Checkpoint (优先加载本地 output_dir 的断点，没有则开始训练)
    last_checkpoint = None
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
    
    if last_checkpoint:
        print(f"🔄 Found checkpoint: {last_checkpoint}. Resuming training...")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("🆕 No valid checkpoint found. Starting fresh training...")
        trainer.train()
    
    trainer.save_model()
    print(f"✅ Stage 2 Training Finished. Saved to {output_dir}")

if __name__ == "__main__":
    main()

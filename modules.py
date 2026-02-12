import os
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
from transformers.models.bert.modeling_bert import BertConfig, BertLayer
from peft import LoraConfig, get_peft_model, TaskType

# ==========================================
# 1. Q-Former (标准版 - 保持不变)
# ==========================================
class QFormer(nn.Module):
    def __init__(self, esm_dim, llm_dim, num_queries=64, num_hidden_layers=6, hidden_size=768):
        super().__init__()
        self.num_queries = num_queries
        self.esm_proj = nn.Linear(esm_dim, hidden_size)
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_size))
        nn.init.normal_(self.query_tokens, std=0.02)

        config = BertConfig(
            hidden_size=hidden_size,
            num_attention_heads=12,
            intermediate_size=hidden_size * 4,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            is_decoder=True,
            add_cross_attention=True,
            num_hidden_layers=num_hidden_layers,
            _attn_implementation="eager"
        )
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(num_hidden_layers)])
        self.llm_proj = nn.Linear(hidden_size, llm_dim) # 输出映射到 Qwen 的维度

    def forward(self, esm_feats, esm_mask):
        batch_size = esm_feats.shape[0]
        encoder_hidden_states = self.esm_proj(esm_feats)
        
        extended_mask = esm_mask[:, None, None, :]
        extended_mask = (1.0 - extended_mask) * -10000.0
        extended_mask = extended_mask.to(dtype=esm_feats.dtype) 
        
        hidden_states = self.query_tokens.expand(batch_size, -1, -1)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=extended_mask
            )[0]
        return self.llm_proj(hidden_states)

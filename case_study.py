import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
import warnings
from modules import QFormer 

warnings.filterwarnings("ignore")

# ================= 配置区域 =================
QWEN_PATH = "Qwen3-4B-Instruct-2507" # 修改为你的 Qwen 路径
ESM_PATH = "models--facebook--esm2_t36_3B_UR50D/snapshots/476b639933c8baad5ad09a60ac1a87f987b656fc"

BENCHMARK_DATASETS = [
    ("case_study_more_mutate","case_study_more_mutate.csv")
]

PER_DEVICE_BATCH_SIZE = 16 
MAX_NEW_TOKENS = 2048 
BOOTSTRAP_ROUNDS = 1

#  ==============================================================================
# 1. 严格对齐的 HLA Validator (Copy from generate_prompt_2_llm_hla_i.txt)
# ==============================================================================
FINAL_HLA_RULES_JSON = """
[
  {
    "rule_id": "HLA_I_A02_HYDROPHOBIC_ANCHOR",
    "description": "A02 Supertype: Index 3 is M/Q. Creates a hydrophobic B-pocket favoring L/M/I/V at P2.",
    "python_condition": {
      "hla_indices": [3], "hla_values": ["M", "Q"], "peptide_index": 1,
      "peptide_allowed": ["L", "M", "I", "V", "Q", "A", "T"], "peptide_forbidden": ["D", "E", "R", "K", "P"]
    }
  },
  {
    "rule_id": "HLA_I_B27_SALT_BRIDGE_FINAL",
    "description": "B27/B14 Supertype: Index 3 is E. Strongly prefers Basic P2 (R/K). Strictly forbids Acidic.",
    "python_condition": {
      "hla_indices": [3], "hla_values": ["E"], "peptide_index": 1,
      "peptide_allowed": ["R", "K", "H", "Q", "L", "M", "V", "I"], "peptide_forbidden": ["D", "E", "P"] 
    }
  },
  {
    "rule_id": "HLA_I_B44_ACIDIC_P2_GOLD",
    "description": "B44 Supertype: Index 3 is K. Strictly requires Acidic P2 (E/D).",
    "python_condition": {
      "hla_indices": [3], "hla_values": ["K"], "peptide_index": 1,
      "peptide_allowed": ["E", "D"], "peptide_forbidden": ["R", "K", "H", "L", "M", "F", "Y", "W"]
    }
  },
  {
    "rule_id": "HLA_I_A03_A11_BASIC_CTERM",
    "description": "A03/A11: Index 22 is D. Prefers Basic C-term, forbids Acidic C-term.",
    "python_condition": {
      "hla_indices": [22], "hla_values": ["D"], "peptide_index": -1,
      "peptide_allowed": ["K", "R", "H", "Y", "F", "L"], "peptide_forbidden": ["D", "E"]
    }
  },
  {
    "rule_id": "HLA_I_A01_TYROSINE_CTERM",
    "description": "A01: Index 22 is Y. Prefers Aromatic/Hydrophobic C-term.",
    "python_condition": {
      "hla_indices": [22], "hla_values": ["Y"], "peptide_index": -1,
      "peptide_allowed": ["Y", "F", "L", "I", "W"], "peptide_forbidden": ["D", "E", "P", "R", "K"]
    }
  },
  {
    "rule_id": "HLA_I_C_HYDROPHOBIC_PREF",
    "description": "HLA-C: Index 22 is F or Y. Prefers Hydrophobic C-term.",
    "python_condition": {
      "hla_indices": [22], "hla_values": ["F", "Y"], "peptide_index": -1,
      "peptide_allowed": ["L", "I", "V", "M", "F", "A"], "peptide_forbidden": ["D", "E", "R", "K"]
    }
  },
  {
    "rule_id": "HLA_I_A26_TYROSINE_CTERM",
    "description": "A26: Index 22 is H. Prefers Aromatic C-term.",
    "python_condition": {
      "hla_indices": [22], "hla_values": ["H"], "peptide_index": -1,
      "peptide_allowed": ["Y", "F", "L", "V"], "peptide_forbidden": ["D", "E"]
    }
  },
  {
    "rule_id": "HLA_I_STERIC_F_GLY",
    "description": "Physics: Bulky F-pocket (F/Y/W at Index 22) discourages Glycine (G) due to entropy.",
    "python_condition": {
      "hla_indices": [22], "hla_values": ["F", "Y", "W"], "peptide_index": -1,
      "peptide_allowed": ["L", "I", "V", "F", "Y"], "peptide_forbidden": ["G", "P"]
    }
  },
  {
    "rule_id": "HLA_I_GENERAL_CHARGE_SAFETY",
    "description": "Physics: Double Charge Repulsion. If both anchors are D/E, binding fails.",
    "python_condition": {
      "hla_indices": [3, 22], "hla_values": [["L", "I", "V", "M", "Y", "F", "W"], ["L", "I", "V", "M", "Y", "F", "W"]], "peptide_index": "BOTH_ANCHORS",
      "peptide_allowed": [], "peptide_forbidden": ["D", "E"]
    }
  },
  {
    "rule_id": "HLA_I_A24_AROMATIC_PRECISION",
    "description": "A24 Supertype: Index 3 is Y/F/M AND Index 20 is F/Y. Prefers Aromatic (Y/F/W) or Leucine at P2.",
    "python_condition": {
      "hla_indices": [3, 20], "hla_values": [["Y", "F", "M"], ["F", "Y"]], 
      "peptide_index": 1,
      "peptide_allowed": ["Y", "F", "W", "L", "M", "I"], "peptide_forbidden": ["D", "E", "R", "K", "P", "G"]
    }
  },
  {
    "rule_id": "HLA_I_B07_PROLINE_PRECISION",
    "description": "B07/B35 Supertype: Index 6 is N/Y/F. Strongly favors Proline (P) or small Hydrophobic (A/V) at P2.",
    "python_condition": {
      "hla_indices": [6], "hla_values": ["N", "Y", "F"], "peptide_index": 1,
      "peptide_allowed": ["P", "A", "V"], "peptide_forbidden": ["W", "F", "Y", "D", "E", "R", "K"]
    }
  }
]
"""

class HLARuleValidator:
    def __init__(self):
        self.rules = json.loads(FINAL_HLA_RULES_JSON)
        self.rule_desc_map = {r['rule_id']: r['description'] for r in self.rules}

    def check_row(self, hla_seq, pep_seq):
        results = {}
        if not isinstance(pep_seq, str) or len(pep_seq) < 2: return {}
        if not isinstance(hla_seq, str) or len(hla_seq) < 34: return {}

        for rule in self.rules:
            rid = rule['rule_id']
            cond = rule['python_condition']
            
            hla_triggered = True
            h_indices = cond.get('hla_indices', [])
            h_values = cond.get('hla_values', [])

            if len(h_indices) > 0:
                for i, h_idx in enumerate(h_indices):
                    if h_idx >= len(hla_seq):
                        hla_triggered = False; break
                    target_val_container = h_values[i] if i < len(h_values) else []
                    if isinstance(target_val_container, str): target_val_container = [target_val_container]
                    if hla_seq[h_idx] not in target_val_container:
                        hla_triggered = False; break
            
            if not hla_triggered: continue

            pep_status = "NEUTRAL"
            p_indices = cond['peptide_index']
            if isinstance(p_indices, int): p_indices = [p_indices]
            elif p_indices == "BOTH_ANCHORS": p_indices = [1, -1]
            
            allowed = cond.get('peptide_allowed', [])
            forbidden = cond.get('peptide_forbidden', [])
            
            has_forbidden = False
            all_allowed = True
            
            for p_idx in p_indices:
                try: res = pep_seq[p_idx]
                except IndexError: has_forbidden = True; break
                if res in forbidden: has_forbidden = True; break
                if allowed and (res not in allowed): all_allowed = False
            
            if has_forbidden: pep_status = "FAIL"
            elif allowed and all_allowed: pep_status = "PASS"
            elif allowed and not all_allowed:
                if "GENERAL" in rid or "SAFETY" in rid: pep_status = "NEUTRAL"
                else: pep_status = "FAIL"
            
            results[rid] = pep_status
        return results


# ==============================================================================
# 2. 严格对齐的 TCR Validator (Copy from generate_prompt_2_llm_tcr.txt)
# ==============================================================================
FINAL_TCR_RULES_JSON = """
[
  {
    "rule_id": "TCR_HYDROPHOBIC_SHIELDING",
    "description": "Desolvation: Highly hydrophobic Peptide Core (>=3 residues) requires Hydrophobic/Aromatic CDR3 to exclude water.",
    "python_condition": {
      "cdr3_region": "CENTER_5",
      "peptide_region": "EXPOSED_CORE",
      "peptide_hydrophobic_threshold": 3,
      "cdr3_must_contain": ["Y", "W", "L", "V", "I", "F", "M", "A"],
      "logic_operator": "HYDROPHOBIC_COMPLEMENTARITY",
      "outcome": "PASS"
    }
  },
  {
    "rule_id": "TCR_CHARGE_REPULSION_POS",
    "description": "Electrostatic Clash: Both CDR3 Center and Peptide Exposed Core contain residues of the same POSITIVE charge, creating local repulsion.",
    "python_condition": {
      "cdr3_region": "CENTER_5",
      "peptide_region": "EXPOSED_CORE",
      "target_property": "NET_CHARGE_POSITIVE",
      "logic_operator": "NO_SAME_CHARGE_SIGN",
      "outcome": "FAIL"
    }
  },
  {
    "rule_id": "TCR_CHARGE_REPULSION_NEG",
    "description": "Electrostatic Clash: Both CDR3 Center and Peptide Exposed Core contain residues of the same NEGATIVE charge, creating local repulsion.",
    "python_condition": {
      "cdr3_region": "CENTER_5",
      "peptide_region": "EXPOSED_CORE",
      "target_property": "NET_CHARGE_NEGATIVE",
      "logic_operator": "NO_SAME_CHARGE_SIGN",
      "outcome": "FAIL"
    }
  },
  {
    "rule_id": "TCR_SALT_BRIDGE_DE_RK",
    "description": "Salt Bridge: CDR3 Center Asp/Glu (D/E) forms salt bridges with Peptide Exposed Core Arg/Lys (R/K).",
    "python_condition": {
      "cdr3_region": "CENTER_5",
      "peptide_region": "EXPOSED_CORE",
      "cdr3_must_contain": ["D", "E"],
      "peptide_must_contain": ["R", "K"],
      "logic_operator": "SALT_BRIDGE_MATCH",
      "outcome": "PASS"
    }
  },
  {
    "rule_id": "TCR_ARG_GOVERNOR_MOTIF",
    "description": "Arginine Governor: Central CDR3 Arginine (R) targets Acidic or Polar residues in Peptide.",
    "python_condition": {
      "cdr3_region": "CENTER_5",
      "peptide_region": "EXPOSED_CORE",
      "cdr3_must_contain": ["R"],
      "peptide_must_contain": ["D", "E", "N", "Q", "S", "T"],
      "logic_operator": "ARG_ACIDIC_POLAR_MATCH",
      "outcome": "PASS"
    }
  },
  {
    "rule_id": "TCR_GLYCINE_HINGE_LONG",
    "description": "Hinge: Peptides longer than 10 AA require Glycine (G) in CDR3 for flexibility.",
    "python_condition": {
      "check_level": "SEQUENCE_META",
      "peptide_length_min": 11,
      "cdr3_must_contain": ["G"],
      "logic_operator": "LENGTH_DEPENDENT_FLEXIBILITY",
      "outcome": "PASS"
    }
  },
  {
    "rule_id": "TCR_TYROSINE_UNIVERSAL",
    "description": "Tyrosine Universal: Y in CDR3 Center enables binding via H-bonding and pi-stacking.",
    "python_condition": {
      "cdr3_region": "CENTER_5",
      "cdr3_must_contain": ["Y"],
      "logic_operator": "TYROSINE_PRESENT",
      "outcome": "PASS"
    }
  },
  {
    "rule_id": "TCR_CENTRAL_HYDROPHOBICITY_MISMATCH",
    "description": "Mismatch: High hydrophobicity CDR3 (>=2) cannot bind hydrophilic Peptide Core (>=3).",
    "python_condition": {
      "cdr3_region": "CENTER_5",
      "peptide_region": "EXPOSED_CORE",
      "cdr3_hydrophobic_threshold": 2,
      "peptide_hydrophilic_threshold": 3,
      "logic_operator": "HYDROPHOBIC_MISMATCH",
      "outcome": "FAIL"
    }
  },
  {
    "rule_id": "TCR_POLAR_NETWORK_Q_N",
    "description": "H-Bond Network: CDR3 uses Amide groups (Q/N) to engage with Polar Peptide residues.",
    "python_condition": {
      "cdr3_region": "CENTER_5",
      "peptide_region": "EXPOSED_CORE",
      "cdr3_must_contain": ["Q", "N"],
      "peptide_must_contain": ["S", "T", "N", "Q", "Y"],
      "logic_operator": "CONTAINS_MATCH",
      "outcome": "PASS"
    }
  },
  {
    "rule_id": "TCR_AROMATIC_STACK_W",
    "description": "Pi-Stacking: CDR3 Tryptophan (W) locks into Peptide Aromatics (F/Y/W).",
    "python_condition": {
      "cdr3_region": "CENTER_5",
      "peptide_region": "EXPOSED_CORE",
      "cdr3_must_contain": ["W"],
      "peptide_must_contain": ["F", "Y", "W"],
      "logic_operator": "CONTAINS_MATCH",
      "outcome": "PASS"
    }
  },
  {
    "rule_id": "TCR_THREONINE_CAP",
    "description": "Threonine Capping: CDR3 Threonine (T) provides geometric fit for hydrophobic/small peptides.",
    "python_condition": {
      "cdr3_region": "CENTER_5",
      "peptide_region": "EXPOSED_CORE",
      "cdr3_must_contain": ["T"],
      "peptide_must_contain": ["A", "V", "L", "I", "S"],
      "logic_operator": "CONTAINS_MATCH",
      "outcome": "PASS"
    }
  }
]
"""

class TCRRuleValidator:
    def __init__(self):
        self.rules = json.loads(FINAL_TCR_RULES_JSON)
        self.rule_desc_map = {r['rule_id']: r['description'] for r in self.rules}
        
        self.POS_CHARGE = set(['R', 'K', 'H'])
        self.NEG_CHARGE = set(['D', 'E'])
        self.HYDROPHOBIC = set(['L', 'V', 'I', 'M', 'F', 'Y', 'W', 'A', 'C'])
        self.HYDROPHILIC = set(['D', 'E', 'K', 'R', 'N', 'Q', 'S', 'T', 'H'])
    
    def get_center_window(self, seq, window=5):
        if pd.isna(seq) or len(str(seq)) == 0: return ""
        length = len(seq)
        if length <= window: return seq
        start = (length - window) // 2
        return seq[start : start + window]

    def get_exposed_core(self, pep_seq):
        if pd.isna(pep_seq) or len(str(pep_seq)) < 5: return pep_seq 
        return pep_seq[2:-1] 

    def get_net_charge_sign(self, seq):
        pos = sum(1 for aa in seq if aa in self.POS_CHARGE)
        neg = sum(1 for aa in seq if aa in self.NEG_CHARGE)
        net = pos - neg
        if net > 0: return 1
        elif net < 0: return -1
        else: return 0

    def check_contains_any(self, seq, target_list):
        return any(aa in seq for aa in target_list)

    def count_matches(self, seq, target_set):
        return sum(1 for aa in seq if aa in target_set)

    def check_row(self, cdr3, peptide):
        results = {}
        if not isinstance(cdr3, str) or not isinstance(peptide, str): return {}
        
        cdr3_center = self.get_center_window(cdr3)
        pep_exposed = self.get_exposed_core(peptide)
        
        for rule in self.rules:
            rid = rule['rule_id']
            cond = rule['python_condition']
            op = cond['logic_operator']
            status = "NEUTRAL"

            # ----------------------------------------------------
            # 严格复刻的逻辑分支
            # ----------------------------------------------------
            if op == "NO_SAME_CHARGE_SIGN":
                c_sign = self.get_net_charge_sign(cdr3_center)
                p_sign = self.get_net_charge_sign(pep_exposed)
                target = cond.get('target_property', '')
                if target == "NET_CHARGE_NEGATIVE":
                    if c_sign < 0 and p_sign < 0: status = "FAIL"
                elif target == "NET_CHARGE_POSITIVE":
                    if c_sign > 0 and p_sign > 0: status = "FAIL"
                else: 
                    if c_sign != 0 and c_sign == p_sign: status = "FAIL"

            elif op == "SALT_BRIDGE_MATCH" or op == "ARG_ACIDIC_POLAR_MATCH" or op == "CONTAINS_MATCH":
                c_req = cond.get('cdr3_must_contain', [])
                if self.check_contains_any(cdr3_center, c_req):
                    p_req = cond.get('peptide_must_contain', [])
                    if self.check_contains_any(pep_exposed, p_req):
                        status = "PASS"

            elif op == "HYDROPHOBIC_COMPLEMENTARITY":
                threshold = cond.get('peptide_hydrophobic_threshold', 3)
                p_count = self.count_matches(pep_exposed, self.HYDROPHOBIC)
                if p_count >= threshold:
                    c_req = cond.get('cdr3_must_contain', [])
                    if self.check_contains_any(cdr3_center, c_req):
                        status = "PASS"

            elif op == "LENGTH_DEPENDENT_FLEXIBILITY":
                min_len = cond.get('peptide_length_min', 11)
                if len(peptide) >= min_len:
                    c_req = cond.get('cdr3_must_contain', [])
                    if self.check_contains_any(cdr3_center, c_req):
                        status = "PASS"

            elif op == "TYROSINE_PRESENT":
                c_req = cond.get('cdr3_must_contain', [])
                if self.check_contains_any(cdr3_center, c_req):
                    status = "PASS"

            elif op == "HYDROPHOBIC_MISMATCH":
                c_thresh = cond.get('cdr3_hydrophobic_threshold', 2)
                p_thresh = cond.get('peptide_hydrophilic_threshold', 3)
                c_count = self.count_matches(cdr3_center, self.HYDROPHOBIC)
                p_count = self.count_matches(pep_exposed, self.HYDROPHILIC)
                if c_count >= c_thresh and p_count >= p_thresh:
                    status = "FAIL"

            results[rid] = status
        
        return results

# ================= 2. Hybrid Dataset (Qwen ChatML Version) =================
class HybridInferenceDataset(Dataset):
    def __init__(self, dataframe, qwen_tokenizer, esm_tokenizer):
        self.df = dataframe.reset_index(drop=True)
        self.qwen_tokenizer = qwen_tokenizer
        self.esm_tokenizer = esm_tokenizer
        
        if 'HLA' in self.df.columns: 
            self.rec_col = 'HLA'; self.task_type = 'HLA'; self.validator = HLARuleValidator()
        elif 'tcr' in self.df.columns: 
            self.rec_col = 'tcr'; self.task_type = 'TCR'; self.validator = TCRRuleValidator()
        else: 
            self.rec_col = 'receptor_seq'; self.task_type = 'TCR'; self.validator = TCRRuleValidator()

        # Qwen Special Tokens
        self.im_start_id = qwen_tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = qwen_tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.nl_ids = qwen_tokenizer.encode("\n", add_special_tokens=False)
        self.user_ids = qwen_tokenizer.encode("user", add_special_tokens=False)
        self.asst_ids = qwen_tokenizer.encode("assistant", add_special_tokens=False)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_seq = row[self.rec_col]
        lig_seq = row['peptide'] if 'peptide' in row else row['ligand_seq']
        
        # ESM Tokenization (Shared)
        esm_rec = self.esm_tokenizer(rec_seq, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        esm_lig = self.esm_tokenizer(lig_seq, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

        # -------------------------------------------------------
        # Path A: Standard SFT Prompt (Qwen Format)
        # -------------------------------------------------------
        sft_text_pre = "Analyze the structural compatibility between "
        sft_text_mid = " and "
        sft_text_post = ".\nPrediction of stable complex formation (Yes/No):"

        ids_pre = self.qwen_tokenizer(sft_text_pre, add_special_tokens=False)['input_ids']
        ids_mid = self.qwen_tokenizer(sft_text_mid, add_special_tokens=False)['input_ids']
        ids_post = self.qwen_tokenizer(sft_text_post, add_special_tokens=False)['input_ids']
        
        # User: <|im_start|>user\n{pre}...
        sft_ids_a = [self.im_start_id] + self.user_ids + self.nl_ids + ids_pre
        # Mid: ...{mid}...
        sft_ids_mid = ids_mid
        # Asst Header: ...{post}<|im_end|>\n<|im_start|>assistant\n
        sft_ids_b = ids_post + [self.im_end_id] + self.nl_ids + [self.im_start_id] + self.asst_ids + self.nl_ids

        # -------------------------------------------------------
        # Path B: CoT Prompt (Qwen Format)
        # -------------------------------------------------------
        validation_res = self.validator.check_row(rec_seq, lig_seq)
        if isinstance(validation_res, str): rule_text = validation_res
        else: rule_text = "\n".join([f"[{s}] {r}: ..." for r, s in validation_res.items() if s in ["PASS", "FAIL"]])
        if not rule_text: rule_text = "No specific rules."

        # validation_res = self.validator.check_row(rec_seq, lig_seq)
        # if isinstance(validation_res, str):
        #     rule_text = validation_res
        # else:
        #     rule_lines = []
        #     for rule_id, status in validation_res.items():
        #         if status in ["PASS", "FAIL"]:
        #             # 从 validator.rule_desc_map 获取完整描述
        #             desc = self.validator.rule_desc_map.get(rule_id, rule_id)
        #             rule_lines.append(f"[{status}] {rule_id}: {desc}")
        #     rule_text = "\n".join(rule_lines) if rule_lines else "No specific rules."
        # if not rule_text: rule_text = "No specific rules."

        if self.task_type == 'TCR':
            user_content = f"""**Task**: Analyze the interface compatibility between the TCR CDR3 and Peptide.\n\n**Input Data**:\n- **TCR CDR3**: {rec_seq}\n- **Peptide**: {lig_seq}\n\n**Biophysical Rule Analysis**:\n{rule_text}\n\n**Instructions**:\n1. Focus on the **CDR3 Center** vs **Peptide Exposed Core**.\n2. Evaluate Charge Complementarity vs Repulsion.\n3. Check for Hydrophobic Shielding and Structural Fit.\n4. **Conclusion**: Output final prediction as [[Yes]] or [[No]]."""

            # user_content = f"""**Task**: Analyze the interface compatibility between the TCR CDR3 and Peptide.\n\n**Input Data**:\n- **TCR CDR3**: {rec_seq}\n- **Peptide**: {lig_seq}\n\n**Biophysical Rule Analysis**:\n{rule_text}\n\n**Instructions**:\n1. **Focus on Shape Complementarity**: Analyze how the CDR3 center (residues 5-9) sterically accommodates the peptide's exposed residues (M4-T7). Mention how flexible residues help avoid steric clashes with large side chains like Trp (W5).\n2. **Evaluate Hydrophobic Packing**: Instead of looking for specific geometry like pi-stacking, evaluate the overall "shielding" effect of CDR3 aromatic/aliphatic residues over the peptide's hydrophobic core.\n3. **Polar Compensation**: Identify potential H-bonds between CDR3 polar side chains and the peptide backbone or polar anchors (e.g., T7).\n4. **Charge Profile**: Confirm the absence of electrostatic repulsion in the central contact area.\n5. **Conclusion**: Output final prediction as [[Yes]] or [[No]]."""

            
        else:
            user_content = f"""**Task**: Analyze HLA binding.\n\n**Input Data**:\n- HLA: {rec_seq}\n- Peptide: {lig_seq}\n\n**Rule Analysis**:\n{rule_text}\n\n**Instructions**:\n1. Analyze rules.\n2. Conclusion: [[Yes]] or [[No]]."""

        prefix_text = "Here is the receptor sequence representation: "
        cot_ids_pre = self.qwen_tokenizer(prefix_text, add_special_tokens=False)['input_ids']
        cot_ids_mid = self.qwen_tokenizer(" and the ligand sequence representation: ", add_special_tokens=False)['input_ids']
        cot_ids_suf = self.qwen_tokenizer(".\n\n", add_special_tokens=False)['input_ids']
        cot_ids_content = self.qwen_tokenizer(user_content, add_special_tokens=False)['input_ids']

        cot_ids_a = [self.im_start_id] + self.user_ids + self.nl_ids + cot_ids_pre
        cot_ids_mid_tensor = cot_ids_mid
        # Suffix + Content + EndUser + StartAsst
        cot_ids_b = cot_ids_suf + cot_ids_content + [self.im_end_id] + self.nl_ids + [self.im_start_id] + self.asst_ids + self.nl_ids

        return {
            "label": int(row['label']),
            "esm_rec_ids": esm_rec['input_ids'][0], "esm_rec_mask": esm_rec['attention_mask'][0],
            "esm_lig_ids": esm_lig['input_ids'][0], "esm_lig_mask": esm_lig['attention_mask'][0],
            
            "sft_ids_a": torch.tensor(sft_ids_a, dtype=torch.long), 
            "sft_ids_mid": torch.tensor(sft_ids_mid, dtype=torch.long), 
            "sft_ids_b": torch.tensor(sft_ids_b, dtype=torch.long),
            
            "cot_ids_a": torch.tensor(cot_ids_a, dtype=torch.long), 
            "cot_ids_mid": torch.tensor(cot_ids_mid_tensor, dtype=torch.long), 
            "cot_ids_b": torch.tensor(cot_ids_b, dtype=torch.long)
        }

def collate_fn(batch, pad_id):
    def pad(key): return torch.nn.utils.rnn.pad_sequence([x[key] for x in batch], batch_first=True, padding_value=pad_id)
    def stack(key): return torch.stack([x[key] for x in batch])
    
    return {
        "labels": torch.tensor([x['label'] for x in batch]),
        "rec_ids": stack('esm_rec_ids'), "rec_mask": stack('esm_rec_mask'),
        "lig_ids": stack('esm_lig_ids'), "lig_mask": stack('esm_lig_mask'),
        
        "sft_ids_a": pad('sft_ids_a'), "sft_mask_a": (pad('sft_ids_a')!=pad_id).long(),
        "sft_ids_mid": pad('sft_ids_mid'), "sft_mask_mid": (pad('sft_ids_mid')!=pad_id).long(),
        "sft_ids_b": pad('sft_ids_b'), "sft_mask_b": (pad('sft_ids_b')!=pad_id).long(),
        
        "cot_ids_a": pad('cot_ids_a'), "cot_mask_a": (pad('cot_ids_a')!=pad_id).long(),
        "cot_ids_mid": pad('cot_ids_mid'), "cot_mask_mid": (pad('cot_ids_mid')!=pad_id).long(),
        "cot_ids_b": pad('cot_ids_b'), "cot_mask_b": (pad('cot_ids_b')!=pad_id).long(),
    }

# ================= 3. Model (Qwen Version) =================
class ESMToQwenHybridModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.esm = AutoModel.from_pretrained(ESM_PATH, dtype=torch.bfloat16)
        # Trust Remote Code for Qwen
        self.qwen = AutoModelForCausalLM.from_pretrained(QWEN_PATH, dtype=torch.bfloat16, trust_remote_code=True)
        self.projector = QFormer(esm_dim=self.esm.config.hidden_size, llm_dim=self.qwen.config.hidden_size, num_queries=args.num_queries, num_hidden_layers=6).to(dtype=torch.bfloat16)

        # Apply LoRA
        peft_config_esm = LoraConfig(r=16, lora_alpha=32, target_modules=["query", "key", "value", "dense"], lora_dropout=0.1, bias='none')
        self.esm = get_peft_model(self.esm, peft_config_esm)
        peft_config_qwen = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=True, r=64, lora_alpha=128, target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj","down_proj","up_proj"])
        self.qwen = get_peft_model(self.qwen, peft_config_qwen)

    def _prepare_inputs(self, ids_a, mask_a, ids_mid, mask_mid, ids_b, mask_b, rec_ids, rec_mask, lig_ids, lig_mask):
        with torch.no_grad():
            rec_out = self.esm(input_ids=rec_ids, attention_mask=rec_mask).last_hidden_state
            lig_out = self.esm(input_ids=lig_ids, attention_mask=lig_mask).last_hidden_state
        rec_embeds = self.projector(rec_out, rec_mask)
        lig_embeds = self.projector(lig_out, lig_mask)
        
        # Qwen Embedding Access
        if hasattr(self.qwen, "get_input_embeddings"): embed = self.qwen.get_input_embeddings()
        else: embed = self.qwen.model.embed_tokens
        
        inputs_embeds = torch.cat([embed(ids_a), rec_embeds, embed(ids_mid), lig_embeds, embed(ids_b)], dim=1)
        
        B, Q = ids_a.shape[0], self.projector.num_queries
        p_mask = torch.ones((B, Q), device=ids_a.device, dtype=mask_a.dtype)
        attention_mask = torch.cat([mask_a, p_mask, mask_mid, p_mask, mask_b], dim=1)
        return inputs_embeds, attention_mask

    def get_logits(self, batch):
        inputs_embeds, attention_mask = self._prepare_inputs(
            batch['sft_ids_a'], batch['sft_mask_a'], batch['sft_ids_mid'], batch['sft_mask_mid'], batch['sft_ids_b'], batch['sft_mask_b'],
            batch['rec_ids'], batch['rec_mask'], batch['lig_ids'], batch['lig_mask']
        )
        outputs = self.qwen(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        last_token_logits = outputs.logits[:, -1, :]
        return last_token_logits

    def generate_text(self, batch):
        inputs_embeds, attention_mask = self._prepare_inputs(
            batch['cot_ids_a'], batch['cot_mask_a'], batch['cot_ids_mid'], batch['cot_mask_mid'], batch['cot_ids_b'], batch['cot_mask_b'],
            batch['rec_ids'], batch['rec_mask'], batch['lig_ids'], batch['lig_mask']
        )
        return self.qwen.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=self.qwen.config.pad_token_id, eos_token_id=self.qwen.config.eos_token_id,
            do_sample=False, 
            temperature=0.,
            repetition_penalty=1.2,
        )

# ================= 4. Main =================
def calculate_auc_from_logits(labels, prob_yes):
    try: return roc_auc_score(labels, prob_yes)
    except: return 0.5

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="/mnt/lustre/guopeijin/Immune_LLM/code/trained_models/sft_qwen_mixed_v4/stage2_sft_lora_lora_mixed/pytorch_model.bin")
    parser.add_argument("--num_queries", type=int, default=64)
    return parser.parse_args()

def main():
    args = parse_args()
    accelerator = Accelerator()
    
    # Tokenizer Setup
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
    if qwen_tokenizer.pad_token is None: qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
    esm_tokenizer = AutoTokenizer.from_pretrained(ESM_PATH)
    
    # Target Token IDs (for Qwen)
    # Qwen 的词表里 Yes/No 的 ID 可能不同，建议打印确认
    yes_id = qwen_tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = qwen_tokenizer.encode("No", add_special_tokens=False)[0]
    
    model = ESMToQwenHybridModel(args)
    if accelerator.is_main_process:
        print(f"Loading weights from {args.checkpoint_path}")
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    model = accelerator.prepare(model)
    model.eval()

    summary = []

    for ds_name, ds_path in BENCHMARK_DATASETS:
        if accelerator.is_main_process: print(f"\n🧪 Hybrid Testing (Qwen): {ds_name}")
        if not os.path.exists(ds_path): continue
        df = pd.read_csv(ds_path)
        
        ds = HybridInferenceDataset(df, qwen_tokenizer, esm_tokenizer)
        dl = DataLoader(ds, batch_size=PER_DEVICE_BATCH_SIZE, shuffle=False, 
                        collate_fn=lambda x: collate_fn(x, qwen_tokenizer.pad_token_id), num_workers=4)
        dl = accelerator.prepare(dl)
        
        all_probs = []
        all_labels = []
        all_texts = []
        
        for i, batch in enumerate(tqdm(dl, disable=not accelerator.is_main_process)):
            with torch.no_grad():
                unwrapped_model = accelerator.unwrap_model(model)

                # ===== 1. 打印完整输入文本（CoT 路径）=====
                if i == 0 and accelerator.is_main_process:
                    # 解码 CoT 输入的三段式 prompt
                    input_ids_a = batch['cot_ids_a'][0]
                    input_ids_mid = batch['cot_ids_mid'][0]
                    input_ids_b = batch['cot_ids_b'][0]
                    
                    # 拼接完整输入（注意：中间会插入 ESM embeddings，此处仅展示文本部分）
                    full_input_ids = torch.cat([input_ids_a, input_ids_mid, input_ids_b], dim=0)
                    full_input_text = qwen_tokenizer.decode(full_input_ids, skip_special_tokens=False)
                    
                    print("\n" + "="*80)
                    print("🔍 完整输入 Prompt (CoT 路径，第1个样本):")
                    print("="*80)
                    print(full_input_text)
                    print("="*80 + "\n")
                
                # 2. 获取 logits（SFT 路径）
                logits = unwrapped_model.get_logits(batch)
                yes_logits = logits[:, yes_id]
                no_logits = logits[:, no_id]
                probs = torch.exp(yes_logits) / (torch.exp(yes_logits) + torch.exp(no_logits))
                
                # 3. 生成文本（CoT 路径）
                if i < 20:  # 仅打印前1个 batch
                    gen_ids = unwrapped_model.generate_text(batch)
                    gen_ids = accelerator.pad_across_processes(gen_ids, dim=1, pad_index=qwen_tokenizer.pad_token_id)
                    gen_ids = accelerator.gather(gen_ids)
                    
                    if accelerator.is_main_process:
                        # ===== 2. 打印完整输出文本 =====
                        texts = qwen_tokenizer.batch_decode(gen_ids, skip_special_tokens=False)
                        for j, txt in enumerate(texts[:2]):  # 打印前2个样本
                            print("\n" + "="*80)
                            print(f"🤖 模型完整输出 (样本 #{j}):")
                            print("="*80)
                            print(txt.strip())
                            print("="*80 + "\n")

                # 1. Logits
                logits = unwrapped_model.get_logits(batch)
                yes_logits = logits[:, yes_id]
                no_logits = logits[:, no_id]
                probs = torch.exp(yes_logits) / (torch.exp(yes_logits) + torch.exp(no_logits))
                
                probs = accelerator.gather_for_metrics(probs)
                labels = accelerator.gather_for_metrics(batch['labels'])
                all_probs.extend(probs.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

                # 2. Generate Text (Sample)
                if i < 2: 
                    gen_ids = unwrapped_model.generate_text(batch)
                    gen_ids = accelerator.pad_across_processes(gen_ids, dim=1, pad_index=qwen_tokenizer.pad_token_id)
                    gen_ids = accelerator.gather(gen_ids)
                    if accelerator.is_main_process:
                        texts = qwen_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                        all_texts.extend(texts)
        
        if accelerator.is_main_process:
            final_probs = np.array(all_probs)
            final_labels = np.array(all_labels)
            
            aucs = []
            for _ in range(BOOTSTRAP_ROUNDS):
                indices = np.random.choice(len(final_probs), size=len(final_probs), replace=True)
                aucs.append(calculate_auc_from_logits(final_labels[indices], final_probs[indices]))
            
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            print(f"   📈 SFT-Path AUC: {mean_auc:.4f} ± {std_auc:.4f}")
            summary.append({"Dataset": ds_name, "AUC": mean_auc})
            
            with open(f"case_study_qwen_{ds_name}.txt", "w") as f:
                for t in all_texts: f.write(t.replace("\n", " ") + "\n")

    if accelerator.is_main_process:
        print("\nSummary:")
        print(pd.DataFrame(summary))

if __name__ == "__main__":
    main()

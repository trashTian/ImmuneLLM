import pandas as pd
import json
import os
from tqdm import tqdm

# ==========================================
# 1. 终极规则集 (The Final Golden Set)
# ==========================================
FINAL_RULES_JSON = """
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

# ==========================================
# 2. 验证逻辑 (Validator V3)
# ==========================================
class HLARuleValidatorV3:
    def __init__(self, rules_json_str):
        self.rules = json.loads(rules_json_str)
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

# ==========================================
# 3. 集成 Prompt 生成器 (Integrated Generator)
# ==========================================
def generate_integrated_jsonl(csv_path, output_jsonl, sample_size=None):
    print(f"Reading {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: File not found at {csv_path}")
        return

    if sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    validator = HLARuleValidatorV3(FINAL_RULES_JSON)
    records = []

    print(f"Processing {len(df)} records -> {output_jsonl}")
    file_prefix = os.path.basename(csv_path).split('.')[0]

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        hla_seq = row['HLA']
        pep_seq = row['peptide']
        label_raw = row['label']
        label_text = "Yes" if label_raw == 1 else "No"
        
        # 1. 规则验证
        validation_res = validator.check_row(hla_seq, pep_seq)
        active_logs = [f"[{status}] {rid}: {validator.rule_desc_map.get(rid)}" 
                       for rid, status in validation_res.items() if status in ["PASS", "FAIL"]]
        
        rule_text = "\n".join(active_logs) if active_logs else "No specific canonical rule triggered."
        
        # ==========================================
        # Prompt A: Blind Test (不带 Ground Truth)
        # ==========================================
        system_blind = "You are an expert Structural Biologist. Perform a blind prediction of HLA-peptide binding. You must end your response strictly with the final prediction enclosed in brackets: [[Yes]] or [[No]]."
        user_blind = f"""**Task**: Analyze the biophysical compatibility between the HLA and Peptide and predict binding.

**Input Data**:
- **HLA Pseudo-sequence**: {hla_seq}
- **Peptide Sequence**: {pep_seq}

**Biophysical Rule Analysis**:
{rule_text}

**Instructions**:
1. Analyze the compatibility based on the Rule Report and general biophysics (Charge, Hydrophobicity, Sterics).
2. Think step-by-step about Anchor matches and potential conflicts.
3. **Conclusion**: Based on your analysis, output the final prediction. 
   Format: **[[Yes]]** for Binding, **[[No]]** for Non-binding.
"""
        messages_blind = [
            {"role": "system", "content": system_blind},
            {"role": "user", "content": user_blind}
        ]

        # ==========================================
        # Prompt B: Hindsight (带 Ground Truth)
        # ==========================================
        system_hindsight = "You are an expert Structural Biologist. You are correcting a previous misdiagnosis. Analyze the biophysics to derive the correct conclusion."
        user_hindsight = f"""**Task**: Re-evaluate the HLA-Peptide interaction mechanism.

**Input Data**:
- **HLA Pseudo-sequence**: {hla_seq}
- **Peptide Sequence**: {pep_seq}
- **Ground Truth**: {label_text} (Use this to guide your reasoning, but DO NOT explicitly mention "Because the label is {label_text}" in the final output).

**Biophysical Rule Analysis**:
{rule_text}

**Instructions**:
Your previous analysis was incorrect. Now, knowing the result is **{label_text}**, perform a "Root Cause Analysis":
1. Look for **overlooked features** (e.g., secondary anchors, backbone H-bonds, or subtle steric compatibility) that justify this label.
2. If the rules failed, explain **biophysically** why the rule was overruled in this specific case.
3. **Style Requirement**: Write the reasoning as a logical deduction derived from the sequence features, leading naturally to the conclusion. Make it sound like a confident, correct *a priori* prediction.
4. **Conclusion**: End strictly with: **[[{label_text}]]**.
"""
        messages_hindsight = [
            {"role": "system", "content": system_hindsight},
            {"role": "user", "content": user_hindsight}
        ]

        # ==========================================
        # 构造单行数据对象
        # ==========================================
        record = {
            "custom_id": f"{file_prefix}_{idx}_{label_raw}",
            "ground_truth": label_raw, # 方便读取时直接对比，不用解析字符串
            "prompt_blind": messages_blind,
            "prompt_hindsight": messages_hindsight
        }
        records.append(record)

    # 保存文件
    with open(output_jsonl, 'w') as f:
        for item in records:
            f.write(json.dumps(item) + '\n')
    
    print(f"Success! Saved {len(records)} records to {output_jsonl}")

# ==========================================
# 4. 执行
# ==========================================
if __name__ == "__main__":
    
    # 1. 训练集
    # train_csv = '/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/train_fold_1.csv'
    # train_out = 'hla_i_train_integrated.jsonl'
    # generate_integrated_jsonl(train_csv, train_out, sample_size=None)
    
    # # # 2. 验证集
    # val_csv = '/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/val_fold_1.csv'
    # val_out = 'hla_i_val_integrated.jsonl'
    # generate_integrated_jsonl(val_csv, val_out, sample_size=None)

    # # 2. 测试集
    val_csv = '/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/external_set.csv'
    val_out = 'hla_i_external_integrated.jsonl'
    generate_integrated_jsonl(val_csv, val_out, sample_size=None)
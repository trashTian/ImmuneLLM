import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm

# ==========================================
# 配置路径
# ==========================================
# 输入：原始 CSV 文件 (必须包含 tcr, peptide, label 列)
INPUT_CSV = "/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/covid_set.csv"

# 输出：生成的 Prompt 文件 (准备喂给 LLM)
OUTPUT_JSONL = "tcr_covid_integrated.jsonl"


# ==========================================
# 1. 终极 TCR 规则集 (The All-Star Set: 11 Rules)
# ==========================================
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

# ==========================================
# 2. Validator V7 (TCR 专用)
# ==========================================
class TCRRuleValidatorV7:
    def __init__(self, rules_json_str):
        self.rules = json.loads(rules_json_str)
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

            # 逻辑分发
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

# ==========================================
# 3. 集成 Prompt 生成逻辑 (Prompt B 已修正)
# ==========================================
def generate_tcr_prompts(csv_path, output_jsonl, sample_size=None):
    print(f"Reading {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: File not found at {csv_path}")
        return

    if sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    # 初始化验证器
    validator = TCRRuleValidatorV7(FINAL_TCR_RULES_JSON)
    records = []

    print(f"Generating prompts for {len(df)} records...")
    file_prefix = os.path.basename(csv_path).split('.')[0]

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        tcr_seq = row['tcr']
        pep_seq = row['peptide']
        label_raw = row['label']
        label_text = "Yes" if label_raw == 1 else "No"
        
        # 1. 运行规则验证
        validation_res = validator.check_row(tcr_seq, pep_seq)
        
        # 2. 提取 Log
        active_logs = [f"[{status}] {rid}: {validator.rule_desc_map.get(rid)}" 
                       for rid, status in validation_res.items() if status in ["PASS", "FAIL"]]
        
        rule_text = "\n".join(active_logs) if active_logs else "No specific strong motifs detected."
        
        # ------------------------------------------
        # Prompt A: Blind Test (不带 Ground Truth)
        # ------------------------------------------
        system_blind = "You are an expert Structural Biologist. Perform a blind prediction of TCR-peptide binding likelihood. You must end your response strictly with the final prediction enclosed in brackets: [[Yes]] or [[No]]."
        
        user_blind = f"""**Task**: Analyze the interface compatibility between the TCR CDR3 and Peptide.

**Input Data**:
- **TCR CDR3**: {tcr_seq}
- **Peptide**: {pep_seq}

**Biophysical Rule Analysis**:
{rule_text}

**Instructions**:
1. Focus on the **CDR3 Center** vs **Peptide Exposed Core**.
2. Evaluate Charge Complementarity vs Repulsion.
3. Check for Hydrophobic Shielding and Structural Fit.
4. **Conclusion**: Output final prediction as [[Yes]] or [[No]].
"""
        messages_blind = [
            {"role": "system", "content": system_blind},
            {"role": "user", "content": user_blind}
        ]

        # ------------------------------------------
        # Prompt B: Hindsight (带 Ground Truth - 对齐 HLA 策略)
        # ------------------------------------------
        # 关键修改：强调 "Correcting a misdiagnosis" 和 "Style Requirement"
        system_hindsight = "You are an expert Structural Biologist. You are correcting a previous misdiagnosis. Analyze the biophysics to derive the correct conclusion."
        
        user_hindsight = f"""**Task**: Re-evaluate the TCR-Peptide interaction mechanism.

**Input Data**:
- **TCR CDR3**: {tcr_seq}
- **Peptide**: {pep_seq}
- **Ground Truth**: {label_text} (Use this to guide your reasoning, but DO NOT explicitly mention "Because the label is {label_text}" in the final output).

**Biophysical Rule Analysis**:
{rule_text}

**Instructions**:
Your previous analysis was incorrect. Now, knowing the result is **{label_text}**, perform a "Root Cause Analysis":
1. Look for **overlooked features** (e.g., conformational flexibility, water-mediated bridges, or specific side-chain packing) that justify this label.
2. If rules conflict with the label, explain **biophysically** why the rule was overruled in this specific context (e.g., distance, screening).
3. **Style Requirement**: Write the reasoning as a logical deduction derived from the sequence features, leading naturally to the conclusion. Make it sound like a confident, correct *a priori* prediction.
4. **Conclusion**: End strictly with: **[[{label_text}]]**.
"""
        messages_hindsight = [
            {"role": "system", "content": system_hindsight},
            {"role": "user", "content": user_hindsight}
        ]

        # ------------------------------------------
        # 构造数据对象
        # ------------------------------------------
        record = {
            "custom_id": f"{file_prefix}_{idx}_{label_raw}",
            "ground_truth": int(label_raw), 
            "prompt_blind": messages_blind,
            "prompt_hindsight": messages_hindsight
        }
        records.append(record)

    # 保存文件
    print(f"Saving to {output_jsonl}...")
    with open(output_jsonl, 'w') as f:
        for item in records:
            f.write(json.dumps(item) + '\n')
    
    print(f"Success! Saved {len(records)} records.")

# ==========================================
# 4. 执行
# ==========================================
if __name__ == "__main__":
    generate_tcr_prompts(INPUT_CSV, OUTPUT_JSONL, sample_size=None)
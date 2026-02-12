import pandas as pd
import json
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. 更新后的 TCR 规则集 (Merged & Enhanced)
# ==========================================
TCR_RULES_JSON = """
[
  {
    "rule_id": "TCR_HYDROPHOBIC_SHIELDING",
    "description": "Desolvation (MVP): Highly hydrophobic Peptide Core (>=3 residues) requires Hydrophobic/Aromatic CDR3 to exclude water.",
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
    "description": "Critical Failure: If CDR3 Center is Net Positive, Peptide Exposed Core cannot be Net Positive.",
    "python_condition": {
      "cdr3_region": "CENTER_5",
      "peptide_region": "EXPOSED_CORE",
      "logic_operator": "NO_SAME_CHARGE_SIGN",
      "outcome": "FAIL"
    }
  },
  {
    "rule_id": "TCR_CHARGE_REPULSION_NEG",
    "description": "Critical Failure: If CDR3 Center is Net Negative, Peptide Exposed Core cannot be Net Negative.",
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
# 2. Validator V7 (支持计数逻辑)
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

            # === Rule 1: NO_SAME_CHARGE_SIGN ===
            if op == "NO_SAME_CHARGE_SIGN":
                c_sign = self.get_net_charge_sign(cdr3_center)
                p_sign = self.get_net_charge_sign(pep_exposed)
                target = cond.get('target_property', '')
                
                # Check specifics if target defined, else generic clash
                if target == "NET_CHARGE_NEGATIVE":
                    if c_sign < 0 and p_sign < 0: status = "FAIL"
                elif target == "NET_CHARGE_POSITIVE":
                    if c_sign > 0 and p_sign > 0: status = "FAIL"
                else: # Generic
                    if c_sign != 0 and c_sign == p_sign: status = "FAIL"

            # === Rule 2: SALT_BRIDGE & ARG_MATCH (Logic same as Contains) ===
            elif op in ["SALT_BRIDGE_MATCH", "ARG_ACIDIC_POLAR_MATCH", "CONTAINS_MATCH"]:
                c_req = cond.get('cdr3_must_contain', [])
                if self.check_contains_any(cdr3_center, c_req):
                    p_req = cond.get('peptide_must_contain', [])
                    if self.check_contains_any(pep_exposed, p_req):
                        status = "PASS"

            # === Rule 3: HYDROPHOBIC_COMPLEMENTARITY ===
            elif op == "HYDROPHOBIC_COMPLEMENTARITY":
                threshold = cond.get('peptide_hydrophobic_threshold', 3)
                p_count = self.count_matches(pep_exposed, self.HYDROPHOBIC)
                if p_count >= threshold:
                    c_req = cond.get('cdr3_must_contain', [])
                    if self.check_contains_any(cdr3_center, c_req):
                        status = "PASS"

            # === Rule 5: LENGTH_DEPENDENT_FLEXIBILITY ===
            elif op == "LENGTH_DEPENDENT_FLEXIBILITY":
                min_len = cond.get('peptide_length_min', 11)
                if len(peptide) >= min_len:
                    c_req = cond.get('cdr3_must_contain', []) # G
                    if self.check_contains_any(cdr3_center, c_req):
                        status = "PASS"

            # === Rule 6: TYROSINE_PRESENT ===
            elif op == "TYROSINE_PRESENT":
                c_req = cond.get('cdr3_must_contain', []) # Y
                if self.check_contains_any(cdr3_center, c_req):
                    status = "PASS"

            # === Rule 7: HYDROPHOBIC_MISMATCH ===
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
# 3. 运行统计
# ==========================================
def analyze_tcr_rules(csv_path, rules_json, sample_size=None):
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: File not found.")
        return None
    
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    validator = TCRRuleValidatorV7(rules_json)
    
    stats = {rule['rule_id']: {'Triggered': 0, 'PASS': 0, 'FAIL': 0, 
                               'PASS_PosLabels': 0, 'FAIL_PosLabels': 0} 
             for rule in validator.rules}
    
    print("Running Validator V7...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        tcr = row['tcr']
        pep = row['peptide']
        label = row['label']
        
        res = validator.check_row(tcr, pep)
        
        for rid, status in res.items():
            if status == "NEUTRAL": continue
            
            stats[rid]['Triggered'] += 1
            stats[rid][status] += 1
            
            if status == "PASS" and label == 1:
                stats[rid]['PASS_PosLabels'] += 1
            if status == "FAIL" and label == 1:
                stats[rid]['FAIL_PosLabels'] += 1 

    report_data = []
    for rid, s in stats.items():
        triggered = s['Triggered']
        if triggered == 0:
            report_data.append({"Rule ID": rid, "Coverage": "0%", "Conflict (FAIL)": "N/A"})
            continue
            
        pass_count = s['PASS']
        fail_count = s['FAIL']
        coverage = (triggered / len(df)) * 100
        
        precision = (s['PASS_PosLabels'] / pass_count * 100) if pass_count > 0 else 0.0
        conflict = (s['FAIL_PosLabels'] / fail_count * 100) if fail_count > 0 else 0.0
        
        report_data.append({
            "Rule ID": rid,
            "Coverage": f"{coverage:.2f}%",
            "PASS Count": pass_count,
            "FAIL Count": fail_count,
            "Precision (PASS)": f"{precision:.1f}%",
            "Conflict (FAIL)": f"{conflict:.1f}%"
        })
        
    return pd.DataFrame(report_data)

if __name__ == "__main__":
    csv_file_path = '/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/dataset.csv'
    report = analyze_tcr_rules(csv_file_path, TCR_RULES_JSON, sample_size=None)
    
    if report is not None:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print("\n===== TCR Rule V7 Evaluation =====")
        print(report)
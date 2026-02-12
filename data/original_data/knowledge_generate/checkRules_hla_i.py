import pandas as pd
import json
import numpy as np
from tqdm import tqdm

# ==========================================
# Validator V3 (Compatible with Complex JSON)
# ==========================================
class HLARuleValidatorV3:
    def __init__(self, rules_json_str):
        self.rules = json.loads(rules_json_str)

    def check_row(self, hla_seq, pep_seq):
        results = {}
        
        # [Safety Check]
        if not isinstance(pep_seq, str) or len(pep_seq) < 2:
            return {r['rule_id']: "ERROR_PEP_LEN" for r in self.rules}
        if not isinstance(hla_seq, str) or len(hla_seq) < 34:
            return {r['rule_id']: "ERROR_HLA_LEN" for r in self.rules}

        for rule in self.rules:
            rid = rule['rule_id']
            cond = rule['python_condition']
            
            # --- Step A: HLA Trigger Logic ---
            hla_triggered = True
            h_indices = cond.get('hla_indices', [])
            h_values = cond.get('hla_values', [])

            # Handle case where h_indices is length 1 but h_values is flat list
            # Normalize h_values to always be a list of possibilities for each index
            if len(h_indices) > 0:
                # 检查 h_values 是否是嵌套列表 (e.g. [["M", "Q"], ["V", "A"]])
                # 还是简单列表 (e.g. ["E"]) 用于单索引
                # 或者混合列表 (e.g. ["N", ["Y", "F"]])
                
                for i, h_idx in enumerate(h_indices):
                    if h_idx >= len(hla_seq):
                        hla_triggered = False; break
                    
                    target_val_container = h_values[i] if i < len(h_values) else []
                    
                    # 如果容器是字符串 (e.g. "N")，转为列表 ["N"]
                    if isinstance(target_val_container, str):
                        target_val_container = [target_val_container]
                    
                    # 检查当前位点是否匹配
                    if hla_seq[h_idx] not in target_val_container:
                        hla_triggered = False
                        break
            
            if not hla_triggered:
                results[rid] = "NOT_TRIGGERED"
                continue

            # --- Step B: Peptide Validation Logic ---
            pep_status = "NEUTRAL"
            
            # 获取需要检查的 peptide 索引 (支持 int 或 list of int)
            p_indices = cond['peptide_index']
            if isinstance(p_indices, int):
                p_indices = [p_indices]
            elif p_indices == "BOTH_ANCHORS": # Legacy support
                p_indices = [1, -1]
            
            # 获取允许/禁止列表
            allowed = cond.get('peptide_allowed', [])
            forbidden = cond.get('peptide_forbidden', [])
            
            # 逻辑：检查所有指定的肽段位置
            # 1. 如果任何一个位置命中 Forbidden -> FAIL (一票否决)
            # 2. 如果 allowed 列表存在：
            #    - 要求所有指定位置都必须在 allowed 里？(Strict)
            #    - 还是只要有一个在？
            #    - 针对 Rule 17 (B35) 和 Rule 38 (Safety)，通常意味着检查的所有位置都必须符合要求。
            
            has_forbidden = False
            all_allowed = True
            
            for p_idx in p_indices:
                try:
                    res = pep_seq[p_idx]
                except IndexError:
                    has_forbidden = True # Treat index error as fail
                    break
                
                if res in forbidden:
                    has_forbidden = True
                    break # Found a violation
                
                if allowed and (res not in allowed):
                    all_allowed = False
            
            if has_forbidden:
                pep_status = "FAIL"
            elif allowed and all_allowed:
                pep_status = "PASS"
            elif allowed and not all_allowed:
                # 这是一个灰色地带。如果定义了 allowed 列表，但残基不在里面，
                # 且也没在 forbidden 里。
                # 对于 Rule 38 (Safety Net)，我们通常只关心 forbidden。
                # 对于 Rule 1 (A02)，不在 allowed (L/M/V...) 里通常意味着不结合。
                # 策略：如果 allowed 列表不为空，且没通过 allowed，视为 FAIL (Strict Mode)
                # 除非规则 ID 包含 "SAFETY" 或 "GENERAL"
                if "GENERAL" in rid or "SAFETY" in rid:
                    pep_status = "NEUTRAL"
                else:
                    pep_status = "FAIL"
            else:
                pep_status = "NEUTRAL"

            results[rid] = pep_status
            
        return results

# ==========================================
# Run Analysis
# ==========================================
def analyze_rules(csv_path, rules_json_str, sample_size=None):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    validator = HLARuleValidatorV3(rules_json_str)
    
    stats = {rule['rule_id']: {'Triggered': 0, 'PASS': 0, 'FAIL': 0, 
                               'PASS_PosLabels': 0, 'FAIL_PosLabels': 0} 
             for rule in validator.rules}
    
    print("Running validator V3...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        hla = row['HLA']
        pep = row['peptide']
        label = row['label']
        
        res = validator.check_row(hla, pep)
        
        for rid, status in res.items():
            if "ERROR" in status or status == "NOT_TRIGGERED" or status == "NEUTRAL":
                continue
            
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
            report_data.append({"Rule ID": rid, "Coverage": "0%"})
            continue
            
        pass_count = s['PASS']
        fail_count = s['FAIL']
        
        coverage = (triggered / len(df)) * 100
        precision = (s['PASS_PosLabels'] / pass_count * 100) if pass_count > 0 else 0.0
        conflict = (s['FAIL_PosLabels'] / fail_count * 100) if fail_count > 0 else 0.0
        
        report_data.append({
            "Rule ID": rid,
            "Triggered": triggered,
            "Coverage": f"{coverage:.2f}%",
            "PASS Count": pass_count,
            "FAIL Count": fail_count,
            "Precision (PASS)": f"{precision:.1f}%",
            "Conflict (FAIL=1)": f"{conflict:.1f}%"
        })
        
    return pd.DataFrame(report_data)

if __name__ == "__main__":
    # 请把大模型新生成的 JSON 放在这里
    NEW_RULES_JSON = """
    [
  {
    "rule_id": "HLA_I_A02_HYDROPHOBIC_ANCHOR",
    "description": "A02 Supertype: Index 3 is M/Q. Creates a hydrophobic B-pocket favoring L/M/I/V at P2.",
    "python_condition": {
      "hla_indices": [3],
      "hla_values": ["M", "Q"],
      "peptide_index": 1,
      "peptide_allowed": ["L", "M", "I", "V", "Q", "A", "T"],
      "peptide_forbidden": ["D", "E", "R", "K", "P"]
    }
  },
  {
    "rule_id": "HLA_I_B27_SALT_BRIDGE_FINAL",
    "description": "B27/B14 Supertype: Index 3 is E. Strongly prefers Basic P2 (R/K), but tolerates Hydrophobic. Strictly forbids Acidic.",
    "python_condition": {
      "hla_indices": [3],
      "hla_values": ["E"],
      "peptide_index": 1,
      "peptide_allowed": ["R", "K", "H", "Q", "L", "M", "V", "I"],
      "peptide_forbidden": ["D", "E", "P"] 
    }
  },
  {
    "rule_id": "HLA_I_B44_ACIDIC_P2_GOLD",
    "description": "B44 Supertype: Index 3 is K. Strictly requires Acidic P2 (E/D). This is a high-confidence physical law.",
    "python_condition": {
      "hla_indices": [3],
      "hla_values": ["K"],
      "peptide_index": 1,
      "peptide_allowed": ["E", "D"],
      "peptide_forbidden": ["R", "K", "H", "L", "M", "F", "Y", "W"]
    }
  },
  {
    "rule_id": "HLA_I_A03_A11_BASIC_CTERM",
    "description": "A03/A11: Index 22 is D. Prefers Basic C-term, forbids Acidic C-term.",
    "python_condition": {
      "hla_indices": [22],
      "hla_values": ["D"],
      "peptide_index": -1,
      "peptide_allowed": ["K", "R", "H", "Y", "F", "L"],
      "peptide_forbidden": ["D", "E"]
    }
  },
  {
    "rule_id": "HLA_I_A01_TYROSINE_CTERM",
    "description": "A01: Index 22 is Y. Prefers Aromatic/Hydrophobic C-term.",
    "python_condition": {
      "hla_indices": [22],
      "hla_values": ["Y"],
      "peptide_index": -1,
      "peptide_allowed": ["Y", "F", "L", "I", "W"],
      "peptide_forbidden": ["D", "E", "P", "R", "K"]
    }
  },
  {
    "rule_id": "HLA_I_C_HYDROPHOBIC_PREF",
    "description": "HLA-C: Index 22 is F or Y. Prefers Hydrophobic C-term.",
    "python_condition": {
      "hla_indices": [22],
      "hla_values": ["F", "Y"],
      "peptide_index": -1,
      "peptide_allowed": ["L", "I", "V", "M", "F", "A"],
      "peptide_forbidden": ["D", "E", "R", "K"]
    }
  },
  {
    "rule_id": "HLA_I_A26_TYROSINE_CTERM",
    "description": "A26: Index 22 is H. Prefers Aromatic C-term.",
    "python_condition": {
      "hla_indices": [22],
      "hla_values": ["H"],
      "peptide_index": -1,
      "peptide_allowed": ["Y", "F", "L", "V"],
      "peptide_forbidden": ["D", "E"]
    }
  },
  {
    "rule_id": "HLA_I_STERIC_F_GLY",
    "description": "Physics: Bulky F-pocket (F/Y/W at Index 22) discourages Glycine (G) due to entropy.",
    "python_condition": {
      "hla_indices": [22],
      "hla_values": ["F", "Y", "W"],
      "peptide_index": -1,
      "peptide_allowed": ["L", "I", "V", "F", "Y"],
      "peptide_forbidden": ["G", "P"]
    }
  },
  {
    "rule_id": "HLA_I_GENERAL_CHARGE_SAFETY",
    "description": "Physics: Double Charge Repulsion. If both anchors are D/E, binding fails.",
    "python_condition": {
      "hla_indices": [3, 22],
      "hla_values": [["L", "I", "V", "M", "Y", "F", "W"], ["L", "I", "V", "M", "Y", "F", "W"]],
      "peptide_index": "BOTH_ANCHORS",
      "peptide_allowed": [],
      "peptide_forbidden": ["D", "E"]
    }
  },
  {
    "rule_id": "HLA_I_A24_AROMATIC_PRECISION",
    "description": "A24 Fix: Instead of generic Index 20 check, we check the specific wall configuration. Index 20 is F/Y AND Index 3 is Y/F/M. Prefers Aromatic P2.",
    "python_condition": {
      "hla_indices": [3, 20],
      "hla_values": [["Y", "F", "M"], ["F", "Y"]], 
      "peptide_index": 1,
      "peptide_allowed": ["Y", "F", "W", "L", "M", "I"],
      "peptide_forbidden": ["D", "E", "R", "K", "P", "G"]
    }
  },
  {
    "rule_id": "HLA_I_B07_PROLINE_PRECISION",
    "description": "B07/B35 Fix: The 'Proline Shelf'. Index 6 (Res 63) is N/Y/F. Strongly favors Proline at P2.",
    "python_condition": {
      "hla_indices": [6],
      "hla_values": ["N", "Y", "F"],
      "peptide_index": 1,
      "peptide_allowed": ["P", "A", "V"],
      "peptide_forbidden": ["W", "F", "Y", "D", "E", "R", "K"]
    }
  }
]
    """
    csv_file_path = '/mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/dataset.csv'
    report = analyze_rules(csv_file_path, NEW_RULES_JSON)
    print(report)



    """
    
                              Rule ID  Triggered Coverage  PASS Count  FAIL Count Precision (PASS) Conflict (FAIL=1)
0    HLA_I_A02_HYDROPHOBIC_ANCHOR   425740.0   42.41%    232388.0    193352.0            44.3%             16.8%
1     HLA_I_B27_SALT_BRIDGE_FINAL   231884.0   23.10%    121772.0    110112.0            43.5%             23.1%
2        HLA_I_B44_ACIDIC_P2_GOLD    87528.0    8.72%     35893.0     51635.0            80.5%              2.5%
3       HLA_I_A03_A11_BASIC_CTERM   283575.0   28.25%    166925.0    116650.0            49.7%             16.0%
4        HLA_I_A01_TYROSINE_CTERM   322770.0   32.15%    166226.0    156544.0            40.8%             21.3%
5        HLA_I_C_HYDROPHOBIC_PREF   111452.0   11.10%     71234.0     40218.0            48.2%              4.5%
6        HLA_I_A26_TYROSINE_CTERM    20977.0    2.09%     11194.0      9783.0            37.7%             17.9%
7              HLA_I_STERIC_F_GLY   111452.0   11.10%     68455.0     42997.0            46.0%             10.7%
8     HLA_I_GENERAL_CHARGE_SAFETY    23449.0    2.34%         0.0     23449.0             0.0%              4.4%
9    HLA_I_A24_AROMATIC_PRECISION   417471.0   41.58%    137934.0    279537.0            44.2%             25.8%
10    HLA_I_B07_PROLINE_PRECISION   238426.0   23.75%     72087.0    166339.0            47.4%             18.9%
    """
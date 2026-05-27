import json
import pandas as pd
import requests
import re
import time
import os
from tqdm import tqdm
from datetime import datetime

# ================= 配置区域 =================
CSV_FILE_PATH = r"D:\Desktop\test_antigenHLAI\Data\crystal_structure\info_noredudant_contacts.csv"
JSONL_FILE_PATH = r"D:\Desktop\immunellm_test_pdb_results.jsonl"
OUTPUT_EVAL_FILE = r"D:\Desktop\eval_immunellm_test_pdb_results.json"
PROGRESS_FILE = r"D:\Desktop\eval_progress.json"
FAILED_FILE = r"D:\Desktop\eval_failed_items.json"  # 新增：记录失败项

# API 配置
API_KEY = "sk-JoCte1XbRxuPD7dJBn846VdDS5cVAypKJYF2BONEbkaIglEu"
API_URL = "https://api1.wangwangyou.cn/v1/chat/completions"
MODEL_NAME = "[A渠道][1额度/次][流式抗截断]gemini-2.5-flash"

# 优化后的速率控制配置
REQUEST_INTERVAL = 5.0      # 请求间隔增加到5秒，减少限流风险
MAX_RETRIES = 100            # 最大重试次数增加到10次
RETRY_BACKOFF = 1.5         # 指数退避倍数
TIMEOUT = 120               # 超时时间增加到120秒，应对慢响应
# ============================================

def evaluate_structural_precision(tcr_seq, pep_seq, true_pep_contacts, true_tcr_contacts, reasoning, max_retries=MAX_RETRIES):
    """
    带重试机制的API调用函数
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"""You are a Structural Biologist evaluating an AI's mechanistic reasoning against true 3D PDB crystal structure data.

**Sequences**:
- TCR CDR3: {tcr_seq}
- Peptide: {pep_seq}

**True 3D Contacts (from PDB < 5 Angstroms)**:
- True Peptide Contacts: {true_pep_contacts}
- True TCR Contacts: {true_tcr_contacts}

**AI Generated Reasoning**:
{reasoning}

**Evaluation Task & CRITICAL RULES**:
1. **Extract Claims**: Extract the specific amino acids the AI claims are involved in the interface/binding (e.g., "TCR W3", "Peptide Y5").
2. **MATCHING RULE (VERY IMPORTANT)**: The "True 3D Contacts" use PDB numbering (e.g., TRP105, GLU96), which typically starts around 90 for TCR CDR3 loops. The AI uses 1-based string indexing (e.g., W3, E5). 
   👉 **DO NOT COMPARE THE ABSOLUTE NUMBERS!** You MUST match them by Amino Acid TYPE (e.g., W = TRP, E = GLU, M = MET, A = ALA) and their relative presence in the sequence. 
   *Example*: If AI claims "M7, A8" and PDB has "MET98, ALA99", this is a 100% PERFECT MATCH.
3. **Calculate Precision**: 
   Precision = (Number of claimed residues that are actually in the True Contacts list) / (Total number of specific residues claimed by AI).
   *Note*: Ignore vague claims like "the central region". Only count specific residues mentioned (e.g., "R6", "Tyr").

**Output Format (Strict JSON)**:
{{
  "Claimed_Residues": "List the residues the AI claimed (e.g., TCR: M7, A8; Pep: V4, F5)",
  "Matched_Residues": "List the successful matches (e.g., TCR: M7 matches MET98; Pep: V4 matches VAL4)",
  "Precision_Score": 1.0 // Float between 0.0 and 1.0. Example: 4 matches out of 5 claims = 0.8
}}
"""
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }
    
    # 重试循环
    for attempt in range(max_retries):
        try:
            response = requests.post(
                API_URL, 
                headers=headers, 
                json=payload, 
                timeout=TIMEOUT  # 使用更长的超时时间
            )
            
            # 处理429限流
            if response.status_code == 429:
                wait_time = (RETRY_BACKOFF ** attempt) * 2  # 限流时等待更久
                print(f"\n⚠️  Rate limit (429). Waiting {wait_time:.1f}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            
            # 处理5xx服务器错误
            if response.status_code >= 500:
                wait_time = RETRY_BACKOFF ** attempt
                print(f"\n⚠️  Server error ({response.status_code}). Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            res_json = response.json()
            content = res_json['choices'][0]['message']['content'].strip()
            content = re.sub(r'^```(?:json)?\s*|\s*```$', '', content, flags=re.MULTILINE).strip()
            return json.loads(content)
            
        except requests.exceptions.Timeout:
            wait_time = RETRY_BACKOFF ** attempt
            print(f"\n⏱️  Timeout (attempt {attempt+1}/{max_retries}). Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)
            continue
        except requests.exceptions.ConnectionError:
            wait_time = RETRY_BACKOFF ** attempt
            print(f"\n🔌  Connection error. Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)
            continue
        except requests.exceptions.RequestException as e:
            wait_time = RETRY_BACKOFF ** attempt
            print(f"\n⚠️  Request error: {type(e).__name__}. Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)
            continue
        except json.JSONDecodeError as e:
            print(f"\n❌  JSON decode error: {e}")
            return None
        except Exception as e:
            print(f"\n❌  Unexpected error: {type(e).__name__}: {e}")
            return None
    
    # 所有重试都失败
    print(f"\n❌  Failed after {max_retries} retries")
    return None

def save_progress(processed_ids, results, failed_items):
    """保存进度、结果和失败项"""
    data = {
        "processed_ids": list(processed_ids),
        "results": results,
        "failed_items": failed_items,
        "timestamp": datetime.now().isoformat()
    }
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_progress():
    """加载进度"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return set(data.get("processed_ids", [])), data.get("results", []), data.get("failed_items", [])
        except:
            pass
    return set(), [], []

def save_failed_items(failed_items):
    """单独保存失败项，方便后续重跑"""
    with open(FAILED_FILE, 'w', encoding='utf-8') as f:
        json.dump(failed_items, f, ensure_ascii=False, indent=2)

def generate_unique_id(idx, item):
    """生成唯一标识符"""
    tcr = item.get('tcr_seq', item.get('receptor_seq', item.get('hla_seq', '')))
    pep = item.get('peptide_seq', item.get('ligand_seq', ''))
    return f"{idx}_{tcr}_{pep}"

def main():
    print("1. Loading PDB Contact Data...")
    df_pdb = pd.read_csv(CSV_FILE_PATH)
    
    pdb_dict = {}
    for _, row in df_pdb.iterrows():
        tcr = row.get('tcr', row.get('cdr3_seq', '')).strip()
        pep = row.get('peptide', row.get('pep_seq', '')).strip()
        key = f"{tcr}_{pep}"
        pdb_dict[key] = {
            "pdbid": row['pdbid'],
            "pep_contacts": row['pep_contacts'],
            "tcr_contacts": row['tcr_contacts']
        }
    print(f"   Loaded {len(pdb_dict)} PDB reference structures.")
    
    print("2. Parsing JSONL Predictions...")
    eval_targets = []
    with open(JSONL_FILE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            if data.get('ground_truth') == 1 and data.get('prediction') == 1:
                tcr_seq = data.get('tcr_seq', data.get('receptor_seq', data.get('hla_seq', '')))
                pep_seq = data.get('peptide_seq', data.get('ligand_seq', ''))
                key = f"{tcr_seq}_{pep_seq}"
                if key in pdb_dict:
                    data['pdb_info'] = pdb_dict[key]
                    eval_targets.append(data)
    print(f"   Found {len(eval_targets)} True Positives matched with PDB data.")
    
    if len(eval_targets) == 0:
        print("No matches found. Please check sequence column names and formats.")
        return
    
    # 加载进度
    print("3. Loading progress...")
    processed_ids, results, failed_items = load_progress()
    print(f"   Previously processed: {len(processed_ids)}, Failed items: {len(failed_items)}")
    
    # 过滤未处理的条目
    remaining_targets = []
    for idx, item in enumerate(eval_targets):
        uid = generate_unique_id(idx, item)
        if uid not in processed_ids:
            remaining_targets.append((idx, item))
    
    print(f"   Items to process: {len(remaining_targets)}")
    
    if len(remaining_targets) == 0 and len(failed_items) == 0:
        print("✅ All items already processed!")
        # 计算并输出结果
        precision_scores = [r['Structural_Evaluation']['Precision_Score'] for r in results 
                          if 'Structural_Evaluation' in r and 'Precision_Score' in r['Structural_Evaluation']]
        if precision_scores:
            avg_precision = sum(precision_scores) / len(precision_scores) * 100
            print(f"\n🏆 Average Structural Precision: {avg_precision:.1f}%")
        return
    
    print("4. Running GPT-4 Structural Evaluation...")
    precision_scores = []
    
    for orig_idx, item in tqdm(remaining_targets, desc="Evaluating"):
        pdb_info = item['pdb_info']
        reasoning = item['reasoning_content']
        
        eval_res = evaluate_structural_precision(
            tcr_seq=item.get('tcr_seq', item.get('receptor_seq', item.get('hla_seq', ''))),
            pep_seq=item.get('peptide_seq', item.get('ligand_seq', '')),
            true_pep_contacts=pdb_info['pep_contacts'],
            true_tcr_contacts=pdb_info['tcr_contacts'],
            reasoning=reasoning
        )
        
        uid = generate_unique_id(orig_idx, item)
        
        if eval_res and "Precision_Score" in eval_res:
            item['Structural_Evaluation'] = eval_res
            precision_scores.append(float(eval_res['Precision_Score']))
            results.append(item)
            processed_ids.add(uid)
            print(f"  ✓ Success: {uid[:50]}...")
        else:
            # 记录失败项，包含完整信息以便后续重试
            failed_items.append({
                "unique_id": uid,
                "original_index": orig_idx,
                "item": item,
                "error_time": datetime.now().isoformat()
            })
            print(f"  ✗ Failed: {uid[:50]}... (saved to failed list)")
        
        # 每次请求后保存进度 + 失败项
        save_progress(processed_ids, results, failed_items)
        save_failed_items(failed_items)
        
        # 请求间隔
        time.sleep(REQUEST_INTERVAL)
    
    # 输出统计
    print("\n" + "="*60)
    print("📊 Evaluation Summary")
    print("="*60)
    print(f"Total items: {len(eval_targets)}")
    print(f"Successfully evaluated: {len(results)}")
    print(f"Failed items: {len(failed_items)}")
    
    if precision_scores:
        avg_precision = sum(precision_scores) / len(precision_scores) * 100
        print(f"Average Structural Precision: {avg_precision:.1f}%")
    
    if failed_items:
        print(f"\n⚠️  {len(failed_items)} items failed. You can re-run this script to retry them.")
        print(f"   Failed items saved to: {FAILED_FILE}")
    
    print("="*60)
    
    # 保存最终结果
    with open(OUTPUT_EVAL_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Results saved to {OUTPUT_EVAL_FILE}")

if __name__ == "__main__":
    main()

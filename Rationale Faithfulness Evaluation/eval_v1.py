import json
import random
import requests
import re
from tqdm import tqdm

# ================= 配置 =================
# INPUT_RESULTS_FILE = r"D:\Desktop\temp.jsonl"
INPUT_RESULTS_FILE = r"D:\Desktop\immunellm_test_results_300.jsonl"
API_KEY = "sk-JoCte1XbRxuPD7dJBn846VdDS5cVAypKJYF2BONEbkaIglEu"
API_URL = "https://api1.wangwangyou.cn/v1/chat/completions"
# ========================================

def evaluate_with_gpt4(hla_seq, pep_seq, reasoning):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 采用 NLP 顶会标准的 1-5 分评价 Prompt，加入了容错机制和裁判思维链
    prompt = f"""You are an expert Structural Biologist evaluating an AI-generated reasoning chain for Receptor-Peptide binding.

**Ground Truth Data**:
- Receptor Sequence: {hla_seq}
- Peptide Sequence: {pep_seq}

**AI Generated Rationale**:
{reasoning}

**Evaluation Task**:
Please evaluate the rationale on two metrics using a 1 to 5 scale. 

**Metric 1: Factuality (1-5)**
- Focus on the *key amino acids* mentioned. 
- *Note*: AI models often use different indexing systems (e.g., 0-based vs 1-based, or pseudo-sequence offsets). Do NOT penalize strictly for minor numbering shifts if the amino acid and its relative context are correct.
- 5: Perfect factuality. All referenced residues exist in the sequences.
- 4: Minor flaws (e.g., slight numbering offset) but key residues are correct.
- 3: Moderate flaws; some hallucinated residues, but the core anchors are correct.
- 2: Major hallucinations; critical anchors are fabricated.
- 1: Completely fabricated sequence data.

**Metric 2: Biophysical Soundness (1-5)**
- Evaluate the scientific logic (e.g., electrostatic complementarity, steric clash, hydrophobic shielding).
- 5: The biophysical logic is perfectly sound and supports the conclusion.
- 4: The logic is mostly correct with minor, non-critical biological inaccuracies.
- 3: Partially correct logic, but oversimplifies or misses a key biophysical interaction.
- 2: Flawed logic (e.g., claiming two positive charges attract).
- 1: Completely nonsensical biophysics.

**Output Format**:
You MUST return ONLY a JSON object. Provide a brief justification first, then the scores.
{{
  "Evaluation_Reasoning": "Briefly explain your scores here...",
  "Factuality_Score": 5,
  "Biophysical_Soundness_Score": 4
}}
"""
    
    payload = {
        "model": "gpt-5.4", # 你的模型名      gpt-5.4
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status() 
        res_json = response.json()
        
        if 'choices' not in res_json or len(res_json['choices']) == 0:
            return None
            
        content = res_json['choices'][0]['message']['content'].strip()
        content = re.sub(r'^```(?:json)?\s*|\s*```$', '', content, flags=re.MULTILINE).strip()
        
        return json.loads(content)
        
    except Exception as e:
        # 隐藏长串的报错，防止刷屏
        return None

def main():
    true_positives = []
    true_negatives =[]
    
    with open(INPUT_RESULTS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            gt = data.get('ground_truth')
            pred = data.get('prediction')
            if gt == 1 and pred == 1:
                true_positives.append(data)
            elif gt == 0 and pred == 0:
                true_negatives.append(data)

    random.seed(42)
    # 取样 (比如测试阶段可以先设小一点，跑10个试试)
    sample_tp = random.sample(true_positives, min(150, len(true_positives)))
    sample_tn = random.sample(true_negatives, min(150, len(true_negatives)))
    eval_samples = true_positives + true_negatives
    random.shuffle(eval_samples)
    print(f"Evaluating {len(eval_samples)} samples...")

    results =[]
    factuality_scores = []
    soundness_scores =[]
    
    for item in tqdm(eval_samples):
        scores = evaluate_with_gpt4(item.get('hla_seq', item.get('receptor_seq', '')), item['peptide_seq'], item['reasoning_content'])
        if scores and "Factuality_Score" in scores:
            f_score = float(scores["Factuality_Score"])
            s_score = float(scores["Biophysical_Soundness_Score"])
            
            factuality_scores.append(f_score)
            soundness_scores.append(s_score)
            
            item['Eval_Factuality'] = f_score
            item['Eval_Soundness'] = s_score
            item['Eval_Reasoning'] = scores.get("Evaluation_Reasoning", "")
            results.append(item)

    if len(factuality_scores) > 0:
        avg_factuality = sum(factuality_scores) / len(factuality_scores)
        avg_soundness = sum(soundness_scores) / len(soundness_scores)
        
        # 将 1-5 分制转换为百分制 (e.g., 4.5/5 -> 90%)
        # 公式: (Score / 5.0) * 100
        perc_factuality = (avg_factuality / 5.0) * 100
        perc_soundness = (avg_soundness / 5.0) * 100
    else:
        avg_factuality = avg_soundness = 0.0
        perc_factuality = perc_soundness = 0.0

    print("\n=== gpt-5.4 Evaluation Results (1-5 Scale) ===")
    print(f"Factuality Score:        {avg_factuality:.2f} / 5.0 ({perc_factuality:.1f}%)")
    print(f"Biophysical Soundness:   {avg_soundness:.2f} / 5.0 ({perc_soundness:.1f}%)")
    print(f"Successfully evaluated:  {len(results)}/{len(eval_samples)}")

    with open("gpt5_eval_results_300.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()

    """ [A渠道][1额度/次][流式抗截断]gemini-2.5-flash
    === GPT-4o Evaluation Results (1-5 Scale) ===
    Factuality Score:        3.05 / 5.0 (61.1%)
    Biophysical Soundness:   4.07 / 5.0 (81.4%)
    Successfully evaluated:  296/300
    """
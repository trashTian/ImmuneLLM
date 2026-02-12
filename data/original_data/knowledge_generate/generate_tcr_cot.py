import requests
import json
import re
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# =================配置区域=================
# API 设置
API_URL = "http://172.16.64.255:8009/v1/chat/completions"
MODEL_NAME = "qwen"
MAX_WORKERS = 10     # 并发数
BATCH_SIZE = 500     # 每处理多少条写一次盘

# 文件路径 (TCR 专用)
# 文件路径
# INPUT_JSONL = "/mnt/lustre/guopeijin/Immune_LLM/code/data_prepare/knowledge_generate/hla_i_train_integrated.jsonl"
# OUTPUT_SFT_JSONL = "hla_i_train_integrated_sft_final.jsonl"

INPUT_JSONL = "/mnt/lustre/guopeijin/Immune_LLM/code/data_prepare/knowledge_generate/hla_i_val_integrated.jsonl"
OUTPUT_SFT_JSONL = "hla_i_val_integrated_sft_final.jsonl"



# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("tcr_processing.log", encoding='utf-8')
    ]
)
# ==========================================

def call_llm_api(messages, retries=3):
    """发送请求给本地 LLM API"""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.6, # 保持随机性以便重试
        "stream": False,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5
    }

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=120)
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logging.warning(f"API Error {response.status_code}: {response.text}")
        except Exception as e:
            logging.warning(f"Request failed (Attempt {attempt+1}/{retries}): {e}")
        
        time.sleep(1)
    
    return None

def parse_prediction(content):
    """解析 [[Yes]] 或 [[No]]"""
    if not content: return -1
    matches = list(re.finditer(r"\[\[(Yes|No)\]\]", content, re.IGNORECASE))
    if matches:
        last_match = matches[-1]
        pred_text = last_match.group(1).lower()
        return 1 if pred_text == "yes" else 0
    return -1

def extract_thought_and_answer(full_text):
    """
    分离思考过程和最终回答。
    兼容 </think> 分隔符
    """
    if not full_text: return "", ""
    separator = "</think>"
    if separator in full_text:
        parts = full_text.split(separator)
        thought = parts[0].strip().replace("<think>", "").strip()
        answer = parts[1].strip() if len(parts) > 1 else ""
        return thought, answer
    else:
        # 如果没有分隔符，尝试智能分离或者全量返回
        return "", full_text

# ==========================================
# 核心业务逻辑 (单个任务的处理)
# ==========================================
def process_single_record(record):
    """
    处理单条数据：Blind 尝试 -> 成功则返回 -> 失败则尝试 Hindsight
    """
    try:
        custom_id = record['custom_id']
        ground_truth = record['ground_truth']
        blind_msgs = record['prompt_blind']
        
        final_output = None
        source_type = ""
        
        # === Step 1: Blind Prompt Loop (Max 3 attempts) ===
        max_blind_attempts = 3
        for i in range(max_blind_attempts):
            response_content = call_llm_api(blind_msgs)
            if response_content is None: continue 
                
            prediction = parse_prediction(response_content)
            # 只要预测正确，立即采纳
            if prediction == ground_truth:
                final_output = response_content
                source_type = "blind_correct"
                break

        # === Step 2: Hindsight Strategy (Fallback) ===
        # 如果 Blind 全失败，调用 Hindsight
        if final_output is None:
            hindsight_msgs = record['prompt_hindsight']
            response_hindsight = call_llm_api(hindsight_msgs)
            
            if response_hindsight:
                final_output = response_hindsight
                source_type = "hindsight_repair"
            else:
                return None # 彻底失败，放弃数据
            
        # === Step 3: Construct SFT Output (移花接木) ===
        # Input 永远是 Blind Prompt 的 User Content (不带答案)
        # 假设 messages 结构是 [System, User]，取 User 的 content
        user_input_text = blind_msgs[-1]['content']
        
        # 分离 Reasoning 和 Answer
        reasoning, answer = extract_thought_and_answer(final_output)
        
        return {
            "custom_id": custom_id,
            "source": source_type,
            # [关键修改] Instruction 针对 TCR 任务
            "instruction": "You are an expert Structural Biologist specialized in TCR-Peptide binding prediction.",
            "input": user_input_text,
            "reasoning_content": reasoning,
            "content": answer,
            "raw_output": final_output
        }
        
    except Exception as e:
        logging.error(f"Error processing ID {record.get('custom_id', 'unknown')}: {e}")
        return None

# ==========================================
# 批量处理引擎 (消费者)
# ==========================================
def process_batch_concurrently(batch_data, output_file, batch_idx):
    """并发处理一个Batch的数据并写入文件"""
    if not batch_data:
        return {"blind": 0, "repair": 0, "failed": 0}

    batch_stats = {"blind": 0, "repair": 0, "failed": 0}
    
    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)

    with open(output_file, "a+", encoding='utf-8') as fw:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_id = {executor.submit(process_single_record, item): item['custom_id'] for item in batch_data}
            
            pbar = tqdm(as_completed(future_to_id), total=len(batch_data), desc=f"Batch {batch_idx}", leave=False)
            
            for future in pbar:
                try:
                    result = future.result()
                    if result:
                        fw.write(json.dumps(result, ensure_ascii=False) + "\n")
                        
                        if result['source'] == 'blind_correct':
                            batch_stats['blind'] += 1
                        elif result['source'] == 'hindsight_repair':
                            batch_stats['repair'] += 1
                    else:
                        batch_stats['failed'] += 1
                except Exception as e:
                    logging.error(f"Critical Worker Error: {e}")
                    batch_stats['failed'] += 1
    
    return batch_stats

# ==========================================
# 主程序 (生产者)
# ==========================================
def main():
    # 1. 检查输入文件
    if not os.path.exists(INPUT_JSONL):
        logging.error(f"Input file not found: {INPUT_JSONL}")
        logging.error("Please run the 'Step 1' prompt generation script first!")
        return

    # 2. 加载断点续传记录
    processed_ids = set()
    if os.path.exists(OUTPUT_SFT_JSONL):
        logging.info("Checking processed records...")
        with open(OUTPUT_SFT_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data['custom_id'])
                except: pass
    logging.info(f"Found {len(processed_ids)} processed records. Resuming...")

    # 3. 数据生成器 (Lazy Loading)
    def input_record_generator():
        with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    record = json.loads(line)
                    if record['custom_id'] not in processed_ids:
                        yield record
                except Exception as e:
                    logging.error(f"JSON Parse Error: {e}")

    # 4. 循环构建 Batch
    current_batch = []
    batch_count = 0
    total_stats = {"blind": 0, "repair": 0, "failed": 0}

    record_iter = input_record_generator()
    
    for record in record_iter:
        current_batch.append(record)
        
        if len(current_batch) >= BATCH_SIZE:
            batch_count += 1
            stats = process_batch_concurrently(current_batch, OUTPUT_SFT_JSONL, batch_count)
            
            for k in total_stats: total_stats[k] += stats.get(k, 0)
            logging.info(f"Batch {batch_count} Done. Stats: {stats}")
            
            current_batch = []

    # 5. 处理剩余数据
    if current_batch:
        batch_count += 1
        stats = process_batch_concurrently(current_batch, OUTPUT_SFT_JSONL, batch_count)
        for k in total_stats: total_stats[k] += stats.get(k, 0)
        logging.info(f"Final Batch {batch_count} Done.")

    logging.info("="*30)
    logging.info("TCR Data Generation Completed!")
    logging.info(f"Total Stats: Blind Correct: {total_stats['blind']}, Repaired: {total_stats['repair']}, Failed: {total_stats['failed']}")

if __name__ == "__main__":
    main()
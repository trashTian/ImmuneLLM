import requests
import json
import re
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# =================配置区域=================
# API 设置
API_URL = "http://172.16.64.255:8009/v1/chat/completions"
MODEL_NAME = "qwen" # 你的模型名称
MAX_WORKERS = 5  # 并发线程数，根据显存和API承受能力调整

# 文件路径 (请修改这里)
INPUT_JSONL = "/mnt/lustre/guopeijin/Immune_LLM/code/data_prepare/knowledge_generate/temp_.jsonl"
OUTPUT_SFT_JSONL = "temp_sft_final.jsonl"
# ==========================================

def call_llm_api(messages, retries=3):
    """发送请求给本地 LLM API"""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.6,
        "stream": False,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5
    }

    # print('原始的问题：')
    # print(messages)
    # print('\n\n\n模型回答：')

    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=120)
            if response.status_code == 200:
                result = response.json()
                # print(result["choices"][0]["message"]["content"])
                return result["choices"][0]["message"]["content"]
            else:
                # 打印错误但不立即退出，尝试重试
                print(f"API Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Request failed: {e}")
        
        time.sleep(1) # 重试等待
    
    return None # 最终失败

def parse_prediction(content):
    """解析 [[Yes]] 或 [[No]]"""
    if not content: return -1
    
    # 使用正则寻找最后出现的 [[Yes]] 或 [[No]]
    matches = list(re.finditer(r"\[\[(Yes|No)\]\]", content, re.IGNORECASE))
    
    if matches:
        last_match = matches[-1] # 取最后一个，防止CoT中间提到
        pred_text = last_match.group(1).lower()
        return 1 if pred_text == "yes" else 0
    
    return -1 # 解析失败


def extract_thought_and_answer(full_text):
    """
    分离思考过程和最终回答。
    依据用户设定：分隔符为 </think>
    """
    if not full_text:
        return "", ""
        
    separator = "</think>"
    
    if separator in full_text:
        parts = full_text.split(separator)
        
        # 提取 reasoning (通常在分隔符前面)
        thought = parts[0].strip()
        # 为了数据干净，尝试去掉开头的 <think> 标签（如果有的话）
        thought = thought.replace("<think>", "").strip()
        
        # 提取 content (通常在分隔符后面)
        if len(parts) > 1:
            answer = parts[1].strip()
        else:
            answer = "" # 只有思考没有回答的情况
            
        return thought, answer
    else:
        # 如果没找到分隔符，这通常是不正常的
        # 策略：如果包含 [[Yes/No]]，则视为回答，否则视情况而定
        # 这里简单处理：全部当作 answer，thought 为空
        return "", full_text

def process_single_record(record):
    """
    处理逻辑更新：
    1. 尝试 Blind Prompt 最多 3 次。
    2. 只要有一次 Blind Prompt 预测正确，就采用该结果并停止。
    3. 如果 3 次全错（或解析失败），才使用 Hindsight Prompt。
    """
    custom_id = record['custom_id']
    ground_truth = record['ground_truth'] # 1 or 0
    blind_msgs = record['prompt_blind']
    
    final_output = None
    source_type = ""
    
    # === Step 1: Blind Prompt Retry Loop (Max 3 times) ===
    max_blind_attempts = 3
    
    for i in range(max_blind_attempts):
        response_content = call_llm_api(blind_msgs)
        
        if response_content is None:
            continue # API 调用失败，继续下一次尝试
            
        prediction = parse_prediction(response_content)
        
        # 核心判断：如果预测正确，立即采纳并跳出循环
        if prediction == ground_truth:
            final_output = response_content
            source_type = "blind_correct"
            # print(f"ID {custom_id}: Blind correct on attempt {i+1}")
            break
        else:
            pass
            # print(f"ID {custom_id}: Blind attempt {i+1} incorrect (Pred:{prediction} vs GT:{ground_truth})")

    # === Step 2: Hindsight Strategy (Fallback) ===
    # 如果上面的循环跑完了，final_output 还是 None，说明3次Blind都失败了
    if final_output is None:
        # print(f"ID {custom_id}: All 3 blind attempts failed. Switching to Hindsight.")
        hindsight_msgs = record['prompt_hindsight']
        response_hindsight = call_llm_api(hindsight_msgs)
        
        if response_hindsight:
            # Hindsight 默认为正确逻辑
            final_output = response_hindsight
            source_type = "hindsight_repair"
        else:
            return None # Hindsight 也挂了，放弃这条数据
        

    # === Step 3: 构造 SFT 数据 (分离版) ===
    # 3.1 提取 Input (永远是 Blind 的提问，不带答案)
    # 假设 messages 结构是 [System, User]，取 User 的 content
    user_input_text = blind_msgs[-1]['content']
    
    # 3.2 分离 Reasoning 和 Answer
    reasoning, answer = extract_thought_and_answer(final_output)
    
    # 3.3 构建新的 JSON 结构
    # 这种结构兼容目前主流的 DeepSeek-R1 蒸馏格式或 LLaMA-Factory 的复杂格式
    sft_record = {
        "custom_id": custom_id,
        "source": source_type,
        "instruction": "You are an expert Structural Biologist specialized in HLA-Peptide binding prediction.",
        "input": user_input_text,
        "reasoning_content": reasoning, # 思考过程
        "content": answer,              # 最终结论 (包含 [[Yes]]/[[No]])
        "raw_output": final_output      # 保留原始输出以备查验
    }
    
    return sft_record

def main():
    # 1. 读取输入文件
    print(f"Loading {INPUT_JSONL}...")
    records = []
    try:
        with open(INPUT_JSONL, 'r') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    except FileNotFoundError:
        print(f"File not found: {INPUT_JSONL}")
        return

    # 2. 断点续传检测
    processed_ids = set()
    if os.path.exists(OUTPUT_SFT_JSONL):
        with open(OUTPUT_SFT_JSONL, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data['custom_id'])
                except: pass
    
    tasks = [r for r in records if r['custom_id'] not in processed_ids]
    print(f"Total: {len(records)}, Processed: {len(processed_ids)}, Remaining: {len(tasks)}")

    # 3. 并发处理
    sft_file = open(OUTPUT_SFT_JSONL, 'a', encoding='utf-8')
    
    # 统计器
    stats = {"blind_correct": 0, "hindsight_repair": 0, "failed": 0}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任务
        future_to_record = {executor.submit(process_single_record, r): r for r in tasks}
        
        # 进度条
        for future in tqdm(as_completed(future_to_record), total=len(tasks), desc="Generating CoT"):
            result = future.result()
            
            if result:
                sft_file.write(json.dumps(result) + "\n")
                sft_file.flush() # 确保写入磁盘
                stats[result['source']] += 1
            else:
                stats["failed"] += 1

    sft_file.close()
    print("\nGeneration Completed!")
    print(f"Stats: Blind Correct: {stats['blind_correct']}, Repaired: {stats['hindsight_repair']}, Failed: {stats['failed']}")

if __name__ == "__main__":
    main()
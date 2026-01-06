import os
import sys
import torch
import numpy as np
import pandas as pd
import argparse 
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef,precision_recall_curve, auc)  
from datetime import datetime  
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import esm
import types
from torch import nn
from FT_utils import Logger, TCRBindingDataset, custom_collate_fn, steering_forward,generate_steering_vector


def load_esm_model(model_name="esm2_t36_3B_UR50D"):
    """使用esm库加载ESM2模型"""
    print(f"\nLoading ESM2 model with esm library: {model_name}")
    
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    batch_converter = alphabet.get_batch_converter()
    # print(model.layers)
    # 冻结参数，仅解冻最后两层
    for param in model.parameters():
        param.requires_grad = False
        # 解冻最后两层，并打印解冻信息
    unfreeze_start = 20
    unfreeze_end = 36
    print(f"\n🔓 Unfreezing layers [{unfreeze_start} to {unfreeze_end - 1}] (total: {unfreeze_end - unfreeze_start} layers):")

    for layer_idx in range(unfreeze_start, unfreeze_end):
        for param in model.layers[layer_idx].parameters():
            param.requires_grad = True

    # 绑定激活引导前向传播方法
    model.steering_forward = types.MethodType(steering_forward, model)
    
    return model, alphabet, batch_converter

class ESM2ForBindingPrediction(torch.nn.Module):
    """基于esm库的预测模型"""
    def __init__(self, esm_model, steering_vectors=None):
        super().__init__()
        self.esm = esm_model
        self.hidden_size = esm_model.embed_dim  # 3B模型为2560
        self.steering_vectors = steering_vectors

        self.prediction_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, 2560), 
            nn.LayerNorm(2560), 
            nn.ReLU(),
            nn.Linear(2560, 256), 
            nn.LayerNorm(256), 
            nn.ReLU(),
            nn.Linear(256, 1)
            )
        
        self._init_prediction_head()
    
    def _init_prediction_head(self):
        for module in self.prediction_head:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, tokens):
        if self.steering_vectors is not None:
            outputs = self.esm.steering_forward(
                tokens=tokens,
                repr_layers=[36],  # esm2_t36共36层，最后一层索引36
                steering_vectors=self.steering_vectors
            )
            last_hidden_state = outputs['representations'][36]
        else:
            outputs = self.esm(
                tokens,
                repr_layers=[36],
                return_contacts=False
            )
            last_hidden_state = outputs['representations'][36]
        
        # 提取[CLS]特征（esm的第一个token是<cls>）
        features = last_hidden_state[:, 0, :]
        predictions = self.prediction_head(features).squeeze(-1)
        return predictions, features


def train_epoch(model, dataloader, optimizer, scaler, device, batch_converter):
    """训练epoch：修改为二分类任务，删除梯度累计"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    criterion = torch.nn.BCEWithLogitsLoss()  # 二分类损失函数
    
    for batch_idx, (samples, labels) in enumerate(tqdm(dataloader, desc="Training Batch")):        
        optimizer.zero_grad()  # 每个batch都重置梯度
        
        # 转换为esm输入格式
        try:
            _, _, tokens = batch_converter(samples)
        except Exception as e:
            print(f"Error: batch_converter处理失败，当前samples={samples[:1]}")
            raise e
        
        tokens = tokens.to(device, non_blocking=True)
        labels = torch.tensor(labels, dtype=torch.float32).to(device, non_blocking=True)
        
        # 混合精度训练
        with torch.cuda.amp.autocast():
            logits, _ = model(tokens)
            loss = criterion(logits, labels)  # 不再除以累计步数
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 每个batch都更新参数
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # 记录指标
        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()  # 转换为概率
        preds = (probs > 0.5).astype(int)  # 转换为类别预测
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    # 计算基础指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)  # 即sensitivity
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    
    # 计算新增指标
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    # 计算PR-AUC
    if len(np.unique(all_labels)) > 1:
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
        pr_auc = auc(recall_curve, precision_curve)
    else:
        pr_auc = 0.0
    
    # 计算specificity
    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    else:
        specificity = 0.0  # 处理单类情况

    # 返回所有指标
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'sensitivity': recall,  # 敏感性即召回率
        'specificity': specificity,
        'f1': f1,
        'auc': roc_auc,
        'pr_auc': pr_auc,
        'mcc': mcc
    }


def evaluate(model, dataloader, device, batch_converter, dataset_name="Validation"):
    """评估函数：修改为二分类任务，添加所有指标"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for samples, labels in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            _, _, tokens = batch_converter(samples)
            tokens = tokens.to(device, non_blocking=True)
            labels = torch.tensor(labels, dtype=torch.float32).to(device, non_blocking=True)
            
            logits, _ = model(tokens)
            probs = torch.sigmoid(logits).cpu().numpy()  # 转换为概率
            preds = (probs > 0.5).astype(int)  # 转换为类别预测
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # 计算基础指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)  # 即sensitivity
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0
    cm = confusion_matrix(all_labels, all_preds)
    
    # 计算新增指标
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    # 计算PR-AUC
    if len(np.unique(all_labels)) > 1:
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
        pr_auc = auc(recall_curve, precision_curve)
    else:
        pr_auc = 0.0
    
    # 计算specificity
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    else:
        specificity = 0.0  # 处理单类情况

    # 返回所有指标
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'sensitivity': recall,  # 敏感性即召回率
        'specificity': specificity,
        'f1': f1,
        'auc': roc_auc,
        'pr_auc': pr_auc,
        'mcc': mcc,
        'confusion_matrix': cm.tolist()
    }, all_preds, all_probs, all_labels


def parse_args():
    """解析命令行参数：删除梯度累计相关参数"""
    parser = argparse.ArgumentParser(description='ESM2模型用于TCR结合预测')
    
    # 模型参数
    parser.add_argument('--model-name', type=str, default='esm2_t36_3B_UR50D',
                      help='使用的ESM模型名称')
    
    # 数据文件路径
    parser.add_argument('--train-file', type=str, 
                      default='/mnt/lustre/guopeijin/data/single_air/AIR_dataset/AIR-binding-specificity/TCR/epitope_LTDEMIAQY_train.csv',
                      help='训练数据文件路径')
    parser.add_argument('--val-file', type=str,
                      default='/mnt/lustre/guopeijin/data/single_air/AIR_dataset/AIR-binding-specificity/TCR/epitope_LTDEMIAQY_val.csv',
                      help='验证数据文件路径')
    parser.add_argument('--test-file', type=str,
                      default='/mnt/lustre/guopeijin/data/single_air/AIR_dataset/AIR-binding-specificity/TCR/epitope_LTDEMIAQY_test.csv',
                      help='测试数据文件路径')
    parser.add_argument('--output-dir', type=str,
                      default='/mnt/lustre/guopeijin/Immune_LLM/code/baselines/logs',
                      help='输出结果保存目录')
    parser.add_argument('--col1', type=str, default='tcr')
    parser.add_argument('--col2', type=str, default='pep')
    parser.add_argument('--col3', type=str, default='label')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=256,
                      help='批次大小')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                      help='学习率')
    parser.add_argument('--num-epochs', type=int, default=5,
                      help='训练轮数')
    parser.add_argument('--max-seq-length', type=int, default=256,
                      help='最大序列长度')
     
    # 激活引导参数
    parser.add_argument('--use-steering',  default=False,
                      help='是否使用激活引导')
    parser.add_argument('--steering-vec-path', type=str,
                      default='/mnt/lustre/guopeijin/data/single_air/steering_vectors/bs_tcr/3B_specificity_LTDEMIAQY_steering_vectors.pt',
                      help='激活引导向量文件路径')
    parser.add_argument('--alpha', type=float, default=1,
                      help='激活引导向量的缩放因子')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda:6',
                      help='使用的设备，如"cuda:0"或"cpu"')
    
    parser.add_argument('--log_mark', type=str, default="LTDEMIAQY", help='训练日志文件名标志')
    
    return parser.parse_args()


def evaluate_repeatedly(model_save_path, test_dataloader, device, batch_converter, output_dir, repeat=10):
    """
    从本地加载模型，在测试集上重复评估 repeat 次，保存所有指标
    """
    print(f"\n🔄 Loading model from {model_save_path} for repeated evaluation...")
    checkpoint = torch.load(model_save_path, map_location=device)
    config = checkpoint['config']
    
    # 重新加载 ESM 模型（保持冻结设置一致）
    model, alphabet, batch_converter = load_esm_model(config["model_name"])
    prediction_model = ESM2ForBindingPrediction(esm_model=model).to(device)
    prediction_model.load_state_dict(checkpoint['model_state_dict'])
    prediction_model.eval()  # 固定为 eval 模式

    all_runs_metrics = []
    for i in range(repeat):
        print(f"\n🔁 Run {i+1}/{repeat}")
        metrics, _, _, _ = evaluate(
            model=prediction_model,
            dataloader=test_dataloader,
            device=device,
            batch_converter=batch_converter,
            dataset_name="Test (Repeated)"
        )
        metrics["run"] = i + 1
        all_runs_metrics.append(metrics)
        print(f"Run {i+1} - AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}, MCC: {metrics['mcc']:.4f}")

    # 转为 DataFrame 并保存
    df_metrics = pd.DataFrame(all_runs_metrics)
    metrics_save_path = os.path.join(output_dir, "test_metrics_10_runs.csv")
    df_metrics.to_csv(metrics_save_path, index=False)
    print(f"\n All {repeat} runs metrics saved to: {metrics_save_path}")
    
    # 打印平均值 ± 标准差
    mean_metrics = df_metrics.mean(numeric_only=True)
    std_metrics = df_metrics.std(numeric_only=True)
    print("\n Average Metrics (± Std):")
    for col in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'pr_auc', 'mcc', 'specificity']:
        if col in mean_metrics:
            print(f"{col}: {mean_metrics[col]:.4f} ± {std_metrics[col]:.4f}")



def main():
    # 解析命令行参数
    args = parse_args()
    
    # 从命令行参数构建配置：删除accumulation_steps
    config = {
        "model_name": args.model_name,
        "train_file": args.train_file,
        "val_file": args.val_file,
        "test_file": args.test_file,
        "output_dir": args.output_dir+'/'+args.log_mark,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "max_seq_length": args.max_seq_length,
        "use_steering": args.use_steering,
        "steering_vec_path": args.steering_vec_path,
        "alpha": args.alpha,
        "device": args.device,  # 新增设备配置
        "log_mark" : args.log_mark
    }

    # 创建输出目录
    os.makedirs(config["output_dir"], exist_ok=True)
    print(f"All results will be saved to: {config['output_dir']}")
    # 生成时间戳（格式：年-月-日_时-分-秒，不含特殊字符，适合文件名）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    original_stdout = sys.stdout  # 保存原始stdout
    sys.stdout = Logger(f"{config['output_dir']}/{config['log_mark']}_{timestamp}.log")  # 重定向输出

    try:  
        print("\n" + "="*50)
        print("                 超参数配置                 ")
        print("="*50)
        for key, value in config.items():
            # 格式化输出，左对齐键名，右对齐值，增强可读性
            print(f"{key.ljust(20)}: {str(value).rjust(5)}")
        print("="*50 + "\n")

        # 设备与混合精度
        device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
        scaler = torch.cuda.amp.GradScaler()
        print(f"\nUsing device: {device}, Batch size: {config['batch_size']}")  # 更新打印信息

        # 加载激活引导向量（校验维度）
        steering_vectors = None
        if config["use_steering"]:
            # 检查激活向量是否存在，如果不存在则生成
            if not os.path.exists(config["steering_vec_path"]):
                print(f"Steering vectors not found at {config['steering_vec_path']}")
                print("Generating new steering vectors...")
                
                # 从路径中提取保存目录和名称标记
                save_folder = os.path.dirname(config["steering_vec_path"])
                name_mark = config["log_mark"]
                
                # 生成激活向量
                generated_path = generate_steering_vector(
                    theshold_pos=1,
                    theshold_neg=0,
                    property='specificity',
                    num_data=None,
                    save_folder=save_folder,
                    name_mark=name_mark,
                    data_path=config["train_file"]  # 使用训练数据生成激活向量
                )
                
                print(f"Successfully generated steering vectors at: {generated_path}")
                config["steering_vec_path"] = generated_path  # 更新路径为生成的路径
            
            # 加载激活向量
            print(f"Loading steering vectors from: {config['steering_vec_path']}")
            pos_vecs, neg_vecs = torch.load(config["steering_vec_path"], map_location=device)
            steering_vectors = (pos_vecs - neg_vecs) * config["alpha"]
            
            # 校验引导向量维度（36层模型需36个向量，每个向量维度=2560）
            expected_layers = 36
            expected_dim = 2560  # esm2_t36_3B_UR50D的隐藏层维度
            if steering_vectors.shape[0] != expected_layers:
                raise ValueError(f"引导向量层数不匹配：需{expected_layers}，实际{steering_vectors.shape[0]}")
            if steering_vectors.shape[1] != expected_dim:
                raise ValueError(f"引导向量维度不匹配：需{expected_dim}，实际{steering_vectors.shape[1]}")
            print(f"Steering vectors validated: shape={steering_vectors.shape}")
        
        # 加载模型
        model, alphabet, batch_converter = load_esm_model(config["model_name"])
        prediction_model = ESM2ForBindingPrediction(
            esm_model=model,
            steering_vectors=steering_vectors
        ).to(device)

        # 加载数据（使用自定义collate_fn）
        train_df = pd.read_csv(config["train_file"])
        val_df = pd.read_csv(config["val_file"])
        test_df = pd.read_csv(config["test_file"])
        print(f"\nData Summary (before filtering):")
        print(f"Train samples: {len(train_df)}")
        print(f"Val samples (independent): {len(val_df)}")
        print(f"Test samples: {len(test_df)}")
        # 创建数据集
        train_dataset = TCRBindingDataset(train_df, max_length=config["max_seq_length"],col1=args.col1, col2=args.col2, col3=args.col3)
        val_dataset = TCRBindingDataset(val_df, max_length=config["max_seq_length"],col1=args.col1, col2=args.col2, col3=args.col3)
        test_dataset = TCRBindingDataset(test_df, max_length=config["max_seq_length"],col1=args.col1, col2=args.col2, col3=args.col3)

        # 创建DataLoader（关键：使用自定义collate_fn）
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            collate_fn=custom_collate_fn  # 强制使用自定义分组逻辑
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )

        # 优化器与调度器
        trainable_params = [p for p in prediction_model.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable_params,
            lr=config["learning_rate"],
            weight_decay=0.01
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",  # 二分类中，我们最大化F1分数或AUC
            factor=0.5,
            patience=2,
            verbose=True,
            min_lr=1e-7
        )
        
        print(f"\n{'='*50} Starting Training (ESM Library + Activation Steering) {'='*50}")
        for epoch in range(config["num_epochs"]):
            epoch_num = epoch + 1
            print(f"\nEpoch {epoch_num}/{config['num_epochs']}")
            
            train_metrics = train_epoch(
                model=prediction_model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                batch_converter=batch_converter
            )

            # 打印训练指标（包含所有新增指标）
            print(f"\nTrain Metrics: "
                  f"Loss={train_metrics['loss']:.4f} | "
                  f"Accuracy={train_metrics['accuracy']:.4f} | "
                  f"Precision={train_metrics['precision']:.4f} | "
                  f"Recall={train_metrics['recall']:.4f} | "
                  f"Sensitivity={train_metrics['sensitivity']:.4f} | "
                  f"Specificity={train_metrics['specificity']:.4f} | "
                  f"F1={train_metrics['f1']:.4f} | "
                  f"AUC={train_metrics['auc']:.4f} | "
                  f"PR-AUC={train_metrics['pr_auc']:.4f} | "
                  f"MCC={train_metrics['mcc']:.4f}")
            
            # 验证
            val_metrics, _, _, _ = evaluate(
                model=prediction_model,
                dataloader=val_dataloader,
                device=device,
                batch_converter=batch_converter,
                dataset_name="Independent Validation"
            )
            # 打印验证指标
            print(f"Val Metrics (Independent): "
                  f"Accuracy={val_metrics['accuracy']:.4f} | "
                  f"Precision={val_metrics['precision']:.4f} | "
                  f"Recall={val_metrics['recall']:.4f} | "
                  f"Sensitivity={val_metrics['sensitivity']:.4f} | "
                  f"Specificity={val_metrics['specificity']:.4f} | "
                  f"F1={val_metrics['f1']:.4f} | "
                  f"AUC={val_metrics['auc']:.4f} | "
                  f"PR-AUC={val_metrics['pr_auc']:.4f} | "
                  f"MCC={val_metrics['mcc']:.4f}")
            
            # 测试
            test_metrics, _, _, _ = evaluate(
                model=prediction_model,
                dataloader=test_dataloader,
                device=device,
                batch_converter=batch_converter,
                dataset_name="Test"
            )
            # 打印测试指标
            print(f"Test Metrics (Independent): "
                  f"Accuracy={test_metrics['accuracy']:.4f} | "
                  f"Precision={test_metrics['precision']:.4f} | "
                  f"Recall={test_metrics['recall']:.4f} | "
                  f"Sensitivity={test_metrics['sensitivity']:.4f} | "
                  f"Specificity={test_metrics['specificity']:.4f} | "
                  f"F1={test_metrics['f1']:.4f} | "
                  f"AUC={test_metrics['auc']:.4f} | "
                  f"PR-AUC={test_metrics['pr_auc']:.4f} | "
                  f"MCC={test_metrics['mcc']:.4f}")
            
            # 学习率调度 - 使用AUC作为指标
            scheduler.step(val_metrics["auc"])
            
            model_save_path = f"{config['output_dir']}/{config['log_mark']}_ESM2_3b.pt"
            torch.save({
                'model_state_dict': prediction_model.state_dict(),
                'config': config,
                'alphabet': alphabet  # 用于后续重建 batch_converter
            }, model_save_path)
            print(f"\n✅ Final model saved to: {model_save_path}")
        # ================== 保存最终模型 ==================

        

        print(f"\n{'='*50} Training Completed {'='*50}")
        print(f"All results stored in: {config['output_dir']}")

    finally:
        # 恢复原始stdout
        sys.stdout.close()
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()

    """
    nohup python ESM2_3b_FT.py --train-file /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/train_fold_1.csv --val-file /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/val_fold_1.csv --test-file /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/independent_set.csv  --log_mark unifyimmune_tcr_independent_FT --device cuda:0 --use-steering False --col1 tcr --col2 peptide --col3 label > ft_unifyimmune_tcr_pep.log 2>&1 &

    nohup python ESM2_3b_FT.py --train-file /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/train_fold_1.csv --val-file /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/val_fold_1.csv --test-file /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/independent_set.csv  --log_mark unifyimmune_HLA_independent_FT --device cuda:1 --use-steering False --col1 HLA --col2 peptide --col3 label > ft_unifyimmune_hla_pep.log 2>&1 &

    #################################################################
    
    python ESM2_3b.py --train-file /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/train_fold_1.csv --val-file /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/val_fold_1.csv --test-file /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_TCR/independent_set.csv  --log_mark unifyimmune_tcr_FT_9ceng --device cuda:0 --use-steering False --col1 tcr --col2 peptide --col3 label


    python ESM2_3b.py --train-file /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/train_fold_1.csv --val-file /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/val_fold_1.csv --test-file /mnt/lustre/guopeijin/Immune_LLM/AIR_dataset/unifyimmune_data/data_HLA/external_set.csv  --log_mark unifyimmune_HLA_FT_9ceng --device cuda:0 --use-steering False --col1 HLA --col2 peptide --col3 label


    """

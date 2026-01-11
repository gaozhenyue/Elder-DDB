from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import  average_precision_score

import time

# 训练与评估框架
class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer, 
                 device='cuda', early_stop_patience=5, lr_scheduler=None):
        """
        改进后的训练器初始化
        参数:
            model: 模型实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            criterion: 损失函数
            optimizer: 优化器
            device: 训练设备
            early_stop_patience: 早停耐心值
            lr_scheduler: 学习率调度器
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.best_auc = 0
        self.best_model = None
        self.early_stop_patience = early_stop_patience
        self.no_improve = 0
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        iter_count = 0
        time_now = time.time()
        
        for i, (X, y) in enumerate(self.train_loader):
            iter_count += 1
            X = X.float().to(self.device)
            y = y.to(self.device)
            
            # 前向传播
            outputs = self.model(X)
            loss = self.criterion(outputs, y.long().squeeze(-1))
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 定期打印训练信息
            if (i + 1) % 100 == 0:
                speed = (time.time() - time_now) / iter_count
                print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                print(f"\tspeed: {speed:.4f}s/iter")
                iter_count = 0
                time_now = time.time()
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        total_loss = []
        preds = []
        trues = []
        
        with torch.no_grad():
            for X, y in data_loader:
                X = X.float().to(self.device)
                y = y.to(self.device)
                
                outputs = self.model(X)
                loss = self.criterion(outputs, y.long().squeeze())
                total_loss.append(loss.item())
                
                preds.append(outputs.detach())
                trues.append(y)
        
        total_loss = np.average(total_loss)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        
        # 计算各类指标
        probs = torch.nn.functional.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        
        accuracy = (predictions == trues).mean()
        positive_probs = probs[:, 1].detach().cpu().numpy()
        auroc = roc_auc_score(trues, positive_probs)
        auprc = average_precision_score(trues, positive_probs)
        
        return total_loss, accuracy, auroc, auprc
    
    def train(self, num_epochs=50):
        """完整训练流程"""
        train_losses = []
        val_losses = []
        val_aucs = []
        
        for epoch in range(num_epochs):
            epoch_time = time.time()
            
            # 训练阶段
            train_loss = self.train_epoch(epoch)
            train_losses.append(train_loss)
            
            # 验证阶段
            val_loss, val_acc, val_auc, val_auprc = self.evaluate(self.val_loader)
            val_losses.append(val_loss)
            val_aucs.append(val_auc)
            
            # 测试阶段
            test_loss, test_acc, test_auc, test_auprc = self.evaluate(self.test_loader)
            
            # 打印epoch信息
            epoch_time = time.time() - epoch_time
            print(f"\nEpoch {epoch + 1}/{num_epochs} cost time: {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | Val AUPRC: {val_auprc:.4f}")
            print(f"Test Acc: {test_acc:.4f} | Test AUC: {test_auc:.4f} | Test AUPRC: {test_auprc:.4f}")
            
            # 学习率调整
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(val_loss)
            
            # 早停机制和模型保存
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.best_model = self.model.state_dict().copy()
                self.no_improve = 0
            else:
                self.no_improve += 1
                if self.no_improve >= self.early_stop_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # 加载最佳模型
        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)
        
        return train_losses, val_losses, val_aucs



# 评估器类
class ModelEvaluator:
    def __init__(self, model, test_loader, device='cuda'):
        """
        初始化评估器
        
        参数:
            model: 训练好的LSTM模型
            test_loader: 测试数据加载器
            device: 评估设备
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
    
    def evaluate(self):
        """评估模型性能"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X, y in tqdm(self.test_loader, desc="Evaluating"):
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        metrics = {
            'auc': roc_auc_score(all_labels, all_preds),
            'auprc': average_precision_score(all_labels, all_preds),
            'accuracy': np.mean((np.array(all_preds) > 0.5) == np.array(all_labels))
        }
        
        return metrics
    
    def predict_patient(self, patient_df, feature_cols, window_size=24):
        """
        为单个患者生成预测
        
        参数:
            patient_df: 单个患者的数据DataFrame
            feature_cols: 特征列名列表
            window_size: 时间窗口大小
        """
        self.model.eval()
        patient_df = patient_df.sort_values('hr')
        max_hr = patient_df['hr'].max()
        
        predictions = []
        for end in range(1, max_hr + 1):
            start = max(0, end - window_size)
            X = patient_df[(patient_df['hr'] >= start) & (patient_df['hr'] < end)][feature_cols].values
            
            if len(X) < window_size:
                pad_len = window_size - len(X)
                X = np.pad(X, ((0, pad_len), (0, 0)), 'constant')
            elif len(X) > window_size:
                X = X[-window_size:]
            
            X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pred = self.model(X_tensor).item()
                predictions.append(pred)
        
        return predictions
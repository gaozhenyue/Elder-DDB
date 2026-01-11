from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import  average_precision_score

# 训练与评估框架
class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer,device='cuda'):
        """
        初始化训练器
        参数:
            model: 模型实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 训练设备
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.best_auc = 0
        self.best_model = None
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [Train]")
        
        for X, y in progress_bar:
            X, y = X.to(self.device), y.to(self.device)
            
            # 前向传播
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            # loss = self.criterion(outputs, y.long().squeeze(-1))
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()
            
            total_loss += loss.item() * X.size(0)
            progress_bar.set_postfix(loss=loss.item())
        
        return total_loss / len(self.train_loader.dataset)
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        # total_loss = []
        # preds = []
        # trues = []
        
        with torch.no_grad():
            for X, y in tqdm(self.val_loader, desc="Validating"):
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)      
                loss = self.criterion(outputs, y)
                
                val_loss += loss.item() * X.size(0)
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        val_loss /= len(self.val_loader.dataset)
        val_auc = roc_auc_score(all_labels, all_preds)
        val_auprc = average_precision_score(all_labels, all_preds)
        
        # 保存最佳模型
        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.best_model = self.model.state_dict().copy()
        
        return val_loss, val_auc, val_auprc
    
    def train(self, num_epochs=50, early_stop_patience=5):
        """完整训练流程"""
        train_losses = []
        val_losses = []
        val_aucs = []
        
        no_improve = 0
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_auc, val_auprc = self.validate()
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_aucs.append(val_auc)
            
            print(f"Epoch {epoch + 1}/{num_epochs}:")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val AUC: {val_auc:.4f} | Val AUPRC: {val_auprc:.4f}")
            
            # # 早停机制
            # if val_auc > self.best_auc:
            #     no_improve = 0
            # else:
            #     no_improve += 1
            #     if no_improve >= early_stop_patience:
            #         print(f"Early stopping at epoch {epoch + 1}")
            #         break
        
        # 加载最佳模型
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
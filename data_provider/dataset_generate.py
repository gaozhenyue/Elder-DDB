import numpy as np
import random
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Dataset, DataLoader, RandomSampler, SequentialSampler

class TimeSeriesDataset(Dataset):
    def __init__(self, df, feature_cols, window_size=24, forecast_horizon=24, stride=1, 
                 mode='sliding', shuffle=True, label=['death_hosp'], random_state=42, max_len=None):
        """
        初始化时间序列数据集
        
        参数:
            df: 包含所有数据的DataFrame
            feature_cols: 使用的特征列名列表
            window_size: 输入序列长度
            forecast_horizon: 预测时间范围
            mode: 'sliding'滑动窗口，'cumulative'累积窗口，‘fix’固定时间窗口
            shuffle: 是否打乱数据顺序
            random_state: 随机种子
            label: 标签
        """
        self.df = df
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.mode = mode
        self.shuffle = shuffle
        self.random_state = random_state
        self.max_len = max_len
        self.indices = []
        self.stride = stride
        self.label = label
        

        # if len(label)==1:
        #     self.label = label[0]
        # else:
        #     self.label = label
        
        # 预计算所有可能的序列索引
        self._precompute_indices()
        
    def _precompute_indices(self):
        """计算所有有效的序列索引"""
        random.seed(self.random_state)
        
        for pid, group in tqdm(self.df.groupby('id')):
            group = group.sort_values('hr')
            max_hr = group['hr'].max()
            
            if self.mode == 'sliding':
                for start in range(1, max_hr - self.window_size, self.stride):
                    end = start + self.window_size
                    forecast_end = end + self.forecast_horizon
                    if len(group[(group['hr'] >= start) & (group['hr'] < end)]) == self.window_size:
                        y = []
                        for label in self.label:
                            condition = (group['hr'] >= end) & (group['hr'] < forecast_end) & (group[label] == 1)
                            y = int(condition.any())
                        self.indices.append((pid, start, end, y))
                        
            elif self.mode == 'cumulative':
                for end in range(1, max_hr + 1, self.stride):
                    start = 1
                    forecast_end = end + self.forecast_horizon
                    y = []
                    for label in self.label:
                        condition = (group['hr'] >= end) & (group['hr'] < forecast_end) & (group[label] == 1)
                        y = int(condition.any())
                    self.indices.append((pid, start, end, y))
                    
            elif self.mode == 'fix':
                start = 1
                end = start + self.window_size
                if len(group[(group['hr'] >= start) & (group['hr'] < end)]) == self.window_size:
                    y = []
                    for label in self.label:
                        condition = (group['hr'] >= end) & (group[label] == 1)
                        y.append(int(condition.any()))
                    self.indices.append((pid, start, end, y))
                        
                    # # condition = (group['hr'] >= end) & (group['death_hosp'] == 1)
                    # condition = group['death_hosp'] == 1
                    # y = int(condition.any())
                    # self.indices.append((pid, start, end, y))
                
        
        if self.shuffle:
            random.shuffle(self.indices)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        pid, start, end, y = self.indices[idx]
        group = self.df[self.df['id'] == pid].sort_values('hr')
        
        # 获取特征序列
        X = group[(group['hr'] >= start) & (group['hr'] < end)][self.feature_cols].values
        
        # 转换为torch张量
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        if self.max_len is not None:
            X_tensor = X_tensor[-self.max_len:]  # 截断到最大长度
            
        seq_len = len(X_tensor)
        
        return X_tensor, y_tensor, seq_len

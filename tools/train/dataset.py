'''
Copyright: © 2021, BeingGod. All rights reserved.
Author: BeingGod
Date: 2021-02-21 23:46:58
LastEditors: BeingGod
LastEditTime: 2021-02-24 00:00:47
Description: 自定义数据集类
'''

import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from PIL import Image

class GarbageDataSet(Dataset):
    def __init__(self, df, transform=None):
        """实现初始化方法，在初始化的时候将数据读载入"""
        self.df = df
        self.transform = transform
        
    def __len__(self):
        '''
        返回df的长度
        '''
        return len(self.df)

    def __getitem__(self, idx):
        '''
        根据 idx 返回一行数据
        '''
        filepath = self.df.iloc[idx].filepath
        label = self.df.iloc[idx].label
        img = Image.open(filepath)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label


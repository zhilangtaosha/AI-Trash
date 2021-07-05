'''
Copyright: © 2021, BeingGod. All rights reserved.
Author: BeingGod
Date: 2021-02-21 23:45:48
LastEditors: BeingGod
LastEditTime: 2021-05-08 16:59:00
Description: 模型训练
'''

import time
import torch,os,torchvision
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.transforms.transforms import CenterCrop, ColorJitter, RandomRotation, RandomVerticalFlip
from dataset import GarbageDataSet
from conf import config


CUDA = torch.cuda.is_available()                               # GPU是否可用
DEVICE = torch.device("cuda" if CUDA else "cpu")               # 选择设备
DATASET_ROOT = config['dataset_root']                          # 数据集根目录


'''
训练参数
'''
CATEGORY_NUM = config['train']['category_num']                 # 需要判断的垃圾种类数
FC_LEARNING_RATE = config['train']['fc_learning_rate']         # FC层学习率
BATCH_SIZE = config['train']['batch_size']
TRAIN_IMG_MEAN = config['train']['img_mean']
TRAIN_IMG_STD = config['train']['img_std']
EPOCH = config['train']['epoch']
SAVE_PER_EPOCH = config['train']['save_per_epoch']

'''
验证参数
''' 
VAL_PER_EPOCH = config['val']['val_per_epoch']
VAL_IMG_MEAN = config['val']['img_mean']
VAL_IMG_STD = config['val']['img_std']


'''
加载数据
'''
def load_data(train_transforms,val_transforms):
    df = pd.read_csv(os.path.join(DATASET_ROOT, 'labels.csv'))  # 读取标签文件

    dataset_names = ['train', 'valid']

    stratified_split = StratifiedShuffleSplit(
        n_splits=1, test_size=0.1)  # 以9：1划分训练集和验证集
    train_split_idx, val_split_idx = next(
        iter(stratified_split.split(df.filepath, df.label)))

    train_df = df.iloc[train_split_idx].reset_index()
    val_df = df.iloc[val_split_idx].reset_index()

    image_transforms = {'train': train_transforms, 'valid': val_transforms}

    train_dataset = GarbageDataSet(
        train_df, transform=image_transforms['train'])  # 实例化训练集
    val_dataset = GarbageDataSet(
        val_df, transform=image_transforms['valid'])  # 实例化验证集
        
    image_dataset = {'train': train_dataset, 'valid': val_dataset}
    
    image_dataloader = {x: DataLoader(
        image_dataset[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0) for x in dataset_names}  # 使用DataLoader加载数据集

    return image_dataloader
    
'''
加载预训练模型
'''
def load_model():
    model_ft = models.resnet18(pretrained=True)  # 下载官方的预训练模型
    # 将所有的参数层进行冻结
    for param in model_ft.parameters():
        param.requires_grad = False

    num_fc_ftr = model_ft.fc.in_features  # 获取到fc层的输入
    # 定义一个新的FC层
    model_ft.fc = nn.Linear(num_fc_ftr, CATEGORY_NUM)
    model_ft = model_ft.to(DEVICE)  # 放到设备中
    
    return model_ft


'''
训练
'''
def train(model,loader,optimizer,criterion,epoch):
    t1 = time.time()

    model.train()
    for batch_idx, data in enumerate(loader):
        x, y = data
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        y_hat = model(x) # 预测结果
        loss = criterion(y_hat, y) # 计算损失
        loss.backward() # 反向传播
        optimizer.step() # 梯度下降
        
    t2 = time.time()
    print('[Train] Epoch: {}  Loss: {:.6f}  Cost Time: {:.2f}s'.format(epoch, loss.item(),t2-t1))


'''
验证
'''
def val(model, loader, optimizer, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    sample_num = len(loader.dataset) # 验证集样本数
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            x, y = data
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            y_hat = model(x)
            val_loss += criterion(y_hat, y).item()
            pred = y_hat.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
            
    val_loss /= sample_num

    print('[Valid] Average Loss: {:.4f}  Accuracy: {}/{} ({:.0f}%)'.format(
        val_loss, correct, sample_num,
        100. * correct / sample_num))

'''
进行训练
'''
def run():
    # 训练集变换
    train_transforms = transforms.Compose([
        transforms.Resize(224),  # 修改图像大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.3,0.3,0.1,0),
        # transforms.CenterCrop(224, 224),
        transforms.ToTensor(),
        transforms.Normalize(TRAIN_IMG_MEAN, TRAIN_IMG_STD)  # 归一化
    ])

    # 验证集变换
    val_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomVerticalFlip(),
        # transforms.CenterCrop(224,224),
        transforms.ToTensor(),
        transforms.Normalize(VAL_IMG_MEAN, VAL_IMG_STD)
    ])

    dataloader = load_data(train_transforms, val_transforms) # 加载数据集
    model = load_model() # 加载模型

    criterion = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = torch.optim.Adam([
        {'params': model.fc.parameters()}
    ], lr=FC_LEARNING_RATE)  # 定义优化器并指定新加的fc层的学习率

    for epoch in range(EPOCH):
        train(model, dataloader['train'],optimizer,criterion, epoch)
        
        if (epoch+1) % VAL_PER_EPOCH == 0:
            # 每VAL_PER_EPOCH个epoch评估模型
            val(model, dataloader['valid'], optimizer, criterion)
        
        if (epoch+1) % SAVE_PER_EPOCH == 0:
            # 每SAVE_PER_EPOCH个epoch保存模型
            torch.save(model, './checkpoints/resnet18_fine_tune_{}.pth'.format(epoch))


'''
入口函数
'''
if __name__ == '__main__':
    run()

'''
Copyright: © 2021, BeingGod. All rights reserved.
Author: BeingGod
Date: 2021-02-22 10:38:36
LastEditors: BeingGod
LastEditTime: 2021-05-16 02:08:41
Description: 训练配置文件
'''
config = {
    'dataset_root': "/home/beinggod/workspace/AI-Trash/dataset/dataset_final/",
    
    'train':
    {
        'epoch': 200, # 训练总轮数
        'save_per_epoch': 10, # 每隔x轮保存一次模型
        'category_num': 11, # 待识别类别数
        'fc_learning_rate': 0.0005, # FC层学习率
        'batch_size': 128, # 6G显存选择128
        'img_mean': [0.72203826,0.80427701,0.81074095],  # 样本均值
        'img_std': [0.08624548,0.11264879,0.13353503],  # 样本标准差
    },

    'val':
    { 
        'val_per_epoch': 10, # 每隔x轮验证一次模型
        'img_mean': [0.72203826,0.80427701,0.81074095],  # 样本均值
        'img_std': [0.08624548,0.11264879,0.13353503],  # 样本标准差
    }
}

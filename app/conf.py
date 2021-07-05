'''
Copyright: © 2021, BeingGod. All rights reserved.
Author: BeingGod
Date: 2021-05-02 11:09:53
LastEditors: BeingGod
LastEditTime: 2021-05-08 17:48:15
Description: 
'''

config = {
    # 标签 -> 物体名
    'label_2_name' : {
        0: '无垃圾',
        1: '烟头',
        2: '砖头',
        3: '瓶子',
        4: '土豆',
        5: '香蕉',
        6: '娃娃菜',
        7: '陶瓷',
        8: '电池',
        9: '苹果',
        10:'易拉罐'
    },

    # 各类垃圾标签
    'label_list': {
        'none':[0],
        'harmful':[8],
        'recycle': [6,10],
        'kitchen': [3,4,5,9],
        'others': [1,2,7],
    },

    'video_path':'./video/demo.mp4', # 宣传视频路径
    'usart': '/dev/ttyTHS1', # 串口名称
    'model_path': './model/resnet18_fine_tune_49.pth', # 模型路径
}

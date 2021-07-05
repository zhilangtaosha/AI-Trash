'''
Copyright: © 2021, BeingGod. All rights reserved.
Author: BeingGod
Date: 2021-05-08 18:33:36
LastEditors: BeingGod
LastEditTime: 2021-05-13 09:50:52
Description: 
'''

import sys
import typing
import cv2
import time
import torch
import serial
from PIL import Image
from conf import config
import torch.nn.functional as F
from traceback import print_exc
from torchvision import transforms

CUDA = torch.cuda.is_available()  # CUDA是否可用
DEVICE = torch.device("cuda" if CUDA else "cpu")  # 指定推理设备

'''
加载参数
'''
LABEL2NAME = config["label_2_name"]
NONE = config["label_list"]["none"]
HARMFUL = config["label_list"]["harmful"]
RECYCLE = config["label_list"]["recycle"]
KITCHEN = config["label_list"]["kitchen"]
OTHERS = config["label_list"]["others"]
MODEL_PATH = config['model_path']
VIDEO_PATH = config['video_path']
USART = config['usart']


class Demo():
    '''
    ===========
    垃圾分类线程
    ===========
    '''
    def __init__(self):
        self.score_thresh = 0.7  # 得分阈值,低于该得分视为无垃圾
        self.camera = False
        self.usart = False
        self.model = self.__load_model()

        self.__init_USB_camera()
        self.__init_USART()

    def __load_model(self):
        '''
        加载模型
        '''
        try:
            model = torch.load(MODEL_PATH)
            model.to(DEVICE)

            # warm up
            x = torch.randn(64, 3, 7, 7).to(DEVICE)
            model(x)

        except Exception as e:
            print_exc(limit=1, file=sys.stdout)
            print("[ERROR] 模型加载失败! 退出程序...")
            sys.exit(-1)

        return model

    def __init_USART(self):
        '''
        初始化串口
        '''
        try:
	        self.usart = serial.Serial(USART, 115200, timeout=0.5)

        except Exception as e:
            print_exc(limit=1, file=sys.stdout)
            print("[ERROR] 串口初始化失败! 退出程序...")
            sys.exit(-1)

    def __init_USB_camera(self):
        '''
        初始化USB相机
        '''
        try:
            self.camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)

        except Exception as e:
            print_exc(limit=1, file=sys.stdout)
            print("[ERROR] USB相机初始化失败! 退出程序...")
            sys.exit(-1)

    def __transforms(self):
        '''
        定义图像变换
        '''
        img_mean = [0.91341771, 0.91523635, 0.90348119]
        img_std = [0.08975674, 0.10395834, 0.10300624]

        image_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(img_mean, img_std)
        ])

        return image_transforms

    def __infer(self, src, image_transforms):
        '''
        推理
        '''
        # 预处理图像
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)  # 颜色空间转换
        src = Image.fromarray(src)
        src_tensor = image_transforms(src)
        src_tensor = src_tensor.view(1, 3, 224, 224)
        src_tensor_gpu = src_tensor.to(DEVICE)  # 将图像转移至GPU

        # 执行推理
        output = self.model(src_tensor_gpu)
        output = F.softmax(output, dim=1).detach().cpu().numpy().flatten()
        category_index = output.argmax()
        score = output[category_index]

        if score < self.score_thresh or category_index == 0:
            # 小于阈值
            print("[LOG] 推理结果 Category: 无垃圾  Score: 0")

            return 0, self.score_thresh
        else:
            print("[LOG] 推理结果 Category: {}  Score: {:.4f}".format(
                LABEL2NAME[category_index], score))
            return category_index, output[category_index]

    def __send(self, category_id):
        '''
        Jetson Nano -> STM32
        '''
        if self.usart == False:
            print("[ERROR] 串口未初始化!")
            return False

        # emit分类信息
        if category_id in NONE:
            self.usart.write('R000'.encode('utf-8'))
            print("[LOG] 无垃圾")
            return True

        if category_id in HARMFUL:
            self.usart.write('R001'.encode('utf-8'))
            print("[LOG] 有害垃圾")
            return True

        if category_id in KITCHEN:
            self.usart.write('R010'.encode('utf-8'))
            print("[LOG] 厨余垃圾")
            return True

        if category_id in RECYCLE:
            self.usart.write('R011'.encode('utf-8'))
            print("[LOG] 可回收垃圾")
            return True

        if category_id in OTHERS:
            self.usart.write('R100'.encode('utf-8'))
            print("[LOG] 其他垃圾")
            return True


    def __receive(self):
        '''
            ==================
            STM32->Jetson Nano
            ==================
            '''
        if self.usart == False:
                print("[ERROR] 串口未初始化!")
                return 0

            cnt = 0
            while True:
                data = self.usart.read(5)
                if len(data) == 0 or data[0] != 'r':
                    return 0

                if data[1] == 1:
                    cnt += 1
                else:
                    cnt = 0

                if cnt >= 3:
                    # 连续检测三次，数据有效
                    if data[1:4] == '001':
                        # 有害垃圾
                        return 1

                    if data[1:4] == '010':
                        # 厨余垃圾
                        return 2

                    if data[1:4] == '011':
                        # 可回收垃圾
                        return 3

                    if data[1:4] == '100':
                        # 其他垃圾
                        return 4

    def run(self):
        trans = self.__transforms()
        frame_cnt = 0

        while True:
            ret,src = self.camera.read()
            cv2.imshow("source",src)
            key = cv2.waitKey(30)
            
            if ret == True:
                frame_cnt+=1
            
            category_id = 0
            if key == 105: # 按键i
                image = cv2.resize(src, (224, 224))
                category_id, score = self.__infer(image, trans)
                self.__send(category_id)

            if key == 49: # 按键1
                self.usart.write('R000'.encode('utf-8'))
                print("[LOG] 无垃圾")

            if key == 50: # 按键2
                self.usart.write('R001'.encode('utf-8'))
                print("[LOG] 有害垃圾")

            if key == 51: # 按键3
                self.usart.write('R010'.encode('utf-8'))
                print("[LOG] 厨余垃圾")

            if key == 52: # 按键4
                self.usart.write('R011'.encode('utf-8'))
                print("[LOG] 可回收垃圾")

            if key == 53: # 按键5
                self.usart.write('R100'.encode('utf-8'))
                print("[LOG] 其他垃圾")

            if key == 27:
                break
            
            feedback = self.__receive()
            if feedback == 1:
                print("[RECEIVED] 有害垃圾")

            if feedback == 2:
                print("[RECEIVED] 厨余垃圾")

            if feedback == 3:
                print("[RECEIVED] 可回收垃圾")

            if feedback == 4:
                print("[RECEIVED] 其他垃圾")

if __name__ == "__main__":
    demo = Demo()
    demo.run()

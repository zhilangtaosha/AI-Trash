'''
Copyright: © 2021, BeingGod. All rights reserved.
Author: BeingGod
Date: 2021-05-09 16:00:59
LastEditors: BeingGod
LastEditTime: 2021-06-14 16:02:10
Description: 初赛
'''

'''
串口通信协议:

Jetson Nano -> STM32
====================
R000 无垃圾
R001 有害垃圾
R010 厨余垃圾
R011 可回收垃圾
R100 其他垃圾
====================
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
from PyQt5.QtWidgets import QWidget, QLabel, QApplication
from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from ui import Ui_Form

CUDA = torch.cuda.is_available() # CUDA是否可用
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


class VideoThread(QThread):
    '''
    ===========
    视频播放线程
    ===========
    '''
    changePixmap = pyqtSignal(QImage)

    def run(self):
        cap = cv2.VideoCapture(VIDEO_PATH)
        while True:
            ret, frame = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_2_qt_format = QImage(
                    rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_2_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)


class ClassifyThread(QThread):
    '''
    ===========
    垃圾分类线程
    ===========
    '''
    changePixmap = pyqtSignal(QImage)
    changeResLabel = pyqtSignal(str)

    def __init__(self, parent: typing.Optional[QObject]) -> None:
        super().__init__(parent=parent)
        self.score_thresh = 0.85  # 得分阈值,低于该得分视为无垃圾
        self.camera = False
        self.usart = False
        self.model = self.__load_model()

        self.__init_USART()
        self.__init_USB_camera()

    def __load_model(self):
        '''
        ========
        加载模型
        ========
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
        =========
        初始化串口
        =========
        '''
        try:
	        self.usart = serial.Serial(USART, 115200, timeout=0.5)

        except Exception as e:
            print_exc(limit=1, file=sys.stdout)
            print("[ERROR] 串口初始化失败! 退出程序...")
            sys.exit(-1)

    def __init_USB_camera(self):
        '''
        ===========
        初始化USB相机
        ===========
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
        ==========
        定义图像变换
        ==========
        '''
        img_mean = [0.68822448, 0.78478137, 0.7905188]  # 样本均值
        img_std = [0.08429461, 0.10463355, 0.12901294]  # 样本标准差

        image_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(img_mean, img_std)
        ])

        return image_transforms

    def __infer(self, src, image_transforms):
        '''
        ==================
                推理
        ==================
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

    def __send(self, object_id):
        '''
        ====================
        Jetson Nano -> STM32
        ====================
        '''
        if self.usart == False:
            print("[ERROR] 串口未初始化!")
            return False
        
        # emit分类信息
        if object_id in NONE:
            self.usart.write('R000'.encode('utf-8'))
            self.changeResLabel.emit("无垃圾")
            print("[LOG] 无垃圾")
            
        if object_id in HARMFUL:
            self.usart.write('R001'.encode('utf-8'))
            self.changeResLabel.emit("有害垃圾")
            print("[LOG] 有害垃圾")

        if object_id in KITCHEN:
            self.usart.write('R010'.encode('utf-8'))
            self.changeResLabel.emit("厨余垃圾")
            print("[LOG] 厨余垃圾")

        if object_id in RECYCLE:
            self.usart.write('R011'.encode('utf-8'))
            self.changeResLabel.emit("可回收垃圾")
            print("[LOG] 可回收垃圾")

        if object_id in OTHERS:
            self.usart.write('R100'.encode('utf-8'))
            self.changeResLabel.emit("其他垃圾")
            print("[LOG] 其他垃圾")

        return True

    def run(self):
        '''
        ==================
        拍摄 -> 识别 -> 控制
        ==================
        '''
        image_transforms = self.__transforms()
        frame_count = 1  # 统计拍摄的图像帧数

        while True:
            if self.camera == False:
                print("[WARNING] 相机未初始化! ")
                continue

            ret, frame = self.camera.read()

            object_id = 0
            if ret:
                # 拍摄到图像
                frame_count += 1

                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_2_qt_format = QImage(
                    rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_2_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)  # emit图像

                if frame_count % 5 == 0:
                    image = cv2.resize(frame, (224, 224))
                    object_id, score = self.__infer(image, image_transforms)
                    object_name = LABEL2NAME[object_id]

                    self.__send(object_id)



class App(QWidget, Ui_Form):
    '''
    =======
    APP界面
    =======
    '''
    def __init__(self):
        super().__init__()
        self.__init_UI()  # 初始化UI界面

    def __init_UI(self):
        self.setupUi(self)
        # 启动垃圾分类线程
        cls_th = ClassifyThread(self)
        cls_th.changePixmap.connect(self.__set_capture_image)
        cls_th.changeResLabel.connect(self.__set_res_label)
        cls_th.start()
        # 启动视频播放线程启动
        vid_th = VideoThread(self)
        vid_th.changePixmap.connect(self.__set_video_image)
        vid_th.start()

        self.show()

    @pyqtSlot(QImage)
    def __set_video_image(self, image):
        '''
        ==============
        更新Video图像帧
        ==============
        '''
        self.video_label.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(QImage)
    def __set_capture_image(self, image):
        '''
        ================
        更新capture图像帧
        ================
        '''
        self.capture_label.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(str)
    def __set_res_label(self, res):
        '''
        ==========
        更新识别结果
        ==========
        '''
        self.result_label.setText(res)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())

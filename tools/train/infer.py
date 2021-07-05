'''
Copyright: © 2021, BeingGod. All rights reserved.
Author: BeingGod
Date: 2021-02-22 23:24:06
LastEditors: BeingGod
LastEditTime: 2021-03-27 13:50:38
Description: 图像推理
'''

import torch
import cv2
import sys
import torchvision
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, models, transforms
from PIL import Image

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

'''
加载标签
'''
def load_label(path):
    label = []
    with open(path, 'r') as f:
        line = f.readline().replace('\n', '')
        while line:
            label.append(line)
            line = f.readline().replace('\n', '')

    return label

'''
推理
'''
def infer(model, array, transforms, label):
    image_array = array
    image_array = image_array.resize((224, 224), Image.ANTIALIAS)

    image_tensor = transforms(image_array)
    image_tensor = image_tensor.view(1, 3, 224, 224)
    image_tensor = image_tensor.to(DEVICE)  # 将图像转移至GPU

    with torch.no_grad():
        model.eval()

        output = model(image_tensor)
        output = F.softmax(output, dim=1).detach().cpu().numpy().flatten()
        category_index = output.argmax()
        score = output[category_index]
        
    if score < 0.6:
        print("None")
    else:
        print("Category: {}  Score: {:.4f}".format(label[category_index],score))


if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("[ERROR] Please specific arguments!")
    #     sys.exit(-1)

    image_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    model = torch.load(sys.argv[1])
    model.to(DEVICE)
    label = load_label('./label.txt')
    cap = cv2.VideoCapture(0)
    retval,frame = cap.read()

    while retval:
        cv2.imshow("source",frame)
        key = cv2.waitKey(30)
        retval,frame = cap.read()
        
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        infer(model, image, image_transforms, label)
        if key == 27:
            break
        
    # infer(model, sys.argv[2], image_transforms, label)

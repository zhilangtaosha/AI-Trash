'''
Copyright: © 2021, BeingGod. All rights reserved.
Author: BeingGod
Date: 2021-05-08 16:17:04
LastEditors: BeingGod
LastEditTime: 2021-05-08 16:25:45
Description: 
'''

import os
import sys

path = sys.argv[1]

os.chdir(path)
files = os.listdir('./')
label = files[0][-5]
for file in files:
    temp_name = "{}_temp.jpg".format(file[:-4])
    os.rename(file,temp_name)

files = os.listdir('./')
cnt = len(files)
for i in range(cnt):
    os.rename(files[i],'{}_{}.jpg'.format(i,label))

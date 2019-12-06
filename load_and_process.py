"""IMPORT NON PEDES RANDOM IMAGES"""
""" DUE TO ABSENCE OF HIGH COMPUTATION POWER, ONLY 1000 IMAGES COULD BE LOADED AT ONCE"""
"""I HAVE UPLOADED THE DATASET FOR TRAIN WHICH WILL INCREASE THE ACCURACY OF THE OUTPUT DETETCION"""

import cv2
import numpy as np 
import matplotlib.pyplot as plt
import random
import os
from PIL import Image
import torch
import torchvision

 #importing images from non-pedes dataset
Non_Pedes_path= '/home/smitu/Desktop/Coding/ML/Pedestrian_Detector/Dataset/NonPedestrians/'
Non_Pedes_batch = os.listdir(Non_Pedes_path)
Non_Pedes = []

#Processing the dataset
for index in range(0,1300):
	i = random.randint(0,6000)
	sample_np= Non_Pedes_batch[i]
	img_path_np = Non_Pedes_path+sample_np
	x=Image.open(img_path_np)
	x = x.resize((48,96),Image.ANTIALIAS)
	x=x.convert('L')
	# x=x.convert('1')
	Non_Pedes.append(x)

#splitting the non Pedes dataset for train and test
label_np = [0]*1300
Non_Pedes=zip(Non_Pedes,label_np)
Non_Pedes=list(Non_Pedes)
Non_Pedes_train=Non_Pedes[:1000]
Non_Pedes_train=list(Non_Pedes_train)
Non_Pedes_test = Non_Pedes[1000:1300]
#print(Non_Pedes_train[:3])

# IMPORTING IMAGES FROM PEDES DATASET
Pedes_path= '/home/smitu/Desktop/Coding/ML/Pedestrain_Detector/Dataset/Pedestrians/48x96/'
Pedes_batch = os.listdir(Pedes_path)
Pedes = []

#PROCESSING THE DATASET
for index in range(0,1001):
	i = random.randint(0,8000)
	sample_p= Pedes_batch[i]
	img_path_p = Pedes_path+sample_p
	y=Image.open(img_path_p)
	y=y.convert('L')
	# /y=y.convert('1')
	Pedes.append(y)

#SPLITTING THE DATA SET INTO TEST AND TRAIN
label_p = [1]*1001
Pedes=zip(Pedes,label_p)
Pedes=list(Pedes)	
Pedes_train= Pedes[:1000]
Pedes_test = Pedes[1000:1001]

#MERGING PEDES AND NON PEDES DATASET
train=Non_Pedes_train + Pedes_train
random.shuffle(train)

#SPLITTING INTO DATA AND LABEL-TRAIN
train_data = [a[0] for a in train]
train_label = [a[1] for a in train]
print(train_data[0].size)
print(train_data[0].mode)

#CONVERTING TO ARRAY
for img in range(0,2000):
	train_data[img]=np.array(train_data[img])
print(train_data[0].size,"QWERTYUIOP")ss
train_data=np.array(train_data)
train_data=train_data/255

train_data=train_data.reshape(2000,1,48,96).astype('float32')
print("ASDFGH",train_data[0].shape,train_data[0])
train_label=np.array(train_label).astype('float32')

#SPLITTING INTO DATA AND LABEL-TEST

test=Non_Pedes_test 
random.shuffle(test)
test_data = [a[0] for a in test]
test_label = [a[1] for a in test]
print(len(test_data), len(test_label))

#CONVERTING TO ARRAY
for img in range(0,300):
	test_data[img]=np.array(test_data[img])
test_data=np.array(test_data)
test_data=test_data/255
test_data=test_data.reshape(300,1,48,96).astype('float32')
test_label=np.array(test_label)
test_label=test_label.astype('float32') # int to float for labels


#EXTRA TEST FOR CONFIRMATION
test2 = [a[0] for a in Pedes_test]
test2_l=[a[1] for a in Pedes_test]
test2[0]=np.array(test2[0])
test2=np.array(test2)
test2=test2/255

test2=test2.reshape(1,48,96,1).astype('float32')
test2_l=np.array(test2_l)

"""
NON PEDES TRAIN-1000
NON PEDES TEST-300
PEDES TRAIN-1000
"""
#-*- coding: utf-8 -*-
# ECG Authentication
import io
import itertools
from collections import Counter
from scipy import signal
#from scipy.signal import butter, filtfilt
import argparse
import glob
import importlib
import os
import numpy as np
import shutil
import sklearn.metrics as skmetrics
import tensorflow as tf

from data import load_data,load_data_ecg, load_data_bcg, get_subject_files, get_subject_files_sleepmat, get_subject_files_sleephmc
from model import TinySleepNet
from minibatching import (iterate_minibatches,
                          iterate_batch_seq_minibatches,
                          iterate_batch_multiple_seq_minibatches)
from utils import (get_balance_class_oversample,
                   print_n_samples_each_class,
                   save_seq_ids,
                   load_seq_ids)
from logger import get_logger

# camera (Face Authentication)
import glob
import serial
import time
import datetime
import threading, requests
import cv2
import numpy as np
import os
import math
import sys
import Jetson.GPIO as GPIO
import numpy as np
from PIL import Image

# Siamese Network
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib.image as img
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os

# ECG
import pandas as pd
import natsort
import numpy as np
import time
import datetime
import os
import csv

# OpenCV Camera
import cv2
import os
import shutil
# feature extraction
import dlib

# ______User Variable_______

global env_enable, fin_enable, ecg_enable, cam_enable, door_enable
env_enable = 0   # environment sensor ON/OFF (camera timeout ON/OFF)
hod_enable = 1   # hands on detection (before ecg) ON/OFF
fin_enable = 1   # fingerprint authentication ON/OFF
ecg_enable = 1   # ECG authentication ON/OFF
cam_enable = 0   # face authentication ON/OFF
door_enable = 0  # 모터제어 기능 ON/OFF
bpf_enable = 1   # ECG signal high pass filtering ON/OFF

global env_illum
env_illum = 10 # 10 sec timeout (if cannot detect face)

global retry_num
retry_num = 2  # 인증 재시도 횟수
# _________________________
global shoot_, cam_fps
shoot_ = 10 # 얼굴사진 몇번 찍을지
cam_fps = 10 # camera fps to authentificate face
# iamese Network_________________________
global model_name
#model_name = "model_epoch100_batch_64_5in_20_bright.pt"
model_name = "model_epoch400_batch_64_5in_20_bright_ver2.pt"
#anomaly_model_name = "model_epoch198_batch_64_anomaly_5.pt"
#anomaly_model_name = "model_epoch173_batch_64_anomaly_6.pt"
anomaly_model_name = "model_epoch197_batch_64_anomaly_6.pt"

global bright_enable, feature_enable, blur_enable
bright_enable = 0
feature_enable = 0
blur_enable = 0

# Dataset Directory
#dataset_dir = 'data/face' # 20 picture per people
dataset_dir = 'data/face_fast' # 2 picture per people
dataset_dir2 = 'data/anomaly'
# The number of files (counting files in s1 folder)
global file_num
file_list = os.listdir(dataset_dir+'/s1')
file_num = len(file_list)
# _______________________________________


# MFA Result variable______
global cam_ok, ecg_ok, fin_ok
cam_ok = 0
ecg_ok = 0
fin_ok = 0
# _________________________

# ECG__________________
global sam_freq, save_sec, iterated
sam_freq = 100  # Hz
save_sec = 5    # 100 Hz * 5 sec (per one line)
#iterated = 50   # 100 Hz * 5 sec * 50 lines (TOTAL DATA = 4 min 10 sec)
#iterated = 360   # 100 Hz * 5 sec * 360 lines (TOTAL DATA = 30 min)
iterated = 2   # 100 Hz * 5 sec * 3 lines (TOTAL DATA = 15 sec)
ecg_realtime_dir = 'data/sleephmc/5ppl_official_test/'
#______________________

# camera number (0, 1, -1)
cam_number = 0

# import Adafruit_DHT
import threading, requests

global save_img_cnt
save_img_cnt = 0

global verify_user_id
verify_user_id = -1

global FTA, FTA_cnt
FTA = 0
FTA_cnt = 0

global c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10
c_1 = 0
c_2 = 0
c_3 = 0
c_4 = 0
c_5 = 0
c_6 = 0
c_7 = 0
c_8 = 0
c_9 = 0
c_10 = 0

TRUE = 1
FALSE = 0

# Basic response message definition
ACK_SUCCESS = 0x00
ACK_FAIL = 0x01
ACK_FULL = 0x04
ACK_NO_USER = 0x05
ACK_TIMEOUT = 0x08
ACK_GO_OUT = 0x0F  # The center of the fingerprint is out of alignment with sensor

# User information definition
ACK_ALL_USER = 0x00
ACK_GUEST_USER = 0x01
ACK_NORMAL_USER = 0x02
ACK_MASTER_USER = 0x03

USER_MAX_CNT = 1000  # Maximum fingerprint number

# Command definition
CMD_HEAD = 0xF5
CMD_TAIL = 0xF5
CMD_ADD_1 = 0x01
CMD_ADD_2 = 0x02
CMD_ADD_3 = 0x03
CMD_MATCH = 0x0C
CMD_DEL = 0x04
CMD_DEL_ALL = 0x05
CMD_USER_CNT = 0x09
CMD_COM_LEV = 0x28
CMD_LP_MODE = 0x2C
CMD_TIMEOUT = 0x2E

CMD_FINGER_DETECTED = 0x14

Finger_WAKE_Pin = 23
Finger_RST_Pin = 24

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(Finger_WAKE_Pin, GPIO.IN)
GPIO.setup(Finger_RST_Pin, GPIO.OUT)
GPIO.setup(Finger_RST_Pin, GPIO.OUT, initial=GPIO.HIGH)

g_rx_buf = []
PC_Command_RxBuf = []
Finger_SleepFlag = 0

# door motor setting
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(12, GPIO.OUT)
GPIO.setup(13, GPIO.OUT)

# rLock = threading.RLock()

# raspberry pi
# ser = serial.Serial("/dev/ttyS0", 19200)

# jetson nano
ser = serial.Serial(
    port="/dev/ttyTHS1",
    baudrate=19200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)

time.sleep(.1)

# Siamese Network _____________________________________________________________________________________
# OpenCV Camera___________________________________________________________________________

def Real_Taking_a_picture(shoot_num, fps):
    cam = cv2.VideoCapture(0)
    cam.set(3, 320)  # set video width
    cam.set(4, 240)  # set video height
    #cam.set(cv2.CAP_PROP_FPS, fps)  # set video fps

    face_detector = cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
    save_cnt = 0
    timeout_cnt = 0
    ## delete past files
    #for root, dirs, files in os.walk(dataset_dir2, topdown=False):
    #    for name in files:
    #        os.remove(os.path.join(root, name))

    # taking face photos
    #for shoot in range(0,shoot_num,1):
    while shoot_num > save_cnt:
        while True:
            timeout_cnt += 1

            if env_enable == 1:
                if timeout_cnt > env_illum*10:
                    print ("UNABLE TO DETECT FACE !!!!!!!!!!!!!")
                    break
                else:
                    pass
            else:
                pass

            ret, img = cam.read()
            # img = cv2.flip(img, 1) # flip video image vertically
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 3, maxSize=(92, 112), minSize=(92, 112))

            cv2.imshow('image', img)

            for (x, y, w, h) in faces:
                w_ = 92 - w
                h_ = 112 - h
                cv2.rectangle(img, (x, y), (x+w+w_, y+h+h_), (250, 253, 15), 2)
#                cv2.rectangle(img, (x, y), (x+w, y+h), (250, 253, 15), 2)

                # Save the captured image into the datasets folder
                gray_face = gray[y:y+h+h_, x:x+w+w_]

                if bright_enable == 1:
                    ## mean bright settings
                    mean_value = gray_face.mean()
                    alpha = 100 / mean_value
                    adjusted = cv2.convertScaleAbs(gray_face, alpha=alpha, beta=0)
                else:
                    adjusted = gray_face

                if blur_enable == 1:
                    adjusted = cv2.fastNlMeansDenoising(adjusted, None, 10, 7, 21)
                else:
                    pass

                if feature_enable == 1:
                    detector = dlib.get_frontal_face_detector()
                    predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
                    rects = detector(adjusted, 0)
                    for (i, rect) in enumerate(rects):
                        shape = predictor(adjusted, rect)
                        for i in range(68):
                            x = shape.part(i).x
                            y = shape.part(i).y
                            cv2.circle(adjusted, (x, y), 2, (0, 255, 0), -1)
                else:
                    pass

                cv2.imwrite(dataset_dir2 + '/' + str(save_cnt+1) + '_realtime.pgm', adjusted)
                cv2.imshow('image', img)
                save_cnt += 1

            k = cv2.waitKey(int(1000/fps)) & 0xff  # Press 'ESC' for exiting video

            if k == 27:
                break

            #print ("save_cnt :", save_cnt)
            if save_cnt >= 1:
                break

        if env_enable == 1:
            if timeout_cnt > env_illum*10:
                break
        else:
            pass

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

    if timeout_cnt > env_illum*10:
        return 0    # timeout
    else:
        return 1    # face image saved successfully


# Siamese Network (START) _________________________________________________________________________

def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


class Config():
    testing_dir = dataset_dir  ##### need to be modified
    train_batch_size = 64
    train_number_epochs = 150


class SiameseNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = self.imageFolderDataset.imgs[0]
        img1_tuple = self.imageFolderDataset.imgs[dataset_index]
        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class SiameseNetwork2(nn.Module):
    def __init__(self):
        super(SiameseNetwork2, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(256 * 6 * 6, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(512, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


def Number_of_files(path):
    total_files = 0
    for _, _, files in os.walk(path):
        if len(files) > 0:
            total_files = total_files + len(files)
    return total_files


# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 25 * 25, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 25 * 25)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DeepNet(nn.Module):
    def __init__(self):
        super(DeepNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 6 * 6, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 256 * 6 * 6)
        x = torch.relu(self.bn5(self.fc1(x)))
        x = torch.relu(self.bn6(self.fc2(x)))
        x = self.fc3(x)
        return x

class DeepDeepNet(nn.Module):
    def __init__(self):
        super(DeepDeepNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 6 * 6, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        batch_size = x.size(0)  # 현재 배치 크기 가져오기
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = x.view(batch_size, 256 * 6 * 6)
        x = torch.relu(self.bn5(self.fc1(x)))
        x = torch.relu(self.bn6(self.fc2(x)))
        x = self.fc3(x)
        return x

# Define dataset class
class PGMImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        for class_index, class_name in enumerate(['users', 'others']):
            class_dir = os.path.join(root_dir, class_name)
            for filename in os.listdir(class_dir):
                image_path = os.path.join(class_dir, filename)
                self.images.append(image_path)
                self.labels.append(class_index)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label


def Data_Anomaly(): # jaxson
    realtime_directory = dataset_dir2
    pgm_files = [file for file in os.listdir(realtime_directory) if file.endswith('.pgm')]

    result_dict = {}

    for pgm_file in pgm_files:

        IMAGE_PATH = dataset_dir2 + '/' + pgm_file
        MODEL_PATH = "model/" + anomaly_model_name

        # Define transforms
        test_transforms = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor()
        ])

        ## Load the model
        #model = Net() # change network 23.04.25
        #model = DeepNet()# change network 23.04.26
        model = DeepDeepNet() # change network 23.04.27
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()

        # Load and transform the input image
        input_image = Image.open(IMAGE_PATH)
        input_tensor = test_transforms(input_image)
        input_batch = input_tensor.unsqueeze(0)

        # Make a prediction
        with torch.no_grad():
            output = model(input_batch)
            output_list = output[0].tolist()
            #print(output)
            _, predicted = torch.max(output.data, 1)
            #print (predicted, output_list[predicted.item()]-output_list[1-predicted.item()])
            preds = predicted.item()
            probs = output_list[preds]-output_list[1-preds]
            print ('[',preds,']', probs)

        result_dict.setdefault('preds', []).append(preds)
        result_dict.setdefault('probs', []).append(probs)

        val_cnt = 0
        for val in result_dict['preds']:
            if val == 0:
                val_cnt += 1
            else:
                pass

    for i in range(1, len(result_dict['probs'])):
        if result_dict['preds'][i] == 1:
            result_dict['probs'][i] *= -1

    preds_sum = sum(result_dict['probs'])

    if preds_sum > 0:
        preds_bool = 0 # PASS
        result_str = 'PASS'
        # maximum score file extraction
        max_prob_val = max(result_dict['probs'])
        max_prob_idx = result_dict['probs'].index(max_prob_val)

    else:
        preds_bool = 1 # FAIL
        result_str = 'FAIL'
        max_prob_idx = 0

    print (">> PREDICT RESULT :", val_cnt, "/", len(result_dict['preds']))
    print (">> FINAL RESULT   :", '[', result_str, ']', preds_sum)

    return preds_bool, max_prob_idx



def Real_Siamese_Network(sub_file):

    delete_realtime_face_files()
    copy_realtime_face_files(sub_file)

    global dataset_index
    global file_num
    global shoot_
    global cam_ok
    dataset_index = 0
    net = SiameseNetwork2()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)

    counter = []
    loss_history = []
    iteration_number = 0

    # load model
    device = torch.device('cpu')
    # net.load_state_dict(torch.load("model/model_epoch"+str(Config.train_number_epochs)+"_batch_"+str(Config.train_batch_size)+"_5in_50_bright_norm.pt", map_location=device))
    net.load_state_dict(torch.load("model/" + model_name, map_location=device))
    net.eval()
    folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
    files_N = Number_of_files(dataset_dir+"/")

    # example.. save_id = { all files(55EA) - shoot_(5EA) } / s1(10EA) = 5 (people)
    # But, delete_realtime_face_files() 함수에서  하나 빼고 다 지움
    # 고로, int((files_N - shoot_) / file_num) -> int((files_N - 1) / file_num)으로 다시 수정
    saved_id = int((files_N - 1) / file_num)
#    print ("The number of id (saved people) : ", saved_id)


    # 5 people loop
    end_process = 0

    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                                    transform=transforms.Compose([transforms.Resize((100, 100)),
                                                                                  transforms.ToTensor()
                                                                                  ])
                                                   , should_invert=False)
    test_dataloader = DataLoader(siamese_dataset, num_workers=0, batch_size=1, shuffle=False)
    dataiter = iter(test_dataloader)
    x0, _, _ = next(dataiter)

    result_list = []
    face_auth = {'s1': 0, 's2': 0, 's3': 0, 's4': 0, 's5': 0}
    for seq in range(0, saved_id, 1):
        # 20 photo loop
        total_distance = 0
        print ('Comparing the face just taken with the face of user [', str(seq+1), ']')

        for i in range(0, file_num, 1):
            #file_num
            # global dataset_index
            dataset_index = dataset_index + 1

            # print (dataset_index, files_N)
            if dataset_index == files_N:
                end_process = 1
                break

            _, x1, label2 = next(dataiter)
            concatenated = torch.cat((x0, x1), 0)
            output1, output2 = net(Variable(x0), Variable(x1))
            euclidean_distance = F.pairwise_distance(output1, output2)
            total_distance = total_distance + euclidean_distance.item()
            print (dataset_index, ":", euclidean_distance.item())

        result_list.append(total_distance)


        # 한 장씩만 비교할 때 유사도( 거리값)  이미지에 표기
        #imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))

        # 여러장 비교할 때 유사도(거리값)  전체 데이터셋을 합산하여  이미지에 표기
        #imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(total_distance))


        if end_process == 1:
            break
        else:
            face_auth['s'+str(seq+1)] = total_distance

    min_key = result_list.index(min(result_list))
    avr_distance = (face_auth['s'+str(min_key+1)])/file_num
    print(">> Result : user", 's'+str(min_key+1), '(', avr_distance, ')')
    if avr_distance < 1:

        if (min_key == 0) or (min_key == 1) or (min_key == 2):
            cam_ok = 1
        else:
            cam_ok = 0
    else:
        cam_ok = 0

# Siamese Network (END) _________________________________________________________________________

## Humidity sensor (DHT11)
# class Environmental_Sensors:

#    def humi_sensor(self, sensor, pin):
#        self.humidity, self.temperature = Adafruit_DHT.read_retry(sensor, pin)
#        if self.humidity is not None and self.temperature is not None:
#            return int(round(self.temperature,0)), int(round(self.humidity,0))
#            #print('Temp={0:0.1f}*C  Humidity={1:0.1f}%'.format(temperature, humidity))
#        else:
#            return 0,0


# ***************************************************************************
# @brief    send a command, and wait for the response of module
# ***************************************************************************/
def TxAndRxCmd(command_buf, rx_bytes_need, timeout):
    # print (command_buf) # 40,0,5,0,0
    # print (type(command_buf)) # list
    global g_rx_buf
    CheckSum = 0
    tx_buf = []
    tx = ""
    if command_buf == None:
        pass
    else:
        tx_buf.append(CMD_HEAD)
        for byte in command_buf:
            tx_buf.append(byte)
            CheckSum ^= byte

        tx_buf.append(CheckSum)
        tx_buf.append(CMD_TAIL)

        # python 2
        #        for i in tx_buf:
        #            tx += chr(i)
        #        ser.flushInput()
        #        ser.write(tx)

        # python3
        ser.write(bytes(tx_buf))  # .decode('hex'))
        #print ("g_rx_buf: ", )

    g_rx_buf = []
    time_before = time.time()
    time_after = time.time()

    while time_after - time_before < timeout and len(g_rx_buf) < rx_bytes_need:  # Waiting for response
        bytes_can_recv = ser.inWaiting()
        if bytes_can_recv != 0:
            # g_rx_buf_utf += ser.read(bytes_can_recv).decode('utf-8')
            g_rx_buf += ser.read(bytes_can_recv)
            # g_rx_buf = g_rx_buf.decode('hex')
        time_after = time.time()

    #print ("g_rx_buf: ", g_rx_buf)
    #print ("len(g_rx_buf): ", len(g_rx_buf))

       # for i in range(len(g_rx_buf)):
       #     g_rx_buf[i] = ord(g_rx_buf[i])

    if len(g_rx_buf) < rx_bytes_need:
        return ACK_TIMEOUT
    if g_rx_buf[0] != CMD_HEAD:
        return ACK_FAIL
    if g_rx_buf[rx_bytes_need - 1] != CMD_TAIL:
        return ACK_FAIL
    if g_rx_buf[1] != tx_buf[1]:
        return ACK_FAIL

    #print ("g_rx_buf[2:4]: ", g_rx_buf[2:4]) #jiseong
    #if g_rx_buf[2:4] == # 하이로우 len이면
    #    g_rx_buf += ser.read(bytes_can_recv)

    CheckSum = 0
    for index, byte in enumerate(g_rx_buf):
        if index == 0:
            continue
        if index == 6:
            if CheckSum != byte:
                return ACK_FAIL
        CheckSum ^= byte
    return ACK_SUCCESS


# ***************************************************************************
# @brief    Get Compare Level
# ***************************************************************************/
def GetCompareLevel():
    global g_rx_buf
    command_buf = [CMD_COM_LEV, 0, 0, 1, 0]
    r = TxAndRxCmd(command_buf, 8, 0.1)
    if r == ACK_TIMEOUT:
        return ACK_TIMEOUT
    if r == ACK_SUCCESS and g_rx_buf[4] == ACK_SUCCESS:
        return g_rx_buf[3]
    else:
        return 0xFF


# ***************************************************************************
# @brief    Set Compare Level,the default value is 5,
#           can be set to 0-9, the bigger, the stricter
# ***************************************************************************/
def SetCompareLevel(level):
    global g_rx_buf
    command_buf = [CMD_COM_LEV, 0, level, 0, 0]
    r = TxAndRxCmd(command_buf, 8, 0.1)

    if r == ACK_TIMEOUT:
        return ACK_TIMEOUT
    if r == ACK_SUCCESS and g_rx_buf[4] == ACK_SUCCESS:
        return g_rx_buf[3]
    else:
        return 0xFF


# ***************************************************************************
# @brief   Query the number of existing fingerprints
# ***************************************************************************/
def GetUserCount():
    global g_rx_buf
    command_buf = [CMD_USER_CNT, 0, 0, 0, 0]
    r = TxAndRxCmd(command_buf, 8, 0.1)
    if r == ACK_TIMEOUT:
        return ACK_TIMEOUT
    if r == ACK_SUCCESS and g_rx_buf[4] == ACK_SUCCESS:
        return g_rx_buf[3]
    else:
        return 0xFF


# ***************************************************************************
# @brief   Get the time that fingerprint collection wait timeout
# ***************************************************************************/
def GetTimeOut():
    global g_rx_buf
    command_buf = [CMD_TIMEOUT, 0, 0, 1, 0]
    r = TxAndRxCmd(command_buf, 8, 5)
    if r == ACK_TIMEOUT:
        return ACK_TIMEOUT
    if r == ACK_SUCCESS and g_rx_buf[4] == ACK_SUCCESS:
        return g_rx_buf[3]
    else:
        return 0xFF


# ***************************************************************************
# @brief   Get image data (Jiseong)
# ***************************************************************************/
def RequestImage():
    global g_rx_buf
    command_buf = [0x24, 0, 0, 0, 0]
    r = TxAndRxCmd(command_buf, 8, 60)
    if r == ACK_TIMEOUT:
        #print ("TIMEOUTTIMEOUTTIMEOUTTIMEOUTTIMEOUTTIMEOUTTIMEOUT")
        return ACK_TIMEOUT
    if r == ACK_SUCCESS and g_rx_buf[4] == ACK_SUCCESS:
        #print ("SUCCESSSUCCESSSUCCESSSUCCESSSUCCESSSUCCESSSUCCESS")
        #print ("g_rx_buf: ", g_rx_buf)
        #print ("Hi(Len): ", g_rx_buf[2])
        #print ("Low(Len): ", g_rx_buf[3])
        return ACK_SUCCESS
    else:
        #print ("ELSEELSEELSEELSEELSEELSEELSEELSEELSEELSEELSEELSE")
        #print ("g_rx_buf: ", g_rx_buf)
        return 0xFF


# ***************************************************************************
# @brief    Register fingerprint
# ***************************************************************************/
def AddUser():
    global g_rx_buf
    # r = GetUserCount()
    usernum = input()
    if r >= USER_MAX_CNT:
        return ACK_FULL

    command_buf = [CMD_ADD_1, 0, usernum, 3, 0]
    print(g_rx_buf)
    r = TxAndRxCmd(command_buf, 8, 6)
    if r == ACK_TIMEOUT:
        return ACK_TIMEOUT
    if r == ACK_SUCCESS and g_rx_buf[4] == ACK_SUCCESS:
        command_buf[0] = CMD_ADD_3
        r = TxAndRxCmd(command_buf, 8, 6)
        print(g_rx_buf)
        if r == ACK_TIMEOUT:
            return ACK_TIMEOUT
        if r == ACK_SUCCESS and g_rx_buf[4] == ACK_SUCCESS:
            return ACK_SUCCESS
        else:
            return ACK_FAIL
    else:
        return ACK_FAIL


# ***************************************************************************
# @brief    Clear fingerprints
# ***************************************************************************/
def ClearAllUser():
    global g_rx_buf
    command_buf = [CMD_DEL_ALL, 0, 0, 0, 0]
    r = TxAndRxCmd(command_buf, 8, 5)
    if r == ACK_TIMEOUT:
        return ACK_TIMEOUT
    if r == ACK_SUCCESS and g_rx_buf[4] == ACK_SUCCESS:
        return ACK_SUCCESS
    else:
        return ACK_FAIL


# ***************************************************************************
# @brief    Check if user ID is between 1 and 3
# ***************************************************************************/
def IsMasterUser(user_id):
    if user_id == 1 or user_id == 2 or user_id == 3:
        return TRUE
    else:
        return FALSE


# ***************************************************************************
# @brief    Fingerprint matching
# ***************************************************************************/
def VerifyUser():
    global g_rx_buf
    global verify_user_id
    command_buf = [CMD_MATCH, 0, 0, 0, 0]
    r = TxAndRxCmd(command_buf, 8, 60)
    if r == ACK_TIMEOUT:
        return ACK_TIMEOUT
    if r == ACK_SUCCESS and IsMasterUser(g_rx_buf[4]) == TRUE:
        # print ("g_rx_buf: ", g_rx_buf)
        verify_user_id = g_rx_buf[3]
        # print ("No.", verify_user_id)
        return ACK_SUCCESS
    elif g_rx_buf[4] == ACK_NO_USER:
        return ACK_NO_USER
    elif g_rx_buf[4] == ACK_TIMEOUT:
        return ACK_TIMEOUT
    else:
        return ACK_GO_OUT  # The center of the fingerprint is out of alignment with sensor


# ***************************************************************************
# @brief    Analysis the command from PC terminal
# ***************************************************************************/
def Analysis_PC_Command(command):
    global Finger_SleepFlag, g_rx_buf, FTA
    global FTA_cnt, retry_num
    if command == "1" and Finger_SleepFlag != 1:
        print ("Number of fingerprints already available:  %d" % GetUserCount())
    elif command == "2" and Finger_SleepFlag != 1:
        print ("Add fingerprint  (Put your finger on sensor until successfully/failed information returned) ")
        r = AddUser()
        if r == ACK_SUCCESS:
            print ("Fingerprint added successfully !")
        elif r == ACK_FAIL:
            print (
                "Failed: Please try to place the center of the fingerprint flat to sensor, or this fingerprint already exists !")
        elif r == ACK_FULL:
            print ("Failed: The fingerprint library is full !")
    elif command == "3" and Finger_SleepFlag != 1:
        # print ("Waiting Finger......Please try to place the center of the fingerprint flat to sensor !")
        r = VerifyUser()
        if r == ACK_SUCCESS:
            #print ("Matching successful !")
            FTA_cnt = 0
            #Finger_SleepFlag = 1
        elif r == ACK_NO_USER:
            if FTA_cnt == retry_num:
                FTA = FTA + 1
                FTA_cnt = 0
                print ("Cannot find Fingerprint (There is no Fingerprint Enrolled)")
            else:
                print ("Retry : Touch your finger again!")
                FTA_cnt = FTA_cnt + 1
            # print ("Failed: Retry") # This fingerprint was not found in the library !")
        elif r == ACK_TIMEOUT:
            print ("Failed: Time out !")
        elif r == ACK_GO_OUT:
            print ("Failed: Please try to place the center of the fingerprint flat to sensor !")
    elif command == "4" and Finger_SleepFlag != 1:
        ClearAllUser()
        print ("All fingerprints have been cleared !")
    elif command == "5" and Finger_SleepFlag != 1:
        GPIO.output(Finger_RST_Pin, GPIO.LOW)
        Finger_SleepFlag = 1
        print (
            "Module has entered sleep mode: you can use the finger Automatic wake-up function, in this mode, only CMD6 is valid, send CMD6 to pull up the RST pin of module, so that the module exits sleep !")
    elif command == "6":
        Finger_SleepFlag = 0
        GPIO.output(Finger_RST_Pin, GPIO.HIGH)
        print ("The module is awake. All commands are valid !")
    elif command == "7" and Finger_SleepFlag != 1:
        print ("_________Request Image Data_________")
        print ("save fingerprint image (Put your finger on sensor until successfully/failed information returned) ")
        r = RequestImage()
        pre_img = g_rx_buf[8:]
        #print ("RequestImage() RESULT :", r)
        #r = GetTimeOut()
        #print ("GetTimeOut() RESULT :", r)
        #pre_img = []
        #print ("Wait a few seconds..")
        #while 1:
        #    TxAndRxCmd(None, 96, 1)
        #    #print (len(g_rx_buf))
        #    #print (g_rx_buf)
        #
        #    if len(g_rx_buf) == 0:
        #        pass
        #    else:
        #        for i in g_rx_buf:
        #            pre_img.append(i)
        #        if g_rx_buf[-1] == 245:
        #            break
        #
        #print (len(pre_img))
        #print (pre_img)

        global save_img_cnt
        if len(pre_img) == 3203:
            save_img_cnt = save_img_cnt + 1
            pre_img = pre_img[1:-2]
            print ("(success) img data len : ", len(pre_img))

            # 40*80 pixel to 80*80 pixel : jaxson jeong
            full_img = []
            for idx, val in enumerate(pre_img):
                trans_hex = format(hex(val)).replace("0x", "")
                upper_int = int(trans_hex[0], 16)
                lower_int = int(trans_hex[1], 16)
                full_img.append(upper_int*15)
                full_img.append(lower_int*15)

            # transform (int to img form np array) : jaxson jeong
            final_img = trans_txt2img(full_img)

            # save image file : jaxson jeong
            im = Image.fromarray(final_img.astype(np.uint8))
            img_filename = "data/fingerprint/realtime.bmp"
            im.save(img_filename)
            print ("(success) save img file as : [", img_filename, "]")

        else:
            print ("Failed recv img data")

    else:
        print ("commands are invalid !")


def trans_txt2img(img_data):
    img_pixel = np.empty((0, 3), int)
    img_line = np.empty((0, 3), int)
    for idx, val in enumerate(img_data):
        if (divmod(idx, 80)[1] == 0):
            list_img = img_data[idx:idx + 80]
            # print (list_img)
            # print (len(list_img))
            for val_ in list_img:
                img_pixel = np.array([[val_, val_, val_]])
                img_line = np.append(img_line, np.array(img_pixel), axis=0)
    img_line = img_line.reshape((80, 80, 3))
    return img_line
    # print ("final:",  img_line)
    # print ("final:",  img_line.shape)


# ***************************************************************************
# @brief   If you enter the sleep mode, then open the Automatic wake-up function of the finger,
#         begin to check if the finger is pressed, and then start the module and match
# ***************************************************************************/
def Auto_Verify_Finger():
    cnt_pass = 0
    cnt_fail = 0
    while True:
        global fin_ok
        Finger_SleepFlag = 1
        # If you enter the sleep mode, then open the Automatic wake-up function of the finger,
        # begin to check if the finger is pressed, and then start the module and match
        if Finger_SleepFlag == 1:
#            if GPIO.input(Finger_WAKE_Pin) == 1:  # If you press your finger
#                # time.sleep(0.01)
#                if GPIO.input(Finger_WAKE_Pin) == 1:
            GPIO.output(Finger_RST_Pin,GPIO.HIGH)  # Pull up the RST to start the module and start matching the fingers
            # time.sleep(0.25)      # Wait for module to start
            # print ("Waiting Finger......Please try to place the center of the fingerprint flat to sensor !")
            r = VerifyUser()
            if r == ACK_SUCCESS:
                cnt_pass = cnt_pass + 1

                #print (" - PASS : Door Opened !  at ", datetime.datetime.now())
                print ("Fingerprint scan finished")

                fin_ok = 1
                #Door_Open_Control("1")
                break # If fin_ok=True break 'finger auth (while True:)' and go to authenticate with face

            # retry chance (three times)
            #elif r == ACK_NO_USER:
            #    cnt_fail = cnt_fail + 1
            #    #Door_Open_Control("0")
            #    print("Retry : Touch your finger again!")
            #    if cnt_fail >= 3:
            #        fin_ok = 0
            #        print (" - FAIL : Door Closed !  at ", datetime.datetime.now())
            #        break
            elif r == ACK_NO_USER:
                cnt_fail = cnt_fail + 1
                #Door_Open_Control("0")
                fin_ok = 0

                #print (" - FAIL : Door Closed !  at ", datetime.datetime.now())
                print ("Fingerprint scan finished")
                break

            elif r == ACK_TIMEOUT:
                pass
                # print ("Failed: Time out !")
            elif r == ACK_GO_OUT:
                print ("Failed: Please try to place the center of the fingerprint flat to sensor !")
            # After the matching action is completed, drag RST down to sleep
            # and continue to wait for your fingers to press
            GPIO.output(Finger_RST_Pin, GPIO.LOW)
            #print ("PASS: ", cnt_pass, "  | FAIL: ", cnt_fail)
            #print ("_________________________________________")

#        else:
#            print ("GPIO.input(Finger_WAKE_Pin): ", GPIO.input(Finger_WAKE_Pin))
#            time.sleep(.2)

#            else:
#                print ("GPIO.input(Finger_WAKE_Pin): ", GPIO.input(Finger_WAKE_Pin))
#                time.sleep(.2)

        else:
            print ("Finger_SleepFlag: ",Finger_SleepFlag)
            time.sleep(.2)



def Door_Open_Control(enable):
    global door_enable
    # stop motor
    if door_enable == 1:
        GPIO.output(12, GPIO.LOW)
        GPIO.output(13, GPIO.LOW)
        time.sleep(.02)

        if enable == "1":
            # door open
            GPIO.output(13, GPIO.LOW)
            GPIO.output(12, GPIO.HIGH)
            time.sleep(2)

        else:
            # door close
            GPIO.output(12, GPIO.LOW)
            GPIO.output(13, GPIO.HIGH)
            time.sleep(2)

        # stop motor
        GPIO.output(12, GPIO.LOW)
        GPIO.output(13, GPIO.LOW)
        time.sleep(.02)
    else:
        pass


def finger_auth():
    GPIO.output(Finger_RST_Pin, GPIO.LOW)
    # time.sleep(0.25)
    GPIO.output(Finger_RST_Pin, GPIO.HIGH)
    # time.sleep(0.25)    # Wait for module to start
    # while SetCompareLevel(5) != 5:
    #    print ("***ERROR***: Please ensure that the module power supply is 3.3V or 5V, the serial line connection is correct.")
    # time.sleep(1)
    #    print ("***************************** WaveShare Capacitive Fingerprint Reader Test *****************************")
    #    print ("Compare Level:  5    (can be set to 0-9, the bigger, the stricter)")
    #    print ("Number of fingerprints already available:  %d "  % GetUserCount())
    #    print (" send commands to operate the module: ")
    #    print ("  CMD1 : Query the number of existing fingerprints")
    #    print ("  CMD2 : Registered fingerprint  (Put your finger on the sensor until successfully/failed information returned) ")
    #    print ("  CMD3 : Fingerprint matching  (Send the command, put your finger on sensor) ")
    #    print ("  CMD4 : Clear fingerprints ")
    #    print ("  CMD5 : Switch to sleep mode, you can use the finger Automatic wake-up function (In this state, only CMD6 is valid. When a finger is placed on the sensor,the module is awakened and the finger is matched, without sending commands to match each time. The CMD6 can be used to wake up) ")
    #    print ("  CMD6 : Wake up and make all commands valid ")

    #    print ("***************************** WaveShare Capacitive Fingerprint Reader Test ***************************** ")
    # print ("  1 : 등록된 지문 개수 확인")
    # print ("  2 : 지문 등록")
    # print ("  3 : 지문 매칭")
    # print ("  4 : 지문 지우기")
    # print ("  5 : HOD(핸즈온디텍션)  모드")
    # print ("  6 : wake up 모드")
    # print ("  7 : 이미지 데이터 저장")

    #t = threading.Thread(target=Auto_Verify_Finger)
    #t.setDaemon(True)
    #t.start()
    #Auto_Verify_Finger()

    # while  True:
    # option-1. manual command
    # str = input("명령어를 입력해주세요 (1-7):")

    # option-2. fixed command (save image by jiseong)
    str = "3"
    # time.sleep(2)
    Analysis_PC_Command(str)
    Finger_SleepFlag = 1
    Auto_Verify_Finger()
 #   print ("Finish")

def get_fingerprint_image():
    try:
        Analysis_PC_Command('7')
        return 1
    except:
        return 0

def delete_realtime_face_files():
    if shoot_ != 1:
        realtime_dir = dataset_dir+"/s0"
        file_list = os.listdir(realtime_dir)
        for idx, file_name in enumerate(file_list):
            os.remove(realtime_dir+"/"+file_name)

def copy_realtime_face_files(sub_file):
    target_file = dataset_dir2 + '/' + sub_file
    tosave_file = dataset_dir+"/s0/" + sub_file
    shutil.copy(target_file, tosave_file)


def camera_auth():
    global cam_ok

    if Real_Taking_a_picture(shoot_, cam_fps) == 0:
        cam_ok = 0

    else:
        result, max_idx = Data_Anomaly()
        high_score_file = str(max_idx+1)+'_realtime.pgm'

        if result == 0:
            Real_Siamese_Network(high_score_file)
        else:
            cam_ok = 0



def Print_Table(user_list):
    c_ = []
    for idx, val in enumerate(user_list):
        if len(str(val)) == 1:
            c_.append("  " + str(val))
        elif len(str(val)) == 2:
            c_.append(" " + str(val))
        elif len(str(val)) == 3:
            c_.append(str(val))

    print ("+------------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+")
    print ("| Class Name | user 1 | user 2 | user 3 | user 4 | user 5 | user 6 | user 7 | user 8 | user 9 | user10 | ",
           "  FTA ")
    print ("+------------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+")
    print ("| Classified |    " + c_[0] + " |    " + c_[1] + " |    " + c_[2] + " |    " + c_[3] + " |    " + c_[
        4] + " |    " + c_[5] + " |    " + c_[6] + " |    " + c_[7] + " |    " + c_[8] + " |    " + c_[9] + " |   ",
           FTA)
    print ("+------------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+")

# sensors (ECG & Humidity)

def serial_id_init():
    import serial.tools.list_ports
    global ser_hum, ser_ecg
    ports = serial.tools.list_ports.comports()
    for idx, port in enumerate(ports):
        if str(port.pid) == "24577": # FT232R USB UART (ECG Signal - AD8232)
            ser_ecg = serial.Serial(str(port.device), 115200)
            print ("[DETECTED] FT232R USB UART (ECG Signal - AD8232)")
        elif str(port.pid) == "67": # Arduino UNO (ECG Signal - AD8232)
            ser_ecg = serial.Serial(str(port.device), 115200)
            print ("[DETECTED] Arduino UNO (ECG Signal - AD8232)")
        else:
            pass

        if str(port.pid) == "29987": # Arduino Nano (humi - DHT22)
            ser_hum = serial.Serial(str(port.device), 115200)
            print ("[DETECTED] Arduino Nano (humi - DHT22)")
        else:
            pass

def get_hum_data():
    while 1:
        humi_data = ser_hum.readline()
        humi_str = humi_data.decode('utf-8')
        humi_str_ = humi_str.replace('%', '')
        if humi_str_ != "NULL\r\n":
            humi_float = float(humi_str_)
            return humi_float
        else:
            print ("humidity sensor value = NULL")


# ECG Authentication______________________________________________________________


def waiting_before_HOD(delay):
    global ser_hum

    fst_cnt = 0
    threshold = 0
    while 1:
        fst_cnt += 1

        #data1 = ser_ecg.readline().decode('unicode_escape')
        #data2 = ser_ecg.readline().decode('unicode_escape')

        try:
            if fst_cnt != 1:
                data1 = ser_ecg.readline().decode('unicode_escape')
                data1 = int(data1)
                #print ("DATA-1 : ", data1)
                data2 = ser_ecg.readline().decode('unicode_escape')
                data2 = int(data2)
                #print ("DATA-2 : ", data2)

                diff = abs(data1-data2)
                #print (diff)

                if diff <= 0:
                    threshold += 1
                else:
                    threshold = 0

                if (ser_hum.in_waiting > 0) or (fst_cnt==2):
                    humi = get_hum_data()
                    print ("Humidity : ", humi)
                else:
                    pass

                if threshold == 10:
                    print ("[HOD] Touch event detected on sensor !")
                    time.sleep(delay)
                    break

        except ValueError:
            pass



def save_ecg2npz(save_dir):

    fst_cnt = 0
    number = 0
    ecg_list = []

    try:
        os.makedirs(save_dir)
    except OSError:
        if not os.path.isdir(save_dir):
            raise

    ecg_data = np.empty((0,sam_freq*save_sec), int)

    while True:

        line = ser_ecg.readline().decode('unicode_escape')
        data = line.strip().split(',')
        #ecg_data = np.empty((0,sam_freq*save_sec), int)


        #if "\x00" in data[0]:
        #    pass


        # x2 ################################################
        if len(data[0]) == 0:
            pass

        else:
            fst_cnt += 1
            if fst_cnt != 1:
                ecg_list += data
                if len(ecg_list) >= (sam_freq * save_sec):
                    number += 1
                    ecg_np = np.array(list(map(float, ecg_list[:sam_freq * save_sec])))
                    ecg_np = np.reshape(ecg_np, (1, -1))

                    # high pass filter
                    if bpf_enable == 1:
                        locut = 0.5
                        hicut = 70
                        fs = 100
                        b, a = butter(3, [locut/(fs*2), hicut/(fs*2)], btype='band')
                        #b, a = butter(2, [low, high], btype='band', analog=False)  # Butterworth 필터 계수 생성
                        ecg_np = filtfilt(b, a, ecg_np)  # High Pass Filter 적용

                    else:
                        pass

                    # stack ecg data
                    if ecg_data.size == 0:
                        ecg_data = ecg_np
                    else:
                        ecg_data = np.vstack((ecg_data, ecg_np))  # 2차원 데이터를 누적시키기 위해 np.vstack() 사용

                    #print(ecg_data)
                    print('[',str(number)+'/'+str(iterated),'] appending ECG data.. | Shape :', ecg_data.shape)

                    ecg_list = ecg_list[sam_freq * save_sec:]

                    if number == iterated:
                        break
                else:
                    pass

    # y2 ################################################
    label_data = np.full((iterated,), 0)

    # fs ################################################
    fs = np.array([[100]], dtype=np.int32)

    save_dict = {
        "x2": ecg_data,
        "y2": label_data,
        "fs": fs,
    }

    # save npz file
    filename = "SN0010.npz"
    np.savez(save_dir + filename, **save_dict)
    print("saved as", filename)

#            first_cnt += 1
#            if first_cnt != 1:
#                ecg_list += data
#                if len(ecg_list) == (sam_freq * save_sec):
#                    number += 1

#                    # x2
#                    ecg_np = np.array(list(map(float, ecg_list)))

#                    ecg_np = np.expand_dims(ecg_np, axis=0)  # 데이터를 한 차원 늘림
#                    ecg_data = np.append(ecg_data, ecg_np, axis=0)  # 데이터를 누적
#                    #ecg_data = np.append(ecg_data, np.array([ecg_np]), axis=0)
#                    print (ecg_data)
#                    print (ecg_data.shape)
#                    ecg_list = []
#
#                if number == iterated:
#                    break
#            else:
#                pass


def compute_performance(cm):
    """Computer performance metrics from confusion matrix.

    It computers performance metrics from confusion matrix.
    It returns:
        - Total number of samples
        - Number of samples in each class
        - Accuracy
        - Macro-F1 score
        - Per-class precision
        - Per-class recall
        - Per-class f1-score
    """

    tp = np.diagonal(cm).astype(np.float)
    tpfp = np.sum(cm, axis=0).astype(np.float) # sum of each col
    tpfn = np.sum(cm, axis=1).astype(np.float) # sum of each row
    acc = np.sum(tp) / np.sum(cm)
    precision = tp / tpfp
    recall = tp / tpfn
    f1 = (2 * precision * recall) / (precision + recall)
    mf1 = np.mean(f1)

    total = np.sum(cm)
    n_each_class = tpfn

    return total, n_each_class, acc, mf1, precision, recall, f1

def predict(
    config_file,
    model_dir,
    output_dir,
    log_file,
    use_best=True,
    official_test_path="./data/sleephmc/5ppl_official_test",########################modified
):
    #print(time.strftime('-3 :' + '%H-%M-%S'))
    #print(official_test_path,"~~~~~!!!!!!!!!!!!!!!!!!!!!!!!!!")##############check
    spec = importlib.util.spec_from_file_location("*", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    config = config.predict

    # Create output directory for the specified fold_idx
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create logger
    logger = get_logger(log_file, level="info")
    #logger = get_logger(log_file, level="error")
    #logger = get_logger(log_file, level="critical")

    #print(time.strftime('-2 :' + '%H-%M-%S'))

    #subject_files = glob.glob(os.path.join(config["data_dir"], "*.npz"))######## original
    subject_files = glob.glob(os.path.join(official_test_path, "*.npz")) ######### modified
    # print(subject_files)

    # Load subject IDs
    fname = "{}.txt".format(config["dataset"])
    seq_sids = load_seq_ids(fname)
    #logger.info("Load generated SIDs from {}".format(fname))#######################################################################################################original
    #logger.info("SIDs ({}): {}".format(len(seq_sids), seq_sids))###################################################################################################original

    # Split training and test sets
    fold_pids = np.array_split(seq_sids, config["n_folds"])

    # Add dummy class weights
    config["class_weights"] = np.ones(config["n_classes"], dtype=np.float32)
    #print(config["class_weights"])

    trues = []
    preds = []
    preds_soft = []  ####################################################################### modified
    F1_Score = -1
    F1_Score_sleep = -1

    s_trues = []
    s_preds = []

    # jiseong
    np_softmax = np.empty((0, 5))  # np_softmax 초기화

    global preds_result_all
    preds_result_all = []

    #print(time.strftime('-1 :' + '%H-%M-%S'))
    for fold_idx in range(config["n_folds"]):

        #logger.info("------ Fold {}/{} ------".format(fold_idx+1, config["n_folds"]))##############################################################################original
        logger.info("------ Processing...{}/{} -------".format(fold_idx + 1, config["n_folds"]))################################original
        #test_sids = fold_pids[fold_idx]
        #logger.info("Test SIDs: ({}) {}".format(len(test_sids), test_sids))

        model = TinySleepNet(
            config=config,
            output_dir=os.path.join(model_dir, str(fold_idx)),
            # use_rnn=True, ## original
            use_rnn=False, ###########################################################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! rnn 미사용 ## for id
            testing=True,
            use_best=use_best,
        )


        # # Get corresponding files
        # s_trues = []
        # s_preds = []

        #for sid in test_sids:
        #logger.info("Subject ID: {}".format(fold_idx))##############################################################################################################original

        #print(time.strftime('0 :' + '%H-%M-%S'))
        #test_files = get_subject_files(
        # test_files = get_subject_files_sleepmat( ### 폴더내 모든 npz 파일 로드
        test_files=get_subject_files_sleephmc(  ### 폴더내 모든 npz 파일 로드

            dataset=config["dataset"],
            files=subject_files,
            sid=fold_idx, # no_use
        )
        #print("test files~~:")
        #print(test_files)


        #print(time.strftime('1 :' + '%H-%M-%S'))

        #for vf in test_files: logger.info("Load files {} ...".format(vf))#############################################################################################original
        #for vf in test_files: logger.info("Load files {}/{} ----------------".format((vf), len(test_files) ))###########################################################original

        # test_x, test_y, _ = load_data(test_files)##################################original ##########model에 맞게 수정필요
        # test_x, test_y, _ = load_data2(test_files) ## 수정 -> 2개의 특징 이용할때
        # test_x, test_y, _ = load_data3(test_files) ## 수정 -> 3개의 특징 이용할때
        # test_x, test_y, _ = load_data4(test_files) ## 수정 -> 4개의 특징 이용할때
        # test_x, test_y, _ = load_data_bcg(test_files)  ## 수정 -> bcg만 특징 이용할때
        test_x, test_y, _ = load_data_ecg(test_files)  ## 수정 -> ecg만 특징 이용할때####################################### for id
        # test_x, test_y, _ = load_data_multi13(test_files)  ## 수정 -> 13개의 특징 이용할때

        #print(time.strftime('2 :' + '%H-%M-%S'))



        ## Print test set
        #logger.info("Test set (n_night_sleeps={})".format(len(test_y)))###############################################################################################original
        #for _x in test_x: logger.info(_x.shape)#######################################################################################################################original
        #print_n_samples_each_class(np.hstack(test_y))#################################################################################################################original

        for night_idx, night_data in enumerate(zip(test_x, test_y)):
            # Create minibatches for testing
            night_x, night_y = night_data
            test_minibatch_fn = iterate_batch_multiple_seq_minibatches(
                [night_x],
                [night_y],
                batch_size=config["batch_size"],
                seq_length=config["seq_length"],
                shuffle_idx=None,
                augment_seq=False,
            )
            #print(time.strftime('2-1 :' + '%H-%M-%S'))
            if (config.get('augment_signal') is not None) and config['augment_signal']: ### False
                # Evaluate
                test_outs = model.evaluate_aug(test_minibatch_fn)

            else: ### True
                # Evaluate
                #print(time.strftime('2-1-1 :' + '%H-%M-%S'))
                test_outs = model.evaluate(test_minibatch_fn)
                # print("!!!!!!~~~~")
                # print(model.logits)
                ##print("!!!!!!")
                ##print(test_outs)
                # prob_outs = model.predict(test_minibatch_fn)
                # print("!!!!!!")
                # print(prob_outs)

            #print(time.strftime('2-2 :' + '%H-%M-%S'))

            s_trues.extend(test_outs["test/trues"])
            s_preds.extend(test_outs["test/preds"])
            trues.extend(test_outs["test/trues"])
            preds.extend(test_outs["test/preds"])
            preds_soft.extend(test_outs["test/softmax"])  ###################################

            # Save labels and predictions (each night of each subject)
            save_dict = {
                "y_true": test_outs["test/trues"],
                "y_pred": test_outs["test/preds"],
                "total_soft": preds_soft,####################################################
            }
            fname = os.path.basename(test_files[night_idx]).split(".")[0]
            save_path = os.path.join(
                output_dir,
                "pred_{}.npz".format(fname)
            )
            #np.savez(save_path, **save_dict)#######################################################################################################################original
            #logger.info("Saved outputs to {}".format(save_path))###################################################################################################original

        #print(time.strftime('3 :' + '%H-%M-%S'))

        # print("s_trues :", s_trues)
        # print("s_preds :", s_preds)
        s_acc = skmetrics.accuracy_score(y_true=s_trues, y_pred=s_preds)
        s_f1_score = skmetrics.f1_score(y_true=s_trues, y_pred=s_preds, average="macro")

        # s_cm = skmetrics.confusion_matrix(y_true=s_trues, y_pred=s_preds, labels=[0,1,2,3,4]) ## original ############################################################
        s_cm = skmetrics.confusion_matrix(y_true=s_trues, y_pred=s_preds, labels= range(config["n_classes"]))#####################for id

        save_dict = {
            # "y_true": test_outs["test/trues"], # original
            "y_true": s_trues,
            # "y_pred": test_outs["test/preds"], # original
            "y_pred": s_preds,
            "total_soft": preds_soft,  ####################################################
            "F1_Score": s_f1_score * 100,
            "Accuracy": s_acc * 100,
            "Confusion_Matrix(row:g_true, Col:SM)": s_cm,
        }
        fname = os.path.basename(test_files[night_idx]).split(".")[0]
        save_path = os.path.join(
            output_dir,
            "predb_{}.npz".format(fname + "_foldID_" + str(fold_idx))
            ###################################################################
        )
        np.savez(save_path, **save_dict)

        # written by jiseong (START)____________________________

        pred_result_list = []
        pred_result_lisq = []
        softmax_result = []

        for idx in range(0, iterated, 1):
            pred_result_lisq.extend(s_preds)
            pred_result = preds_soft[idx]

            for i in range(0,len(pred_result),1):
                pred_result_list.append(round(pred_result[i],6))

            print (" >> [", idx, "] PREDICT RESULT", pred_result_list)
            softmax_result.append(pred_result_list)
            pred_result_list = []

        print ("pred_result_lisq : ", pred_result_lisq)

        #print (" [PREDICT RESULT - 1]", preds_soft[0])
        #print (" [PREDICT RESULT - 2]", preds_soft[1])

        #current_datetime = datetime.datetime.now()
        #milliseconds = current_datetime.strftime("%f")[:-3]
        #softmaxfilename = current_datetime.strftime("%Y-%m-%d_%H-%M-%S.")
        #softmaxfilename = softmaxfilename + milliseconds

#        append preds_soft
        #np_softmax = np.append(np_softmax, preds_soft, axis=0)
        #np_softmax = np.concatenate((np_softmax, preds_soft), axis=0)

        # written by jiseong (END)______________________________

#######################################save_b_NPZ######################################################################
        # if( F1_Score < s_f1_score) :
        #     F1_Score = s_f1_score
        #     # Save labels and predictions (each night of each subject)
        #     save_dict = {
        #         # "y_true": test_outs["test/trues"], # original
        #         "y_true": s_trues,
        #         # "y_pred": test_outs["test/preds"], # original
        #         "y_pred": s_preds,
        #         "total_soft": preds_soft,  ####################################################
        #         "F1_Score" : s_f1_score*100,
        #         "Accuracy" : s_acc*100,
        #         "Confusion_Matrix(row:g_true, Col:SM)" : s_cm,
        #     }
        #     fname = os.path.basename(test_files[night_idx]).split(".")[0]
        #     save_path = os.path.join(
        #         output_dir,
        #         "predb_{}.npz".format(fname +"_foldID_"+ str(fold_idx)) ###################################################################
        #     )
        #     np.savez(save_path, **save_dict)
        #     if(s_f1_score>0.7) :
        #         logger.info("*****************************")

######################################################################################################################################################################original
###
        # logger.info("n={}, acc={:.1f}, mf1={:.1f}".format(
        #     len(s_preds),
        #     s_acc*100.0,
        #     s_f1_score*100.0,
        #     ##preds_soft, ##################### 시간 오래걸려 주석처리
        # ))
        #
        # logger.info(">> Confusion Matrix")
        # logger.info(s_cm)
        # ################################################################################################################
        # save_path_total_cm = os.path.join(
        #     output_dir,
        #     "Confusion_total_{}.npz".format(fold_idx)
        # )
        # #np.savez(save_path_total_cm, s_cm)############################################################################################################################
###
######################################################################################################################################################################original
       # jiseong
        preds_result_all.extend(pred_result_lisq)
        pred_result_lisq = []

        tf.reset_default_graph()
        s_trues = []  ############################################################### initialize
        s_preds = []  ############################################################### initialize
        preds_soft = []  ############################################################ initialize

        ###############################################################################################################
        save_dict_total = {
            "y_true": trues,
            "y_pred": preds,
        }
        save_path_total = os.path.join(
            output_dir,
            "pred_total.npz"
        )
        #np.savez(save_path_total, **save_dict_total)##################################################################################################################original
        ####################################################
        tf.reset_default_graph()

        #logger.info("----------------------------------")
        #logger.info("")

    #np.savez("ecg_softmax_s1/"+softmaxfilename+".npz", *np_softmax)
    #print("saved result as", "ecg_softmax_s1/"+softmaxfilename+".npz")

    ########################################################################################################

    ####################################################################################################################################################################original
    # acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
    # f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="macro")
    # cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0,1,2,3,4])
    #
    # logger.info("")
    # logger.info("=== Overall ===")
    # print_n_samples_each_class(trues)
    # logger.info("n={}, acc={:.1f}, mf1={:.1f}".format(
    #     len(preds),
    #     acc*100.0,
    #     f1_score*100.0,
    # ))
    #
    # logger.info(">> Confusion Matrix")
    # logger.info(cm)
    #
    # metrics = compute_performance(cm=cm)
    # logger.info("Total: {}".format(metrics[0]))
    # logger.info("Number of samples from each class: {}".format(metrics[1]))
    # logger.info("Accuracy: {:.1f}".format(metrics[2]*100.0))
    # logger.info("Macro F1-Score: {:.1f}".format(metrics[3]*100.0))
    # logger.info("Per-class Precision: " + " ".join(["{:.1f}".format(m*100.0) for m in metrics[4]]))
    # logger.info("Per-class Recall: " + " ".join(["{:.1f}".format(m*100.0) for m in metrics[5]]))
    # logger.info("Per-class F1-Score: " + " ".join(["{:.1f}".format(m*100.0) for m in metrics[6]]))
    #
    # # Save labels and predictions (all)
    # save_dict = {
    #     "y_true": trues,
    #     "y_pred": preds,
    #     "seq_sids": seq_sids,
    #     "config": config,
    # }
    # save_path = os.path.join(
    #     output_dir,
    #     "{}.npz".format(config["dataset"])
    # )
    # np.savez(save_path, **save_dict)
    # logger.info("Saved summary to {}".format(save_path))
    ####################################################################################################################################################################original

    # jiseong
    print (preds_result_all)
    ecg_final_id = find_mode(preds_result_all)
    print ("preds_result_all: ", preds_result_all)
    print ("[0] preds_counts", preds_result_all.count(0))
    print ("[1] preds_counts", preds_result_all.count(1))
    print ("[2] preds_counts", preds_result_all.count(2))
    print ("[3] preds_counts", preds_result_all.count(3))
    print ("[4] preds_counts", preds_result_all.count(4))

    return ecg_final_id


# find mode(id) in list(models)
def find_mode(lst):
    counter = Counter(lst)
    max_count = max(counter.values())
    modes = [num for num, count in counter.items() if count == max_count]
    return modes


def save_ecg2txt(save_dir):

    ecg_list_save = []
    fst_cnt = 0
    number = 0

    while 1:
        fst_cnt += 1
        line = ser_ecg.readline().decode('utf-8')
        #line = ser_ecg.readline().decode('unicode_escape')

        #print (line)
        #print (len(line))


        if fst_cnt == 1:
            pass

        elif len(line) >= 6:
            #print("#############################################################")
            #print("#############################################################")
            #print("#############################################################")
            #print("#############################################################")
            #print("#############################################################")
            print("#############################################################")


        else:
            data = line.strip().split(',')
            data = [val.replace('\xc0', '') for val in data]
            data = [val.replace('\xb9', '') for val in data]
            ecg_list_save += data
            #print (data)

            if len(ecg_list_save) == (sam_freq * save_sec):
                f_name = "ecg_data_" + str(number) +".txt"
                with open(save_dir + f_name, 'w', encoding='utf-8') as f:
                    f.write(','.join(str(val) for val in ecg_list_save))
                #f = open(save_dir + f_name, 'w', encoding='UTF8')
                #for idx, val in enumerate(ecg_list_save):
                #    f.write(str(val) + ",")
                #f.close()

                ecg_list_save = []
                number += 1
                print("["+str(number)+'/'+str(iterated)+"] Saved ECG data as", f_name)

                if number == iterated:
                    break

            else:
                pass



def readtxt_BPF(target_dir, save_dir, fs, locut, hicut):

    df = os.listdir(target_dir)
    txt_files = [file for file in os.listdir(target_dir) if file.endswith('.txt')]
    txt_files = [file for file in txt_files if '_hpf' not in file]

    for idx, file in enumerate(txt_files):
        data = np.genfromtxt(target_dir + file, delimiter=',')
        b, a = signal.butter(3, [locut/(fs*2), hicut/(fs*2)], btype='band')
        #b, a = butter(2, [low, high], btype='band', analog=False)  # Butterworth 필터 계수 생성
        filtered_data = signal.lfilter(b, a, data)
        filtered_data_ = filtered_data[~np.isnan(filtered_data)]
        filtered_data_list = filtered_data_.tolist()

        print(len(filtered_data_list))

        filtered_data_str = str(filtered_data_list).replace("[","")
        filtered_data_str = filtered_data_str.replace("]","")
        filtered_data_str = filtered_data_str.replace(" ","")
        filename = file.replace(".txt","_hpf.txt")

        with open(save_dir + filename, "w") as file:
            file.write(filtered_data_str)  # 문자열을 파일에 작성

        print ("saved as", filename)


        #filename = file.replace(".txt","_hpf.txt")
        #np.savetxt(save_dir + filename, filtered_data_, delimiter=',', fmt='%d')

        #print ("saved as", filename)

        # Drawing Graph
        if (idx == 1): # or (idx == 0):
            t = np.arange(len(data)) / fs
            plt.figure(figsize=(10, 4))
            plt.plot(t, data, label='Original Data')
            plt.plot(t, filtered_data, label='Filtered Data')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title('Filtered ECG Data')
            plt.legend()
            #plt.show()
            plt.show(block=False)
            plt.pause(2)
            plt.close()


def trans_txt2npz(target_dir, save_dir, ECG_plot):

    df = os.listdir(target_dir)
    txt_files = [file for file in os.listdir(target_dir) if file.endswith('.txt')]

    # hpf 이름을 포함하는 파일만 취급하도록 리스트 정리
    if bpf_enable == 1:
        txt_files = [file for file in txt_files if '_hpf' in file]
    else:
        txt_files = [file for file in txt_files if '_hpf' not in file]

    ecg_data = np.empty((0,sam_freq*save_sec), int)

    for idx, file in enumerate(txt_files):
        print (file)
        f = open(target_dir + file, 'r', encoding='utf-8')
        data = f.read()
        ecg_list = data.split(",")
        ecg_list = [x for x in ecg_list if x != '']

        print (len(ecg_list))
        print (ecg_list)

        # Drawing ECG signal graph
        if idx == 1:
            if ECG_plot == 1:
                t = np.arange(len(ecg_list))
                plt.figure(figsize=(10, 4))
                plt.plot(t, ecg_list, label='ECG Data')
                plt.xlabel('Time (s)')
                plt.gca().axes.yaxis.set_visible(False)
                plt.ylabel('Amplitude')
                plt.legend()
                #plt.show()
                plt.show(block=False)
                plt.pause(2)
                plt.close()
            else:
                pass
        #else:
        #    pass

        # x2
        ecg_np = np.array(list(map(float, ecg_list)))
        ecg_data = np.append(ecg_data, np.array([ecg_np]), axis=0)

        # y2
        label_data = np.full((iterated,), 0)

        # fs
        fs = np.array([[100]], dtype=np.int32)

        save_dict = {
            "x2": ecg_data,                       # ecg_signal
            "y2": label_data,                     # label (trash data for inference code)
            "fs": fs,                             # sampling freq
        }

        filename = "SN0010.npz"
        np.savez(save_dir + filename, **save_dict)
        print("saved as", filename)



def delete_all_txt_files(directory):
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            file_path = os.path.join(directory, file)
            os.remove(file_path)

#show_fingerprint('data/fingerprint/', 'realtime.bmp')
def show_fingerprint(save_dir, save_name):

    try:
        os.makedirs(save_dir)
    except OSError:
        if not os.path.isdir(save_dir):
            raise

    color_img = cv2.imread(save_dir + save_name, cv2.IMREAD_COLOR)
    resized_img = cv2.resize(color_img, (color_img.shape[1] * 8, color_img.shape[0] * 8))
    cv2.imshow("Title_color", resized_img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()




if __name__ == "__main__":
    if bpf_enable == 1:
        model_directory_name = 'out_id_sleephmc_5sec_230618_bandpass_std_onlysteering_d1234'
        #model_directory_name = 'out_id_sleephmc_5sec_230614_nobandpass_std_onlysteering'
    else:
        #model_directory_name = 'out_id_sleephmc_5sec_230605_nobandpass_std_onlysteering'
        #model_directory_name = 'out_id_sleephmc_5sec_230614_nobandpass_std_onlysteering'
        model_directory_name = 'out_id_sleephmc_5sec_230618_bandpass_std_onlysteering_d1234'

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="config/id_sleephmc_5sec.py")
    parser.add_argument("--model_dir", type=str, default=model_directory_name+"/train")
    parser.add_argument("--output_dir", type=str, default=model_directory_name+"/predict")
    parser.add_argument("--log_file", type=str, default=model_directory_name+"/predict.log")
    parser.add_argument("--use-best", dest="use_best", action="store_true")
    parser.add_argument("--no-use-best", dest="use_best", action="store_false")
    parser.add_argument("--official_test_path", type=str, default="./data/sleephmc/5ppl_official_test") #######
    parser.set_defaults(use_best=True)
    args = parser.parse_args()



    while 1:
        try:
            #print ("CAM:", cam_enable)
            #print ("FIN:", fin_enable)
            #print ("ECG:", ecg_enable)
            face = []
            fing = []


            # 1. Camera Authentication____________________________
            if cam_enable == 1:
                while 1:
                    print ("Face Authentication (START)")
                    camera_auth()

                    if (cam_ok == 1) or (env_enable == 1):
                        break
                    else:
                        # Door Close User Interface
                        Door_Open_Control("0")
                        color_img = cv2.imread('data/ui/door_close.png', cv2.IMREAD_COLOR)
                        cv2.imshow("Title_color", color_img)
                        cv2.waitKey(3000)
                        cv2.destroyAllWindows()


            # 2. FingerPrint Authentication____________________________
            if fin_enable == 1:
                print ("Touch your fingerprint to the sensor !")
                img_show_bool = get_fingerprint_image()

                print ("Fingerprint authentication processing..")
                finger_auth()

                if img_show_bool == 1:
                    print ("Show Fingerprint Image")
                    show_fingerprint('data/fingerprint/', 'realtime.bmp')
                else:
                    pass

            else:
                pass


            # 3. ECG Authentication____________________________
            if ecg_enable == 1:
                print("Detecting ECG serial")
                serial_id_init()
                #print ("Humidity check..")
                #humi = get_hum_data()
                #print (humi)

                if hod_enable == 1:
                    #waiting_before_HOD(5)
                    #time.sleep(4)
                    humi = get_hum_data()
                    humi = humi * 0.5
                    print (">> ECG Weight          : ", humi, "%")
                    print (">> Fingerprint Weight  : ", 100-humi, "%")

                    ser_ecg.reset_input_buffer()
                else:
                    pass

                print ("Strat ECG Authentication__________________________________")
                # read ecg data and save as csv file
                #t1 = time.time()
                #save_ecg2npz(ecg_realtime_dir)
                #t2 = time.time()
                #elapsed_time = t2-t1
                #print(f"@@@@@@@@ [ECG SAVE] processing time : {elapsed_time}")

                # save ecg data as txt files
                save_ecg2txt(ecg_realtime_dir)

                # band pass filter
                if bpf_enable == 1:
                    readtxt_BPF(ecg_realtime_dir, ecg_realtime_dir, sam_freq, locut=0.5, hicut=70)
                else:
                    pass

                # transform txt to npz file
                trans_txt2npz(ecg_realtime_dir, ecg_realtime_dir, ECG_plot=0)

                ## delete txt files
                #delete_all_txt_files(ecg_realtime_dir)

                # ECG Authentication
                #final_id = predict(
                #    config_file=args.config_file,
                #    model_dir=args.model_dir,
                #    output_dir=args.output_dir,
                #    log_file=args.log_file,
                #    use_best=args.use_best,
                #    official_test_path=args.official_test_path,#######################################################
                #)
                #print ("$$$$$$$$$$$$$$$$ ECG ID: ", final_id)
                #
                #if 0 in final_id or 1 in final_id:
                #    ecg_ok = 1
                #else:
                #    ecg_ok = 0

            else:
                pass


            ## Enable Monitor
            # print ("Enable List_________________________")
            # print ("[CAM] ", cam_enable)
            # print ("[ECG] ", ecg_enable)
            # print ("[FIN] ", fin_enable)

            #print ("Auth Result_________________________")
            #print ("[CAM] ", cam_ok)
            #print ("[ECG] ", ecg_ok)
            #print ("[FIN] ", fin_ok)

            if fin_ok == 1:

                print ("                                                                                      ")
                print (",--. ,--.                         ,--.   ,--.            ,--. ,---.,--.          ,--. ")
                print ("|  | |  | ,---.  ,---. ,--.--.     \  `.'  /,---. ,--.--.`--'/  .-'`--' ,---.  ,-|  | ")
                print ("|  | |  |(  .-' | .-. :|  .--'      \     /| .-. :|  .--',--.|  `-,,--.| .-. :' .-. | ")
                print ("'  '-'  '.-'  `)\   --.|  |          \   / \   --.|  |   |  ||  .-'|  |\   --.\ `-' | ")
                print (" `-----' `----'  `----'`--'           `-'   `----'`--'   `--'`--'  `--' `----' `---'  ")
                print ("                                                                                      ")

                Door_Open_Control("1")

                color_img = cv2.imread('data/ui/door_open.png', cv2.IMREAD_COLOR)
                cv2.imshow("Title_color", color_img)
                cv2.waitKey(3000)
                cv2.destroyAllWindows()
                #img_test = img.imread('data/ui/door_open.png')
                #plt.gca().axes.yaxis.set_visible(False)
                #plt.gca().axes.xaxis.set_visible(False)
                #plt.imshow(img_test)
                #plt.show()
                #plt.show(block=False)
                #plt.pause(2)
                #plt.close()

            else:

                print ("                                                            ")
                print (",--.  ,--.         ,--.      ,--. ,--.                      ")
                print ("|  ,'.|  | ,---. ,-'  '-.    |  | |  | ,---.  ,---. ,--.--. ")
                print ("|  |' '  || .-. |'-.  .-'    |  | |  |(  .-' | .-. :|  .--' ")
                print ("|  | `   |' '-' '  |  |      '  '-'  '.-'  `)\   --.|  |    ")
                print ("`--'  `--' `---'   `--'       `-----' `----'  `----'`--'    ")
                print ("                                                            ")

                Door_Open_Control("0")

                color_img = cv2.imread('data/ui/door_close.png', cv2.IMREAD_COLOR)
                cv2.imshow("Title_color", color_img)
                cv2.waitKey(3000)
                cv2.destroyAllWindows()
                #img_test = img.imread('data/ui/door_close.png')
                #plt.gca().axes.yaxis.set_visible(False)
                #plt.gca().axes.xaxis.set_visible(False)
                #plt.imshow(img_test)
                #plt.show()
                #plt.show(block=False)
                #plt.pause(2)
                #plt.close()


            ## Door auto close
            if door_enable == 1:
                print ("Try auto lock process")
                time.sleep(1)
                Door_Open_Control("0")
                time.sleep(1)
                Door_Open_Control("0")
                print ("CLOSED")
            else:
                pass

            # If both fingerprint and face authentication are successful, the queue is initialized. (jiseong jeong)


        except KeyboardInterrupt:
            if ser != None:
                ser.close()
            if ser_hum != None:
                ser_hum.close()
            if ser_ecg != None:
                ser_ecg.close()
            GPIO.cleanup()
            print("\n\n Test finished ! \n")
            sys.exit()






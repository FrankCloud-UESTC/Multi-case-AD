# -*- coding: utf-8 -*-
import numpy as np
import os
import sys

import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
from thop import profile
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import math
import matplotlib
from model.utils import DataLoader
import random
import glob
from tqdm import tqdm
import logging
import argparse

parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--alpha', type=float, default=0.6, help='weight for the anomality score')
parser.add_argument('--num_workers_test', type=int, default=0, help='number of workers for the test loader')
parser.add_argument('--dataset_path', type=str, default='./',
                    help='directory of data')
parser.add_argument('--model_dir', type=str, default="./model_jinan_512.pth",
                    help='directory of model')
parser.add_argument('--m_items_dir', type=str, default='./jinan_512_keys.pt', help='directory of model')
parser.add_argument('--output_dir', type=str, default='./result/', help='directory of output result images')
args = parser.parse_args()
torch.manual_seed(2023)

device = "cuda:1"
test_dataset = DataLoader(args.dataset_path, None, resize_height=args.h, resize_width=args.w, class_name="rail_80")
test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                             shuffle=False, num_workers=args.num_workers_test, drop_last=False)

# Loading the trained model
model = torch.load(args.model_dir, map_location=torch.device(device))
m_items_test = [torch.load(args.m_items_dir, map_location=torch.device(device))]

os.makedirs("result/", exist_ok=True)
model.eval()

result_out = []
print(str(len(test_batch)) + ' samples')

for k, (imgs, label, name) in tqdm(enumerate(test_batch)):
    imgs = Variable(imgs)
    if torch.cuda.is_available():
        imgs = imgs.to(device)
    hotmap, fea = model.forward(imgs, m_items_test, False, False)

    imgs = imgs[0, :, :, :]
    imgs = imgs.cpu().detach().numpy()
    imgs = np.moveaxis(imgs, 0, 2)
    imgs = (imgs + 1) * 127

    mask = hotmap[0, :, :, 0].detach().cpu().numpy()
    mask = cv2.resize(mask, (args.w, args.h))

    name = str(name[0])
    os.makedirs('./result/', exist_ok=True)
    fig_img, ax_img = plt.subplots(1, 1, figsize=(6, 6))
    ax_img.axes.xaxis.set_visible(False)
    ax_img.axes.yaxis.set_visible(False)
    ax_img.imshow(imgs[:, :, 0], cmap='gray', interpolation='none')
    ax_img.imshow(mask, cmap='jet', alpha=0.2, interpolation='none')
    fig_img.savefig(args.outupt_dir + name, bbox_inches='tight', pad_inches=0)
    plt.close()

import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
from thop import profile
from torch.autograd import Variable
import matplotlib.pyplot as plt
import math
from collections import OrderedDict
from model.utils import DataLoader
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from model.utils import *
import random
from tqdm import tqdm
import cv2 as cv
import argparse
from sklearn.cluster import KMeans

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--gpus', nargs='+', type=str, help='gpus')
parser.add_argument('--loss_compact', type=float, default=0.5, help='weight of the feature compactness loss')
parser.add_argument('--loss_separate', type=float, default=0.5, help='weight of the feature separateness loss')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--fdim', type=int, default=512, help='channel dimension of the features')
parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
parser.add_argument('--msize', type=int, default=20, help='number of the memory items')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the train loader')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs for training')
parser.add_argument('--h', type=int, default=128, help='height of input images')
parser.add_argument('--w', type=int, default=128, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')

parser.add_argument('--dataset_type', type=str, default='ECPT', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--dataset_path', type=str, default='./dataset/',
                    help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
args = parser.parse_args()
torch.manual_seed(2023)
device = "cuda:1" if torch.cuda.is_available() else "cpu"

train_folder = "./dataset/ECPT/new_training/frame_A0004_least"

print("train pic dir is: ", train_folder)
c_name = train_folder.split('/')[-1]
print(c_name)

train_dataset = DataLoader(train_folder, None, resize_height=args.h, resize_width=args.w, class_name=c_name)
train_size = len(train_dataset)
train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=True)

# # Model setting
model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim)
# model = torch.load('./exp/ECPT/log/model_0531_A15.pth', map_location='cuda:1')
params_encoder = list(model.encoder.parameters())
params_decoder = list(model.decoder.parameters())
params = params_encoder + params_decoder
optimizer = torch.optim.Adam(params, lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

if torch.cuda.is_available():
    model.to(device)

# Report the training process
log_dir = os.path.join('./exp', args.dataset_type, args.exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
loss_func_mse = nn.MSELoss(reduction='none')


# Training
# Initialize the memory items
def downMemory(m_items):
    origin_memoryBank = m_items.detach().cpu().numpy()
    memory_kmeans = KMeans(n_clusters=10, random_state=0).fit(origin_memoryBank)
    new_Memory = memory_kmeans.cluster_centers_
    returnMem = torch.tensor(new_Memory).to(device)
    return returnMem


m_items = F.normalize(torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1).to(device)

#
# m_items = torch.load(os.path.join(log_dir, '0531_A15_keys512.pt'), map_location='cuda:1')
# m_items = downMemory(m_items)

print("trained on " + args.dataset_type + "," + str(len(train_batch) * args.batch_size) + " samples.")

g_defect_memory = torch.empty(0, 512).to(device)
for epoch in range(args.epochs):
    model.train()
    loss_pixels = []
    loss_coms = []
    loss_separas = []
    mem_loss = []
    # g_defect_memory = []
    # g_defect_memory.append(torch.empty((0, 512)).to(device))
    loss_values = []
    for j, (imgs, labels, name) in tqdm(enumerate(train_batch)):
        if torch.cuda.is_available():
            imgs = Variable(imgs).to(device)
            labels = [Variable(label).to(device) for label in labels]

        outputs, pred, fea, separateness_loss, compactness_loss, m_items, g_defect_memory = \
            model.forward(imgs, m_items, True, labels[0].sum().item() > 0, labels, epoch, g_defect_memory)
        optimizer.zero_grad()

        loss_pixel = torch.mean(loss_func_mse(outputs, imgs))
        loss = compactness_loss + separateness_loss + loss_pixel
        loss.backward(retain_graph=True)
        optimizer.step()

        loss_pixels.append(loss_pixel.item())
        if compactness_loss != 0:
            loss_coms.append(compactness_loss.item())
        if separateness_loss != 0:
            loss_separas.append(separateness_loss.item())

        loss_values.append(loss.item())

    plt.figure()
    plt.plot(loss_values, label='Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    scheduler.step()
    g_defect_memory = torch.empty(0, 512).to(device)
    print('----------------------------------------')
    print('Epoch:', epoch + 1)
    print('Loss: Reconstruction {:.6f}/ Compactness {:.6f}/ Separateness {:.6f}'
          .format(np.mean(np.array(loss_pixels)),
                  np.mean(np.array(loss_coms)),
                  np.mean(np.array(loss_separas))))
    print('Memory_items:')
    print('----------------------------------------')
    print(log_dir)
    torch.save(model, os.path.join(log_dir, 'model_2024_least_A04.pth'))

print('Training is finished')
# Save the model and the memory items
torch.save(model.state_dict(), os.path.join(log_dir, 'model.pth'))
torch.save(m_items, os.path.join(log_dir, 'keys.pt'))

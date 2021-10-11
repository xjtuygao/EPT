from __future__ import print_function

# basic functions
import argparse
import os
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import cv2 

# torch functions
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import grad as torch_grad
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

#--------------------------------------------------------------------
# input arguments
parser = argparse.ArgumentParser(description='EPT')
parser.add_argument('--dataset', required=True, help='mnist | fashionmnist | cifar10')
parser.add_argument('--dataroot', required=True, help='path to dataset')

parser.add_argument('--gpuDevice', type=str, default='2', help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='input image size')

parser.add_argument('--nLoop', type=int, default=10000, help='maximum Outer Loops')
parser.add_argument('--nDiter', type=int, default=1, help='number of D update')
parser.add_argument('--nPiter', type=int, default=20, help='number of particle update')
parser.add_argument('--nProj', type=int, default=20, help='number of G projection')
parser.add_argument('--nPool', type=int, default=20, help='times of batch size for particle pool')
parser.add_argument('--nBatch', type=int, default=1, help='times of batch size for particle pool')
parser.add_argument('--period', type=int, default=100, help='period of saving ckpts') 

parser.add_argument('--coef_gp', type=float, default=5, help='coef for the gradient penalty')
parser.add_argument('--eta', type=float, default=0.5, help='learning rate for particle update')
parser.add_argument('--lrg', type=float, default=0.0001, help='learning rate for G, default=0.0001')
parser.add_argument('--lrd', type=float, default=0.0001, help='learning rate for D, default=0.0001')
parser.add_argument('--decay_g', type=bool, default=True, help='lr_g decay')
parser.add_argument('--decay_d', type=bool, default=True, help='lr_d decay')

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
parser.add_argument('--resume_loop', type=int, default=0)
parser.add_argument('--start_save', type=int, default=1000)
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpuDevice

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print('Random Seed: ', opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

train_transforms = transforms.Compose([
                   transforms.Resize(opt.imageSize),
                   transforms.ToTensor(),
                                     ])
if opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=train_transforms)
    nc = 1
    nclass = 10

elif opt.dataset == 'fashionmnist':
    dataset = dset.FashionMNIST(root=opt.dataroot, download=True,
                           transform=train_transforms)
    nc = 1
    nclass = 10

elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=train_transforms)
    nc = 3
    nclass = 10

else:
    raise NameError

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device('cuda:0')
device_cpu = torch.device('cpu')
nrow = int(math.sqrt(opt.batchSize))

# dataloader_iter = iter(dataloader)
# real_show, _ = next(dataloader_iter)
# vutils.save_image(real_show, './%s_real_img.png' % opt.dataset, nrow=10, padding=0)

particle_final = torch.FloatTensor(10, nc, opt.imageSize, opt.imageSize).to(device)
sample_similar = torch.FloatTensor(10, nc, opt.imageSize, opt.imageSize).to(device)
data_iter = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize).to(device)
img_path_mnist = "./mnist_particle_evol.png"
img_path_cf10 = "./cf10_particle_evol.png"
transform1 = transforms.Compose([transforms.ToTensor()])
if opt.dataset == 'mnist':
    img = cv2.imread(img_path_mnist, 0)
elif opt.dataset == 'cifar10':
    img = cv2.imread(img_path_cf10, 1)
    img= img[:, :, ::-1].copy()
img_tmp = transform1(img).cuda()

for j in range(10):
  particle_final[j, :, :, :] = img_tmp[:, 96:128, 32*j: 32*(j+1)].unsqueeze(0).cuda()
vutils.save_image(particle_final, './%s_particle_final.png' % opt.dataset, nrow=10, padding=0)

dataloader_iter = iter(dataloader)
for idx, data in enumerate(dataloader):
    data_iter, _ = data
    if idx == 0:
        for j in range(10):
            sample_similar[j,:,:,:] = data_iter
    for j in range(10):
        if torch.norm(particle_final[j,:,:,:] - sample_similar[j,:,:,:]) - torch.norm(particle_final[j,:,:,:] - data_iter.cuda()) > 0:
            sample_similar[j,:,:,:] = data_iter
            print("1")

vutils.save_image(sample_similar, './%s_sample_similar.png' % opt.dataset, nrow=10, padding=0)




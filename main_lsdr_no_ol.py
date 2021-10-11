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

# torch functions
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# local functions
from network_nobn_nosn import *
from resnet import * 

#--------------------------------------------------------------------
# input arguments
parser = argparse.ArgumentParser(description='EPT')
parser.add_argument('--divergence', '-div', type=str, default='Pearson', help='Pearson | KL | JS')
parser.add_argument('--dataset', required=True, help='mnist | fashionmnist | cifar10')
parser.add_argument('--dataroot', required=True, help='path to dataset')

parser.add_argument('--gpuDevice', type=str, default='2', help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='input image size')
parser.add_argument('--ndf', type=int, default=128)

parser.add_argument('--nInnerLoop', type=int, default=15000, help='maximum Inner Loops')
parser.add_argument('--nDiter', type=int, default=5, help='number of D update')
parser.add_argument('--nPool', type=int, default=100, help='times of batch size for particle pool')
parser.add_argument('--period', type=int, default=1000, help='period of saving plots') 

parser.add_argument('--eta', type=float, default=0.5, help='learning rate for particle update')
parser.add_argument('--lrd', type=float, default=0.0005, help='learning rate for D, default=0.0001')
parser.add_argument('--decay_d', type=bool, default=False, help='lrd decay')

parser.add_argument('--net', required=True, default='resnet', help='resnet')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netD', default='', help='path to netD (to continue training)')
parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
parser.add_argument('--resume_loop', type=int, default=0)
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpuDevice

try:
    os.makedirs('./particle_results')
except OSError:
    pass

try:
    os.makedirs('./particle_loss')
except OSError:
    pass

try:
    os.makedirs(os.path.join('./particle_results', opt.dataset))
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print('Random Seed: ', opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

train_transforms = transforms.Compose([
                   transforms.Resize(opt.imageSize),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5,), (0.5,)),
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

eta = float(opt.eta)
nrow = int(math.sqrt(opt.batchSize))

# nets
if opt.net == "resnet":
    netD = D_resnet(nc, opt.ndf)
elif opt.net == "info":
    netD = D_info(opt.imageSize, opt.imageSize, nc)

netD.apply(weights_init)
netD.to(device)
print('#-----------GAN initializd-----------#')

poolSize = opt.batchSize * opt.nPool

try:
    os.makedirs('./fake_particle_%s_%sk' 
                % (opt.dataset, str(poolSize/1000)))
except OSError:
    pass
    
img_real = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize).to(device)
img_fake = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize).to(device)
p_img = torch.FloatTensor(poolSize, nc, opt.imageSize, opt.imageSize).to(device_cpu)

# set optimizer
optim_D = optim.RMSprop(netD.parameters(), lr=opt.lrd)

if opt.dataset == 'mnist':
    scheduler_D = MultiStepLR(optim_D, milestones=[6000, 10000, 14000, 18000], gamma=0.5)

elif opt.dataset == 'fashionmnist':
    scheduler_D = MultiStepLR(optim_D, milestones=[6000, 10000, 14000, 18000], gamma=0.5)

elif opt.dataset == 'cifar10':
    scheduler_D = MultiStepLR(optim_D, milestones=[5000, 10000, 15000, 20000, 25000], gamma=0.5)
    
#--------------------------- main function ---------------------------#
LOSS_DR = []
GRAD_NORM = []
p_img.normal_()
idx_fix = random.sample(range(poolSize), opt.batchSize)
vutils.save_image(p_img[idx_fix] / 6 + 0.5, './particle_results/%s/particle-%s-%s-%s-%s.png' 
                  % (opt.dataset, "t_zero", opt.divergence, opt.dataset, str(opt.eta)), nrow=nrow, padding=0)

for loop in range(0, opt.nInnerLoop):  
    # input_pool
    netD.train()
    LOSS_dr = []
    for _ in range(opt.nDiter):
        # Update D
        netD.zero_grad()
        real_img, _ = next(iter(dataloader))
        img_real.copy_(real_img.to(device))
        img_b_idx = random.sample(range(poolSize), opt.batchSize)
        img_fake.copy_(p_img[img_b_idx])
        loss_dr = (netD(img_real) ** 2).mean() - 2 * netD(img_fake).mean()
        loss_dr.backward()
        optim_D.step()
        LOSS_dr.append(loss_dr.detach().cpu().item())
        # decay lr
        if opt.decay_d:
            scheduler_D.step()

    # update particle pool
    p_img_t = p_img.clone().to(device)

    p_img_t.requires_grad_(True)
    if p_img_t.grad is not None:
        p_img_t.grad.zero_()
    fake_D_score = netD(p_img_t)

    # set s(x)
    if opt.divergence == 'Pearson':
        s = torch.ones_like(fake_D_score.detach())

    elif opt.divergence == 'KL':
        s = 1 / fake_D_score.detach()

    elif opt.divergence == 'logD':
        s = 1 / (1 + fake_D_score.detach())
            
    elif opt.divergence == 'JS':
        s = 1 / (1 + fake_D_score.detach()) / fake_D_score.detach()

    else:
        raise ValueError('The divergence is not found.')

    s.unsqueeze_(1).unsqueeze_(2).unsqueeze_(3).expand_as(p_img_t)
    fake_D_score.backward(torch.ones(len(p_img_t)).to(device))
    p_img = torch.clamp(p_img - eta * s.cpu() * p_img_t.grad.cpu(), -1, 1)
    GRAD_NORM.append(p_img_t.grad.norm(p=2).detach().cpu().item())
    LOSS_DR.append(np.mean(LOSS_dr))

    if loop % 20 == 0:
        vutils.save_image(p_img[idx_fix] / 2 + 0.5, './particle_results/%s/particle-%s-%s-%s-%s.png' 
                          % (opt.dataset, str(loop).zfill(4), opt.divergence, opt.dataset, str(opt.eta)), nrow=nrow, padding=0)
        print('Inner loop(%s/%s)%d: %.4f | %.4f' 
              % (opt.divergence, opt.dataset, loop, LOSS_DR[-1], GRAD_NORM[-1]))
    
    #-----------------------------------------------------------------
    if loop % opt.period == 0:
        fig = plt.figure(figsize=(20, 20))
        plt.style.use('ggplot')
        plt.subplot(211)
        plt.plot(LOSS_DR, label=opt.divergence)
        plt.xlabel('Inner loop')
        plt.ylabel('LSDR fitting loss')

        plt.subplot(212)
        plt.plot(GRAD_NORM, label=opt.divergence)
        plt.xlabel('Inner loop')
        plt.ylabel('Gradient norm')
        fig.savefig('./particle_loss/loss-%s-%s-%s.png' % (opt.divergence, opt.dataset, str(opt.eta)))
        plt.close()
        
        idx_loop = np.arange(loop+1) + 1
        saved_loss = np.vstack((idx_loop, np.array(LOSS_DR), np.array(GRAD_NORM))).transpose()
        columns_name = ['Inner_loop', 'Loss_dr', 'Grad_norm']
        dataframe = pd.DataFrame(saved_loss, index=idx_loop, columns=columns_name)
        dataframe.to_csv('./particle_loss/saved_loss-%s-%s-%s.csv' % (opt.divergence, opt.dataset, str(opt.eta)), sep=',')
for i in range(poolSize):
    vutils.save_image(p_img[i] / 2 + 0.5, './fake_particle_%s_%sk/fake-particle-%s-%s.png' 
                        % (opt.dataset, str(poolSize/1000), i, opt.dataset), padding=0)


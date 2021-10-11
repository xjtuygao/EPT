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
from torch.autograd import grad as torch_grad
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
from utils import poolSet, inceptionScore

#--------------------------------------------------------------------
# input arguments
parser = argparse.ArgumentParser(description='EPT')
parser.add_argument('--divergence', '-div', type=str, default='KL', help='Pearson | KL | JS')
parser.add_argument('--dataset', required=True, help='mnist | fashionmnist | cifar10')
parser.add_argument('--dataroot', required=True, help='path to dataset')

parser.add_argument('--gpuDevice', type=str, default='2', help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='input image size')

parser.add_argument('--nz', type=int, default=128, help='size of the latent vector')
parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--ndf', type=int, default=128)

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

parser.add_argument('--net', required=True, default='resnet', help='resnet')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help='path to netG (to continue training)')
parser.add_argument('--netD', default='', help='path to netD (to continue training)')
parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
parser.add_argument('--resume_loop', type=int, default=0)
parser.add_argument('--start_save', type=int, default=1000)
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--increase_nProj', type=bool, default=False, help='increase the projection times')

opt = parser.parse_args()
print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpuDevice

try:
    os.makedirs('./results')
except OSError:
    pass

try:
    os.makedirs('./loss')
except OSError:
    pass

try:
    os.makedirs(os.path.join('./results', opt.dataset))
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

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
eta = float(opt.eta)
nrow = int(math.sqrt(opt.batchSize))

# nets
if opt.net == "resnet":
    netG = G_resnet(nc, ngf, nz)
    netD = D_resnet(nc, ndf)
elif opt.net == "dcgan":
    netG = G_dcgan(nc, ngf, nz)
    netD = D_dcgan(nc, ndf)
elif opt.net == "dcgan_sn":
    netG = G_dcgan_sn(nc, ngf, nz)
    netD = D_dcgan_sn(nc, ndf)

netG.apply(weights_init)
netG.to(device)
netD.apply(weights_init)
netD.to(device)
print('#-----------GAN initializd-----------#')

if opt.resume:
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    state = torch.load('./checkpoint/UPT-%s-%s-%s-%s-ckpt-gp.t7' % (opt.divergence, opt.dataset, str(opt.resume_loop), str(opt.eta)))
    netG.load_state_dict(state['netG'])
    netD.load_state_dict(state['netD'])
    start_loop = state['loop'] + 1
    is_score = state['is_score']
    best_is = state['best_is']
    loss_G = state['loss_G']
    print('#-----------Resumed from checkpoint-----------#')

else:
    start_loop = 0
    is_score = []
    best_is = 0.0

netIncept = PreActResNet18(nc)
netIncept.to(device)
netIncept = torch.nn.DataParallel(netIncept)

if torch.cuda.is_available() and not opt.cuda:
    checkpoint = torch.load('./checkpoint/resnet18-%s-ckpt.t7' % opt.dataset)
    netIncept.load_state_dict(checkpoint['net'])

else:
    checkpoint = torch.load('./checkpoint/resnet18-%s-ckpt.t7' % opt.dataset, map_location=lambda storage, loc: storage)
    netIncept.load_state_dict(checkpoint['net'])

print('#------------Classifier load finished------------#')


poolSize = opt.batchSize * opt.nPool

z_b = torch.FloatTensor(opt.batchSize, nz).to(device)
img_real = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize).to(device)
img_fake = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize).to(device)
p_z = torch.FloatTensor(poolSize, nz).to(device_cpu)
p_img = torch.FloatTensor(poolSize, nc, opt.imageSize, opt.imageSize).to(device_cpu)
show_z_b = torch.FloatTensor(opt.batchSize, nz).to(device)
eval_z_b = torch.FloatTensor(250, nz).to(device)

# set optimizer
optim_D = optim.RMSprop(netD.parameters(), lr=opt.lrd)
optim_G = optim.RMSprop(netG.parameters(), lr=opt.lrg)

if opt.dataset == 'mnist':
    scheduler_D = MultiStepLR(optim_D, milestones=[400, 800, 1200], gamma=0.5)
    scheduler_G = MultiStepLR(optim_G, milestones=[400, 800, 1200], gamma=0.5)

elif opt.dataset == 'fashionmnist':
    scheduler_D = MultiStepLR(optim_D, milestones=[400, 800, 1200], gamma=0.5)
    scheduler_G = MultiStepLR(optim_G, milestones=[400, 800, 1200], gamma=0.5)

elif opt.dataset == 'cifar10':
    scheduler_D = MultiStepLR(optim_D, milestones=[800, 1600, 2400], gamma=0.5)
    scheduler_G = MultiStepLR(optim_G, milestones=[800, 1600, 2400], gamma=0.5)

# set criterion
criterion_G = nn.MSELoss()

def gradient_penalty(real_data, D_real):
    batch_size = real_data.size()[0]
    gradients = torch_grad(outputs=D_real, inputs=real_data,
                           grad_outputs=torch.ones(D_real.size()).cuda(),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]
    gp = (gradients ** 2).view(batch_size, -1).sum(dim=1).mean(dim=0)
    return gp

#--------------------------- main function ---------------------------#
show_z_b.normal_()
dataloader_iter = iter(dataloader)
real_show, _ = next(dataloader_iter)
vutils.save_image(real_show / 2 + 0.5, './results/%s/real-%s-gp.png' % (opt.dataset, opt.dataset), nrow=nrow, padding=0)
LOSS_DR = []
LOSS_GP = []
GRAD_NORM = []
LOSS_PROJ = []

for loop in range(start_loop, start_loop + opt.nLoop):    
    # input_pool
    netD.train()
    netG.eval()
    p_z.normal_()
    with torch.no_grad():
        for i in range(opt.nPool):
            p_img[opt.batchSize*i : opt.batchSize*(i+1)] = netG(p_z[opt.batchSize*i : opt.batchSize*(i+1)].cuda()).detach()

    for t in range(opt.nPiter): 
        LOSS_dr = []
        LOSS_gp = []
        Grad_norm = []
        for _ in range(opt.nDiter):
            
            # Update D
            netD.zero_grad()
            try:
                real_img, _ = next(dataloader_iter)
            except:
                dataloader_iter = iter(dataloader)
                real_img, _ = next(dataloader_iter)
                
            img_real = real_img.to(device).clone()
            z_b_idx = random.sample(range(poolSize), opt.batchSize)
            img_fake.copy_(p_img[z_b_idx])

            img_real.requires_grad_(True)
            if img_real.grad is not None:
                img_real.grad.zero_()
            D_img_real = netD(img_real)

            loss_dr = (D_img_real ** 2).mean() - 2 * netD(img_fake).mean()
            loss_gp = opt.coef_gp * gradient_penalty(img_real, D_img_real)
            loss_dr_gp = loss_dr + loss_gp
            loss_dr_gp.backward()
            optim_D.step()
            if opt.decay_d:
                scheduler_D.step()
        LOSS_dr.append(loss_dr.detach().cpu().item())
        LOSS_gp.append(loss_gp.detach().cpu().item())

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
            
        elif opt.divergence == 'JS':
            s = 1 / (1 + fake_D_score.detach()) / fake_D_score.detach()

        else:
            raise ValueError("The divergence is not found.")

        s.unsqueeze_(1).unsqueeze_(2).unsqueeze_(3).expand_as(p_img_t)
        fake_D_score.backward(torch.ones(len(p_img_t)).to(device))
        p_img = torch.clamp(p_img - eta * s.cpu() * p_img_t.grad.cpu(), -1, 1)
        Grad_norm.append(p_img_t.grad.norm(p=2).detach().cpu().item())
    LOSS_DR.append(np.mean(LOSS_dr))
    LOSS_GP.append(np.mean(LOSS_gp))
    GRAD_NORM.append(np.mean(Grad_norm))

    # update G
    netG.train()
    netD.eval()
    poolset = poolSet(p_z, p_img)
    poolloader = torch.utils.data.DataLoader(poolset, batch_size=opt.nBatch*opt.batchSize, shuffle=True, num_workers=opt.workers)

    loss_G = []

    for _ in range(opt.nProj):

        loss_G_t = []
        for _, data_ in enumerate(poolloader, 0):
            netG.zero_grad()

            input_, target_ = data_
            pred_ = netG(input_.to(device))
            loss = criterion_G(pred_, target_.to(device))
            loss.backward()

            optim_G.step()
            if opt.decay_g:
                scheduler_G.step()
            loss_G_t.append(loss.detach().cpu().item())

        loss_G.append(np.mean(loss_G_t))
    LOSS_PROJ.append(np.mean(loss_G))   

    vutils.save_image(target_ / 2 + 0.5, './results/%s/particle-%s-%s-%s-%s-gp.png' 
                      % (opt.dataset, str(loop).zfill(4), opt.divergence, opt.dataset, str(opt.eta)), nrow=nrow, padding=0)
    print('Loop(%s/%s)%d: dr: %.4f | gp: %.4f | norm: %.4f | proj: %.4f' 
          % (opt.divergence, opt.dataset, loop, LOSS_DR[-1], LOSS_GP[-1], GRAD_NORM[-1], LOSS_PROJ[-1]))    
    #-----------------------------------------------------------------
    if loop % opt.period == 0:
        fig = plt.figure()
        plt.style.use('ggplot')
        plt.plot(loss_G, label=opt.divergence)
        plt.xlabel('Loop')
        plt.ylabel('Inner Projection Loss')
        plt.legend()
        fig.savefig('./loss/inner_projection-%s-%s-%s-gp.png' 
                    % (opt.divergence, opt.dataset, str(opt.eta)))
        plt.close()

        fig = plt.figure(figsize=(20, 20))
        plt.style.use('ggplot')
        plt.subplot(411)
        plt.plot(LOSS_DR, label=opt.divergence)
        plt.xlabel('Loop')
        plt.ylabel('DR Loss')

        plt.subplot(412)
        plt.plot(LOSS_GP, label=opt.divergence)
        plt.xlabel('Loop')
        plt.ylabel('GP Loss')

        plt.subplot(413)
        plt.plot(GRAD_NORM, label=opt.divergence)
        plt.xlabel('Loop')
        plt.ylabel('Gradient Norm')

        plt.subplot(414)
        plt.plot(LOSS_PROJ, label=opt.divergence)
        plt.xlabel('Loop')
        plt.ylabel('Projection Loss')
        fig.savefig('./loss/loss-%s-%s-%s-gp.png' 
                    % (opt.divergence, opt.dataset, str(opt.eta)))
        plt.close()

        # show image
        netG.eval()
        fake_img = netG(show_z_b)
        vutils.save_image(fake_img.detach().cpu() / 2 + 0.5, './results/%s/fake-%s-%s-%s-%s-gp.png' 
                          % (opt.dataset, str(loop).zfill(4), opt.divergence, opt.dataset, str(opt.eta)), nrow=nrow, padding=0)

        # inception score
        is_score.append(inceptionScore(netIncept, netG, device, nz, nclass))
        print('[%d] Inception Score is: %.4f' % (loop, is_score[-1]))
        best_is = max(is_score[-1], best_is)

        fig = plt.figure()
        plt.style.use('ggplot')
        plt.plot(opt.period * (np.arange(loop//opt.period + 1)), is_score, label=opt.divergence)
        plt.xlabel('Loop')
        plt.ylabel('Inception Score')
        plt.legend()
        fig.savefig('loss/IS-%s-%s-%s-gp.png' % (opt.divergence, opt.dataset, str(opt.eta)))
        plt.close()

        if best_is == is_score[-1]:
            print('Save the best Inception Score: %.4f' % is_score[-1])
        else:
            pass

    if loop > opt.start_save and loop % 100 == 0:
        state = {
            'netG': netG.state_dict(),
            'netD': netD.state_dict(),
            'is_score': is_score,
            'loss_G': loss_G,
            'loop': loop,
            'best_is': best_is
            }
        torch.save(state, './checkpoint/UPT-%s-%s-%s-%s-ckpt-gp.t7' % (opt.divergence, opt.dataset, str(loop), str(opt.eta)))

    # save IS
    if loop % 500 == 0:
        dataframe = pd.DataFrame({'IS-%s' % opt.divergence: is_score})
        dataframe.to_csv('loss/IS-%s-%s-%s-gp.csv' % (opt.divergence, opt.dataset, str(opt.eta)), sep=',')
    torch.cuda.empty_cache()


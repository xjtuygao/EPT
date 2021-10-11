# The transport map can be calculated from a source distribution to 
# a target distribution with our EPT algorithm in the 2D case. 
# This PyTorch script shows the computed transport map with or 
# without the gradient penalty and surface plots of the estimated 
# density-ratio function. It also shows the generative performance 
# and the convergence behavior of our EPT algorithm on 2D 
# examples.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.ticker import LinearLocator, FormatStrFormatter
plt.switch_backend('agg')
import seaborn as sns
import os
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import grad as torch_grad

from toy_data import toy_data_gen

parser = argparse.ArgumentParser(description='EPT')
parser.add_argument('--nloop', type=int, default=20000)
parser.add_argument('--npool', type=int, default=50000)
parser.add_argument('--nplot', type=int, default=50000)
parser.add_argument('--dataSize', type=int, default=50000)
parser.add_argument('--print_period', type=int, default=200)
parser.add_argument('--plot_period', type=int, default=200)
parser.add_argument('--gpuDevice', type=str, default='2', help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--data_target', type=str, default='large_4gaussians')
parser.add_argument('--data_source', type=str, default='small_4gaussians')
parser.add_argument('--eta', type=float, default=0.005, help='learning rate for particle update')
parser.add_argument('--coef_gp', type=float, default=0.5, help='coef for the gradient penalty')
parser.add_argument('--lrd', type=float, default=0.0005, help='learning rate for D, default=0.0005')
parser.add_argument('--nbins', type=int, default=500)
parser.add_argument('--nl', type=int, default=64, help='width of hidden layers')
parser.add_argument('--noise_target', type=bool, default=False, help='add Gaussian noise to target samples')
parser.add_argument('--noise_source', type=bool, default=False, help='add Gaussian noise to source samples')

opt = parser.parse_args()
print(opt)
npool = opt.npool
nplot = opt.nplot
dataSize = opt.dataSize
print_period = opt.print_period
plot_period = opt.plot_period
gpuDevice = opt.gpuDevice
data_target = opt.data_target
eta = opt.eta
coef_gp = opt.coef_gp
lrd = opt.lrd
nbins = opt.nbins
nl = opt.nl
nloop = opt.nloop
data_source = opt.data_source
nvisu = 1000
nmap = 200

div = 'Pearson'
batchSize = 1000
nx = 2
T = 5
LOW = -4
HIGH = 4

try:
	os.mkdir('./loss')
except:
    pass
try:
	os.mkdir('./figure')
except:
    pass
try:
	os.mkdir('./vf')
except:
    pass
try:
	os.mkdir('./figure/%s' % data_target)
except:
    pass
try:
	os.mkdir('./vf/%s' % data_target)
except:
    pass
os.environ['CUDA_VISIBLE_DEVICES'] = gpuDevice
device = torch.device('cuda:0')
# device = torch.device('cpu')

fake_X = torch.FloatTensor(batchSize, nx).to(device)
real_X = torch.FloatTensor(batchSize, nx).to(device)
Gz = torch.FloatTensor(npool, nx).to(device)
Gz_tmp = torch.FloatTensor(npool, nx).to(device)
Gz_plot = torch.FloatTensor(nplot, nx).to(device)
tar_data = torch.FloatTensor(dataSize, nx).to(device)
#--------------------- net --------------------
class D_mlp(nn.Module):
	def __init__(self, nx, nl):
		super(D_mlp, self).__init__()
		main = nn.Sequential(
			nn.Linear(nx, nl),
			nn.ReLU(True),	
			nn.Linear(nl, nl),
			nn.ReLU(True),	
			nn.Linear(nl, nl),
			nn.ReLU(True),	
			nn.Linear(nl, 1)
			)
		self.main = main

	def forward(self, x):
		out = self.main(x)
		return out.squeeze()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)

    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        # m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0.0)

def gradient_penalty(real_data, D_real):
    batch_size = real_data.size()[0]
    gradients = torch_grad(outputs=D_real, inputs=real_data,
                           grad_outputs=torch.ones(D_real.size()).to(device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]
    gp = (gradients ** 2).view(batch_size, -1).sum(dim=1).mean(dim=0)
    return gp

netD = D_mlp(nx, nl)
netD.apply(weights_init)
netD.to(device)

# set optimizer
optim_D = optim.RMSprop(netD.parameters(), lr=lrd)

# visu target
target_data = toy_data_gen(data_target, batch_size=dataSize)
if opt.noise_target:
	target_data = target_data + toy_data_gen('gaussian_noise', batch_size=dataSize)
x = target_data[:, 0]
y = target_data[:, 1]

sns.set(style="white", font_scale=1.5)
sns.despine()
plt.figure(figsize=(HIGH, HIGH))
# sns.set(style="white", color_codes=True)
sns.kdeplot(x, y, shade=True, cmap='viridis', shade_lowest=True)
plt.xticks([LOW, HIGH])
plt.yticks([LOW, HIGH])
plt.savefig("./figure/%s/kde_%s_real_targ_gp.pdf" % (data_target, data_target))
plt.close()

plt.figure(figsize=(HIGH, HIGH))
plt.scatter(x, y, s=0.1)
plt.xticks([LOW, HIGH])
plt.yticks([LOW, HIGH])
plt.savefig("./figure/%s/scatter_%s_real_targ_gp.pdf" % (data_target, data_target))
plt.close()

# plt.figure(figsize=(HIGH, HIGH))
# plt.hist2d(x,y, bins=nbins, range=[[LOW, HIGH], [LOW, HIGH]], cmap='viridis')
# plt.xticks([LOW, HIGH])
# plt.yticks([LOW, HIGH])
# plt.savefig("./figure/%s/hist_%s_real_targ_gp.pdf" % (data_target, data_target))
# plt.close()

# visu source
source_data = toy_data_gen(data_source, batch_size=dataSize)
if opt.noise_source: 
	source_data = source_data + toy_data_gen('gaussian_noise', batch_size=dataSize)
x = source_data[:, 0]
y = source_data[:, 1]

sns.set(style="white", font_scale=1.5)
sns.despine()
plt.figure(figsize=(HIGH, HIGH))
# sns.set(style="white", color_codes=True)
sns.kdeplot(x, y, shade=True, cmap='viridis', shade_lowest=True)
plt.xticks([LOW, HIGH])
plt.yticks([LOW, HIGH])
plt.savefig("./figure/%s/kde_%s_real_source_gp.pdf" % (data_target, data_target))
plt.close()

plt.figure(figsize=(HIGH, HIGH))
plt.scatter(x, y, s=0.1)
plt.xticks([LOW, HIGH])
plt.yticks([LOW, HIGH])
plt.savefig("./figure/%s/scatter_%s_real_source_gp.pdf" % (data_target, data_target))
plt.close()

# source and evolution
tar_data = torch.from_numpy(target_data).float().to(device)
Gz = torch.from_numpy(source_data).float().to(device)

LOSS_DR = []
LOSS_GP = []
Grad_norm = []

idx_visu = random.sample(range(npool), nvisu)
z_visu = Gz[idx_visu].cpu().numpy()
idx_map = random.sample(range(nvisu), nmap)
z_map = z_visu[idx_map]

Y, X = np.mgrid[LOW : HIGH+1 : 0.25, LOW : HIGH+1 : 0.25]
size = X.shape[1]
grad_X = np.zeros((size, size))
grad_Y = np.zeros((size, size))
mesh_DR = np.zeros((size, size))
#-------------------- main ---------------------
for loop in range(nloop):
	loss_iter_dr = []
	loss_iter_gp = []
	for t in range(T):
		# update D
		netD.train()
		netD.zero_grad()
		real_idx = random.sample(range(dataSize), batchSize)
		real_X = tar_data[real_idx].clone()
		fake_idx = random.sample(range(npool), batchSize)
		fake_X = Gz[fake_idx].clone()

		real_X.requires_grad_(True)
		if real_X.grad is not None:
			real_X.grad.zero_()
		D_real_X = netD(real_X)
		
		loss_dr = (D_real_X ** 2).mean() - 2 * netD(fake_X).mean() 
		loss_gp = coef_gp * gradient_penalty(real_X, D_real_X)
		loss_dr_gp = loss_dr + loss_gp
		loss_dr_gp.backward()
		optim_D.step()
		loss_iter_dr.append(loss_dr.detach().cpu().item())
		loss_iter_gp.append(loss_gp.detach().cpu().item())
	LOSS_DR.append(np.mean(loss_iter_dr))
	LOSS_GP.append(np.mean(loss_iter_gp))

	# update particles
	Gz_tmp = Gz.clone()
	Gz_tmp.requires_grad_(True)
	if Gz_tmp.grad is not None:
		Gz_tmp.grad.zero_()
	Gz_D_score = netD(Gz_tmp)

    # set s(x)
	if div == 'Pearson':
		s = torch.ones_like(Gz_D_score.detach())
	else:
		raise Exception("The divergence is not found.")

	s.unsqueeze_(1).expand_as(Gz).to(device)
	Gz_D_score.backward(torch.ones(len(Gz)).to(device))
	Gz = Gz - eta * s * Gz_tmp.grad
	Grad_norm.append(Gz_tmp.grad.norm(p=2).detach().cpu().item())

	if (loop+1) % print_period == 0 or loop == 0:
		print('loop(%s/%s)%d: %.4f | %.4f | %.4f' 
			  % (div, data_target, loop, LOSS_DR[-1], LOSS_GP[-1], Grad_norm[-1]))

		# plot vector field
		mesh_Y = torch.from_numpy(Y.reshape((-1, 1))).float().to(device)
		mesh_X = torch.from_numpy(X.reshape((-1, 1))).float().to(device)
		mesh = torch.cat((mesh_X, mesh_Y), 1)

		mesh.requires_grad_(True)
		if mesh.grad is not None:
			mesh.grad.zero_()
		mesh_D_score = netD(mesh)
		mesh_D_score.backward(torch.ones(len(mesh)).to(device))
		mesh_grad = - mesh.grad.detach().cpu().numpy()
		grad_X = grad_X + mesh_grad[:, 0].reshape((size, size))
		grad_Y = grad_Y + mesh_grad[:, 1].reshape((size, size))
		# grad_X = mesh_grad[:, 0].reshape((size, size))
		# grad_Y = mesh_grad[:, 1].reshape((size, size))
		mesh_DR = mesh_D_score.detach().cpu().numpy().reshape((size, size))
		
		# vector field
		fig, ax = plt.subplots(figsize=(10, 10))
		color = 2 * np.log(np.hypot(grad_X, grad_Y))
		q = ax.streamplot(X, Y, grad_X, grad_Y, color=color, linewidth=0.8, cmap='viridis',
              			  density=2, arrowstyle='->', arrowsize=1.0)
		plt.xlim(LOW,HIGH)
		plt.ylim(LOW,HIGH)
		plt.savefig('./vf/%s/gf_%s_%s_%s_gp.pdf' % (data_target, data_target, div, loop), bbox_inches='tight')
		plt.close()

		# surface
		fig = plt.figure(figsize=(10, 8))
		ax = plt.axes(projection='3d')
		surf = ax.plot_surface(X, Y, mesh_DR, cmap='viridis',
								 linewidth=0, antialiased=False)
		ax.set_zlim(0.8, 1.2)
		ax.zaxis.set_major_locator(LinearLocator(5))
		ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
		fig.colorbar(surf, shrink=1, aspect=1)
		ax.set_xlim(LOW,HIGH)
		ax.set_ylim(LOW,HIGH)
		plt.savefig('./vf/%s/3d_dr_%s_%s_%s_gp.pdf' % (data_target, data_target, div, loop), bbox_inches='tight')
		plt.close()


	if (loop+1) % plot_period == 0 or loop == 0:
		Gz_plot = Gz.detach().cpu().numpy()
		Gz_visu = Gz[idx_visu].detach().cpu().numpy()
		Gz_map = Gz_visu[idx_map]

		# plt.figure(figsize=(HIGH, HIGH))
		# sns.kdeplot(Gz_plot[:, 0], Gz_plot[:, 1], shade=True, cmap='viridis', shade_lowest=True)
		# plt.xticks([LOW, HIGH])
		# plt.yticks([LOW, HIGH])
		# plt.savefig("./figure/%s/kde_%s_%s_%d_gp.pdf" % (data_target, data_target, div, loop))
		# plt.close()

		# plt.figure(figsize=(HIGH, HIGH))
		# plt.scatter(Gz_plot[:, 0], Gz_plot[:, 1], s=0.1)
		# plt.xticks([LOW, HIGH])
		# plt.yticks([LOW, HIGH])
		# plt.savefig("./figure/%s/scatter_%s_%s_%d_gp.pdf" % (data_target, data_target, div, loop))
		# plt.close()

		# plt.figure(figsize=(HIGH, HIGH))
		# plt.hist2d(Gz_plot[:, 0], Gz_plot[:, 1], bins=nbins, range=[[LOW, HIGH], [LOW, HIGH]], cmap='viridis')
		# plt.xticks([LOW, HIGH])
		# plt.yticks([LOW, HIGH])		
		# plt.savefig("./figure/%s/hist_%s_%s_loop_%d_gp.pdf" % (data_target, data_target, div, loop))
		# plt.close()

		plt.figure(figsize=(HIGH, HIGH))
		plt.scatter(z_visu[:, 0], z_visu[:, 1], s=0.1, c='tab:green')
		plt.scatter(Gz_visu[:, 0], Gz_visu[:, 1], s=0.1, c='tab:pink')
		plt.plot([z_map[:, 0], Gz_map[:, 0]], [z_map[:, 1], Gz_map[:, 1]], 
				 color='tab:gray', marker='o', linestyle='-', linewidth=0.3, markersize=1)
		plt.xticks([LOW, HIGH])
		plt.yticks([LOW, HIGH])
		plt.savefig("./vf/%s/visu_map_%s_%s_%d_gp.pdf" % (data_target, data_target, div, loop))
		plt.close()

		#----------------------------------
		idx_loop = np.arange(loop+1) + 1
		fig = plt.figure()
		plt.plot(idx_loop, LOSS_DR, label='Loss_dr')
		plt.xlabel('loop')
		plt.ylabel('Loss_dr')
		plt.legend()
		fig.savefig('./loss/Loss_dr-%s_gp.png' % data_target)
		plt.close()

		fig = plt.figure()
		plt.plot(idx_loop, LOSS_GP, label='Loss_gp')
		plt.xlabel('loop')
		plt.ylabel('Loss_gp')
		plt.legend()
		fig.savefig('./loss/Loss_gp-%s_gp.png' % data_target)
		plt.close()

		fig = plt.figure()
		plt.plot(idx_loop, Grad_norm, label='Grad_norm')
		plt.xlabel('loop')
		plt.ylabel('Grad_norm')
		plt.legend()
		fig.savefig('./loss/Grad_norm-%s_gp.png' % data_target)
		plt.close()

		saved_loss = np.vstack((idx_loop, np.array(LOSS_DR), np.array(LOSS_GP), np.array(Grad_norm))).transpose()
		columns_name = ['Loop', 'Loss_dr', 'Loss_gp', 'Grad_norm']
		dataframe = pd.DataFrame(saved_loss, index=idx_loop, columns=columns_name)
		dataframe.to_csv('saved_loss-%s_gp.csv' % data_target, sep=',')




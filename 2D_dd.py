import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import Normalize
plt.switch_backend('agg')
import seaborn as sns
import os
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from toy_data import toy_data_gen

parser = argparse.ArgumentParser(description='EPT')
parser.add_argument('--nloop', type=int, default=20000)
parser.add_argument('--npool', type=int, default=50000)
parser.add_argument('--nplot', type=int, default=50000)
parser.add_argument('--dataSize', type=int, default=50000)
parser.add_argument('--print_period', type=int, default=200)
parser.add_argument('--plot_period', type=int, default=1000)
parser.add_argument('--gpuDevice', type=str, default='0', help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--data_target', type=str, default='large_4gaussians')
parser.add_argument('--data_source', type=str, default='small_4gaussians')
parser.add_argument('--eta', type=float, default=0.005, help='learning rate for particle update')
parser.add_argument('--lrd', type=float, default=0.0005, help='learning rate for D, default=0.0005')
parser.add_argument('--nbins', type=int, default=500)
parser.add_argument('--nl', type=int, default=64, help='width of hidden layers')
parser.add_argument('--base_std', type=int, default=20, help='standard deviation of the base Guassian')
parser.add_argument('--coef_dd', type=float, default=5, help='coef_dd')

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
lrd = opt.lrd
nbins = opt.nbins
nl = opt.nl
nloop = opt.nloop
data_source = opt.data_source
nvisu = 1000
nmap = 200
base_mean = 0
base_std = opt.base_std

div = 'l2'
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
	os.mkdir('./figure/%s' % data)
except:
    pass
try:
	os.mkdir('./vf/%s' % data)
except:
    pass
os.environ['CUDA_VISIBLE_DEVICES'] = gpuDevice
device = torch.device('cuda:0')
# device = torch.device('cpu')

fake_X = torch.FloatTensor(batchSize, nx).to(device)
real_X = torch.FloatTensor(batchSize, nx).to(device)
base_X = torch.FloatTensor(batchSize, nx).to(device)
Gz = torch.FloatTensor(npool, nx).to(device)
Gz_tmp = torch.FloatTensor(npool, nx).to(device)
Gz_plot = torch.FloatTensor(nplot, nx).to(device)
target_data = torch.FloatTensor(dataSize, nx).to(device)
zeros = torch.zeros(1).to(device)
torch.pi = torch.acos(torch.zeros(1)) * 2

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
			nn.Linear(nl, 1),
			# nn.Tanh(),
			)
		self.main = main
		self.coef_leaky = 0.01
		self.alpha = 10
		self.zeros = zeros
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
        m.bias.data.fill_(0.0)

def normal_pdf(x, mean, std):
	return 1 / (torch.sqrt(2*torch.pi) * std) * torch.exp(-((x-mean)**2).mean(-1) / (2 * std**2))
def normal_sample(x, mean, std):
	return x.normal_(mean=mean, std=std)
	
netD = D_mlp(nx, nl)
netD.apply(weights_init)
netD.to(device)

# set optimizer
optim_D = optim.RMSprop(netD.parameters(), lr=lrd)

s = toy_data_gen(data, batch_size=nplot)
x = s[:, 0]
y = s[:, 1]

sns.set(style="white", font_scale=1.5)
sns.despine()
plt.figure(figsize=(HIGH, HIGH))
# sns.set(style="white", color_codes=True)
sns.kdeplot(x, y, shade=True, cmap='viridis', shade_lowest=True)
plt.xticks([LOW, HIGH])
plt.yticks([LOW, HIGH])
plt.savefig("./figure/%s/kde_%s_real_targ-dd.pdf" % (data, data))
plt.close()

plt.figure(figsize=(HIGH, HIGH))
plt.scatter(x, y, s=0.1)
plt.xticks([LOW, HIGH])
plt.yticks([LOW, HIGH])
plt.savefig("./figure/%s/scatter_%s_real_targ-dd.pdf" % (data, data))
plt.close()

# plt.figure(figsize=(HIGH, HIGH))
# plt.hist2d(x,y, bins=nbins, range=[[LOW, HIGH], [LOW, HIGH]], cmap='viridis')
# plt.xticks([LOW, HIGH])
# plt.yticks([LOW, HIGH])
# plt.savefig("./figure/%s/hist_%s_real_targ-dd.pdf" % (data, data))
# plt.close()

s = toy_data_gen(data_source, batch_size=nplot)
x = s[:, 0]
y = s[:, 1]

sns.set(style="white", font_scale=1.5)
sns.despine()
plt.figure(figsize=(HIGH, HIGH))
# sns.set(style="white", color_codes=True)
sns.kdeplot(x, y, shade=True, cmap='viridis', shade_lowest=True)
plt.xticks([LOW, HIGH])
plt.yticks([LOW, HIGH])
plt.savefig("./figure/%s/kde_%s_real_ref-dd.pdf" % (data, data))
plt.close()

plt.figure(figsize=(HIGH, HIGH))
plt.scatter(x, y, s=0.1)
plt.xticks([LOW, HIGH])
plt.yticks([LOW, HIGH])
plt.savefig("./figure/%s/scatter_%s_real_ref-dd.pdf" % (data, data))
plt.close()

Gz = torch.from_numpy(toy_data_gen(data_source, batch_size=dataSize)).float().to(device)
target_data = torch.from_numpy(toy_data_gen(data_target, batch_size=dataSize)).float().to(device)

LOSS_DD = []
Grad_norm = []

idx_visu = random.sample(range(npool), nvisu)
z_visu = Gz[idx_visu].cpu().numpy()
idx_map = random.sample(range(nvisu), nmap)
z_map = z_visu[idx_map]

Y, X = np.mgrid[LOW : HIGH+1 : 0.25, LOW : HIGH+1 : 0.25]
size = X.shape[1]
grad_X = np.zeros((size, size))
grad_Y = np.zeros((size, size))
mesh_DD = np.zeros((size, size))
#-------------------- main ---------------------
for loop in range(nloop):
	loss_iter_dd = []
	for t in range(T):
		# update D
		netD.train()
		netD.zero_grad()
		real_idx = random.sample(range(dataSize), batchSize)
		real_X.copy_(target_data[real_idx])
		fake_idx = random.sample(range(npool), batchSize)
		fake_X.copy_(Gz[fake_idx])
		loss_dd = netD(real_X).mean() - netD(fake_X).mean() + opt.coef_dd * (netD(normal_sample(base_X, base_mean, base_std)) ** 2).mean()
		loss_dd.backward()
		optim_D.step()
		loss_iter_dd.append(loss_dd.detach().cpu().item())
	LOSS_DD.append(np.mean(loss_iter_dd))

	# update particles
	Gz_tmp = Gz.clone()
	Gz_tmp.requires_grad_(True)
	if Gz_tmp.grad is not None:
		Gz_tmp.grad.zero_()
	Gz_D_score = netD(Gz_tmp)

    # set s(x)
	if div == 'l2':
		s = torch.ones_like(Gz_D_score.detach())
	else:
		raise Exception("The divergence is not found.")

	s.unsqueeze_(1).expand_as(Gz).to(device)
	Gz_D_score.backward(torch.ones(len(Gz)).to(device))
	Gz = Gz - eta * s * Gz_tmp.grad
	Grad_norm.append(Gz_tmp.grad.norm(p=2).detach().cpu().item())

	if (loop+1) % print_period == 0 or loop == 0:
		print('loop(%s/%s)%d: %.4f | %.4f' 
			  % (div, data, loop, LOSS_DD[-1], Grad_norm[-1]))

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
		mesh_DD = mesh_D_score.detach().cpu().numpy().reshape((size, size))
		
		# vector field
		fig, ax = plt.subplots(figsize=(10, 10))
		color = 2 * np.log(np.hypot(grad_X, grad_Y))
		q = ax.streamplot(X, Y, grad_X, grad_Y, color=color, linewidth=0.8, cmap='viridis',
              			  density=2, arrowstyle='->', arrowsize=1.0)
		plt.xlim(LOW,HIGH)
		plt.ylim(LOW,HIGH)
		plt.savefig('./vf/%s/gf_%s_%s_%s-dd.pdf' % (data, data, div, loop), bbox_inches='tight')
		plt.close()

		# surface
		fig = plt.figure(figsize=(10, 8))
		ax = plt.axes(projection='3d')
		surf = ax.plot_surface(X, Y, mesh_DD, cmap='viridis',
								 linewidth=0, antialiased=False)
		ax.set_zlim(-1, 1)
		ax.zaxis.set_major_locator(LinearLocator(5))
		ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
		fig.colorbar(surf, shrink=1, aspect=1)
		ax.set_xlim(LOW,HIGH)
		ax.set_ylim(LOW,HIGH)
		plt.savefig('./vf/%s/3d_dd_%s_%s_%s-dd.pdf' % (data, data, div, loop), bbox_inches='tight')
		plt.close()

	if (loop+1) % plot_period == 0 or loop == 0:
		Gz_plot = Gz.detach().cpu().numpy()
		Gz_visu = Gz[idx_visu].detach().cpu().numpy()
		Gz_map = Gz_visu[idx_map]

		plt.figure(figsize=(HIGH, HIGH))
		sns.kdeplot(Gz_plot[:, 0], Gz_plot[:, 1], shade=True, cmap='viridis', shade_lowest=True)
		plt.xticks([LOW, HIGH])
		plt.yticks([LOW, HIGH])
		plt.savefig("./figure/%s/kde_%s_%s_%d-dd.pdf" % (data, data, div, loop))
		plt.close()

		plt.figure(figsize=(HIGH, HIGH))
		plt.scatter(Gz_plot[:, 0], Gz_plot[:, 1], s=0.1)
		plt.xticks([LOW, HIGH])
		plt.yticks([LOW, HIGH])
		plt.savefig("./figure/%s/scatter_%s_%s_%d-dd.pdf" % (data, data, div, loop))
		plt.close()

		plt.figure(figsize=(HIGH, HIGH))
		plt.scatter(z_visu[:, 0], z_visu[:, 1], s=0.1, c='tab:green')
		plt.scatter(Gz_visu[:, 0], Gz_visu[:, 1], s=0.1, c='tab:pink')
		plt.plot([z_map[:, 0], Gz_map[:, 0]], [z_map[:, 1], Gz_map[:, 1]], 
				 color='tab:gray', marker='o', linestyle='-', linewidth=0.3, markersize=1)
		plt.xticks([LOW, HIGH])
		plt.yticks([LOW, HIGH])
		plt.savefig("./vf/%s/visu_map_%s_%s_%d-dd.pdf" % (data, data, div, loop))
		plt.close()

		#----------------------------------
		idx_loop = np.arange(loop+1) + 1
		fig = plt.figure()
		plt.plot(idx_loop, LOSS_DD, label='Loss_dd')
		plt.xlabel('loop')
		plt.ylabel('Loss_dd')
		plt.legend()
		fig.savefig('./loss/Loss_dd-%s-dd.png' % data)
		plt.close()

		fig = plt.figure()
		plt.plot(idx_loop, Grad_norm, label='Grad_norm')
		plt.xlabel('loop')
		plt.ylabel('Grad_norm')
		plt.legend()
		fig.savefig('./loss/Grad_norm-%s-dd.png' % data)
		plt.close()

		saved_loss = np.vstack((idx_loop, np.array(LOSS_DD), np.array(Grad_norm))).transpose()
		columns_name = ['Loop', 'Loss_dd', 'Grad_norm']
		dataframe = pd.DataFrame(saved_loss, index=idx_loop, columns=columns_name)
		dataframe.to_csv('saved_loss-%s-dd.csv' % data, sep=',')




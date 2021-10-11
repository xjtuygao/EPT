import argparse
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import cv2

parser = argparse.ArgumentParser(description='img_merge')
parser.add_argument('--dataset', required=True, help='mnist | fashionmnist | cifar10')
opt = parser.parse_args()
print(opt)

img_path_mnist = "./mnist_particle_evol.png"
img_path_cifar10 = "./cf10_particle_evol.png"
path_similar_mnist = "./mnist_sample_similar.png"
path_similar_cifar10 = "./cifar10_sample_similar.png"

if opt.dataset == 'mnist':
    img_evol = cv2.imread(img_path_mnist, 0)
    img_simi = cv2.imread(path_similar_mnist, 0)
elif opt.dataset == 'cifar10':
    img_evol = cv2.imread(img_path_cifar10, 1)
    img_simi = cv2.imread(path_similar_cifar10, 1)
img_merge = np.vstack([img_evol, img_simi])
cv2.imwrite("./%s_particle_evol_new.png" % opt.dataset, img_merge)
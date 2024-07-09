""" Scripts for training and evaluating various meta-learning algorithms for MLP. """

import numpy as np
import pickle
import torch
from torch import nn

from mlp_meta_tools import train_GD, train_meta_opt, test_meta_opt
from plot import plot_all

sigma = nn.LeakyReLU(0.1)
sigma_inv = nn.LeakyReLU(10)

# Example: [4, 5, 6, 7, 8] -> X: 5x4, W1:6x5, W2:7x6, W3:8x7, Y:8x4
dim = [20, 20, 20, 20] 

# training with GD
epoch = 200
lr = 1e-4
time_GD, loss_GD, dLdt_GD = \
    train_GD(dim, n_run=5, n_epoch=epoch, lr=1e-5)

# training with GD w/ teleportation
epoch = 200
lr = 1e-4
time_GDtel, loss_GDtel, dLdt_GDtel = \
    train_GD(dim, n_run=5, n_epoch=epoch, K=[10], teleport=False, lr=1e-5, lr_teleport=1e-7)

meta_opt_list, meta_opt_update = train_meta_opt(dim, n_run=50, n_epoch=100, unroll=10, lr=lr, lr_meta=1e-3, learn_lr=True, learn_tele=False, learn_update=False)
time_lr, loss_lr, dLdt_lr, lr_lr = \
    test_meta_opt(meta_opt_list, meta_opt_update, dim, n_run=5, n_epoch=100, learn_lr=True, learn_tele=False, learn_update=False)

meta_opt_list, meta_opt_update = train_meta_opt(dim, n_run=50, n_epoch=100, unroll=10, lr=lr, lr_meta=1e-3, learn_lr=True, learn_tele=True, learn_update=True)
time_update, loss_update, dLdt_update, lr_update = \
    test_meta_opt(meta_opt_list, meta_opt_update, dim, n_run=5, n_epoch=100, learn_lr=True, learn_tele=True, learn_update=True)

plot_all([time_GD, time_GDtel, time_lr, time_update], \
    [loss_GD, loss_GDtel, loss_lr, loss_update], \
    [dLdt_GD, dLdt_GDtel, dLdt_lr, dLdt_update], \
    ['GD', 'GD w tel', "lr only", "update only"], n_epoch=100)

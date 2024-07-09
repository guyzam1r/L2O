""" Scripts for training and evaluating various meta-learning algorithms for MLP. """

import numpy as np
import pickle
import torch
from torch import nn

from mlp_meta_tools import train_GD, train_meta_opt, test_meta_opt
from plot import plot_all

sigma = nn.LeakyReLU(0.1)
sigma_inv = nn.LeakyReLU(10)
dim = [20, 20, 20, 20]
lr = 1e-4
train_runs = 500
teleport = False

# equal weights
meta_opt_list, meta_opt_update = train_meta_opt(dim, n_run=train_runs, n_epoch=100, unroll=10, lr=lr, lr_meta=1e-3, learn_tele=teleport, exp_weights=-1)
time_equal, loss_equal, dLdt_equal, lr_equal = \
    test_meta_opt(meta_opt_list, meta_opt_update, dim, n_run=5, n_epoch=100, learn_tele=teleport, T_magnitude=0.01)

# w = 0.1
meta_opt_list, meta_opt_update = train_meta_opt(dim, n_run=train_runs, n_epoch=100, unroll=10, lr=lr, lr_meta=1e-3, learn_tele=teleport, exp_weights=0.1)
time_0p1, loss_0p1, dLdt_0p1, lr_0p1 = \
    test_meta_opt(meta_opt_list, meta_opt_update, dim, n_run=5, n_epoch=100, learn_tele=teleport, T_magnitude=0.01)

# w = 0.5
meta_opt_list, meta_opt_update = train_meta_opt(dim, n_run=train_runs, n_epoch=100, unroll=10, lr=lr, lr_meta=1e-3, learn_tele=teleport, exp_weights=0.5)
time_0p5, loss_0p5, dLdt_0p5, lr_0p5 = \
    test_meta_opt(meta_opt_list, meta_opt_update, dim, n_run=5, n_epoch=100, learn_tele=teleport, T_magnitude=0.01)

# w = 0.9
meta_opt_list, meta_opt_update = train_meta_opt(dim, n_run=train_runs, n_epoch=100, unroll=10, lr=lr, lr_meta=1e-3, learn_tele=teleport, exp_weights=0.9)
time_0p9, loss_0p9, dLdt_0p9, lr_0p9 = \
    test_meta_opt(meta_opt_list, meta_opt_update, dim, n_run=5, n_epoch=100, learn_tele=teleport, T_magnitude=0.01)

# w = 1.5
meta_opt_list, meta_opt_update = train_meta_opt(dim, n_run=train_runs, n_epoch=100, unroll=10, lr=lr, lr_meta=1e-3, learn_tele=teleport, exp_weights=1.5)
time_1p5, loss_1p5, dLdt_1p5, lr_1p5 = \
    test_meta_opt(meta_opt_list, meta_opt_update, dim, n_run=5, n_epoch=100, learn_tele=teleport, T_magnitude=0.01)


plot_all([time_equal, time_0p1, time_0p5, time_0p9, time_1p5], \
    [loss_equal, loss_0p1, loss_0p5, loss_0p9, loss_1p5], \
    [dLdt_equal, dLdt_0p1, dLdt_0p5, dLdt_0p9, dLdt_1p5], \
    ['Equal Weights', 'w=0.1', 'w=0.5', 'w=0.9', 'w=1.5'], n_epoch=100, xticks=[0,30,60,90], loc='weight_figs')
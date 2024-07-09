""" Scripts for training and evaluating various meta-learning algorithms for MLP. """

import numpy as np
import torch
from torch import nn

from mlp_meta_tools import train_GD, train_meta_opt, test_meta_opt
from plot import plot_all

sigma = nn.LeakyReLU(0.1)
sigma_inv = nn.LeakyReLU(10)
dim = [10, 10, 10, 10] 
lr = 1e-4
train_runs = 400
teleport = False

# train an lstm that takes only the gradient as input
meta_opt_list, meta_opt_update = train_meta_opt(dim, n_run=train_runs, n_epoch=100, unroll=10, lr=lr, lr_meta=1e-3, learn_tele=teleport)
time, loss, dLdt, lr = \
    test_meta_opt(meta_opt_list, meta_opt_update, dim, n_run=5, n_epoch=100, learn_tele=teleport, T_magnitude=0.01)

# train an lstm that takes the gradient and iterates as input
meta_opt_list, meta_opt_update = train_meta_opt(dim, n_run=train_runs, n_epoch=100, unroll=10, lr=lr, lr_meta=1e-3, learn_tele=teleport, input_iterates=True)
time_iter, loss_iter, dLdt_iter, lr_iter = \
    test_meta_opt(meta_opt_list, meta_opt_update, dim, n_run=5, n_epoch=100, learn_tele=teleport, input_iterates=True, T_magnitude=0.01)

# train an lstm that takes the gradient and the update step as input
meta_opt_list, meta_opt_update = train_meta_opt(dim, n_run=train_runs, n_epoch=100, unroll=10, lr=lr, lr_meta=1e-3, learn_tele=teleport, input_update=True)
time_update, loss_update, dLdt_update, lr_update = \
    test_meta_opt(meta_opt_list, meta_opt_update, dim, n_run=5, n_epoch=100, learn_tele=teleport, input_update=True, T_magnitude=0.01)

# train an lstm that takes the gradient, iterates, and update step as input
meta_opt_list, meta_opt_update = train_meta_opt(dim, n_run=train_runs, n_epoch=100, unroll=10, lr=lr, lr_meta=1e-3, learn_tele=teleport, input_iterates=True, input_update=True)
time_both, loss_both, dLdt_both, lr_both = \
    test_meta_opt(meta_opt_list, meta_opt_update, dim, n_run=5, n_epoch=100, learn_tele=teleport, input_iterates=True, input_update=True, T_magnitude=0.01)

plot_all([time, time_iter, time_update, time_both], \
    [loss, loss_iter, loss_update, loss_both], \
    [dLdt, dLdt_iter, dLdt_update, dLdt_both], \
    ['Gradient Only', 'Grad + Iters', 'Grad + Updates', 'Grad + Iters + Updates'], n_epoch=80, xticks=[0,20,40,60,80], loc='input_figs')
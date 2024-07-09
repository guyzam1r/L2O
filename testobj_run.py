import numpy as np
import torch
from torch import nn

from testobj_meta_tools import train_meta_opt, test_opts, plot_level_sets, init_obj
from first_order_optims import GD_Opt, Momentum_Opt, Adam_Opt
from plot import plot_all
from matplotlib import pyplot as plt

train_runs = 800
test_epochs = 10

#ellipsoid objective function
A = torch.tensor([[0.3, 2.0],[0.6, 1.0]])
b = torch.tensor([-7.0, -5.0])
h, h_inv = init_obj("elli", [A,b])

#rosenbrock objective function
#a = 1
#b = -1
#f = lambda x : -3
#g = lambda x : 0.4*(x**2)#+5*torch.sin(x)
#h, h_inv = init_obj("ros", [a,b,f,g])

meta_opt = train_meta_opt(n_run=train_runs, learn_tele=True, update_first=True, safeguard=False)
meta_opt_gate = train_meta_opt(n_run=train_runs, learn_tele=True, update_first=True, learn_tele_gate=True, safeguard=False)
meta_opt_notele = train_meta_opt(n_run=train_runs)

opts = [meta_opt, meta_opt_gate, meta_opt_notele, Adam_Opt(lr=0.5)]
x_init = [-5.0, 5.0]

times, losses, grad_norms, x_arrs, y_arrs, x_tele_arrs, y_tele_arrs, all_x_arrs, all_y_arrs = \
    test_opts(opts, objective=[h,h_inv], x_init=x_init, n_epoch=test_epochs)

#Plot Losses and Gradient Norms
plot_all([times[:1], times[1:2], times[2:3], times[3:4]], \
    [losses[:1], losses[1:2], losses[2:3], losses[3:4]], \
    [grad_norms[:1], grad_norms[1:2], grad_norms[2:3], grad_norms[3:4]], \
    ['update first', 'learn tele gate', 'updates only', 'Adam'], n_epoch=test_epochs, xticks=[0,5,10,15], loc='func_figs')

#Plot Level Sets and Trajectories
colors = [["yellow","green"],["black", "orange"],["cyan","cyan"], ["pink","pink"]]
labels = [["update first (update step)", "update first (tele step)"], \
            ["gate (update step)", "gate (tele step)"], \
            ["updates only", ""], \
            ["Adam", ""]]
alphas = np.linspace(1,0.3,test_epochs+1)
plt.figure()
plot_level_sets(h_inv)
optimum = h_inv(torch.zeros(2))
plt.scatter(optimum[0], optimum[1], marker="*", s=40, color='#2ca02c')
for i in range(len(opts)):
    plt.scatter(x_arrs[i], y_arrs[i], c=colors[i][0], alpha=alphas, label=labels[i][0])
    plt.scatter(x_tele_arrs[i], y_tele_arrs[i], c=colors[i][1], alpha=alphas, label=labels[i][1])
    plt.plot(all_x_arrs[i], all_y_arrs[i], c=colors[i][0])
plt.legend(fontsize=10)
plt.savefig('func_figs/level_sets.pdf', bbox_inches='tight')

#TO DO
#implement imitation learning
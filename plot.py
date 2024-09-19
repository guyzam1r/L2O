""" Helper functions for figures. """

import numpy as np
import torch
from obj_tools import fixed_task
from matplotlib import pyplot as plt
import os

def plot_level_sets(h_inv):
    L = np.array([1e0, np.sqrt(10)*1e0, 1e1, np.sqrt(10)*1e1, 1e2, np.sqrt(10)*1e2, 1e3, np.sqrt(10)*1e3, 1e4])
    detail = 1000
    t = np.linspace(0, 2*np.pi, detail)
    for loss in L:
        uv = np.array([np.sqrt(loss) * np.cos(t), np.sqrt(loss) * np.sin(t)])
        uvT = uv.T
        xy = np.zeros((detail,2))
        for i in range(detail):
            xy[i] = h_inv(torch.from_numpy(uvT[i]).float()).numpy()
        xy = xy.T
        plt.plot(xy[0], xy[1], color='gray')
    plt.xlim(-10.0, 10.0)
    plt.ylim(-10.0, 10.0)

def plot_trajectories(iter_arrs, task_dist, labels, loc):
    x_arrs = iter_arrs[0]
    y_arrs = iter_arrs[1]
    x_tele_arrs = iter_arrs[2]
    y_tele_arrs = iter_arrs[3]
    all_x_arrs = iter_arrs[4]
    all_y_arrs = iter_arrs[5]

    colors = [["red","red"], \
            ["green", "lime"], \
            ["olive", "yellow"], \
            ["blue", "cyan"], \
            ["black","black"]]

    _,_,hinv = fixed_task(task_dist)
    alphas = np.linspace(1,0.2,20)
    plt.figure()
    plot_level_sets(hinv)
    optimum = hinv(torch.zeros(2))
    plt.scatter(optimum[0], optimum[1], marker="*", s=40, color='#2ca02c')
    for i in range(len(labels)):
        plt.scatter(x_arrs[i], y_arrs[i], c=colors[i][0], alpha=alphas, label=labels[i][0])
        plt.scatter(x_tele_arrs[i], y_tele_arrs[i], c=colors[i][1], alpha=alphas, label=labels[i][1])
        plt.plot(all_x_arrs[i], all_y_arrs[i], c=colors[i][0])
    plt.legend(fontsize=10)
    plt.title(label="Trained on " + task_dist, fontsize=14)
    plt.savefig(loc + '/trajectories.pdf', bbox_inches='tight')

def plot_all(time_arr_list, loss_arr_list, dL_dt_arr_list, label_list, n_epoch=30, xticks=[0,10,20,30], loc='figures', ylim=-1):
    if not os.path.exists(loc):
        os.mkdir(loc)

    # compute mean and std across multiple runs
    loss_mean_list = []
    loss_std_list = []
    time_mean_list = []
    time_std_list = []
    for i in range(len(time_arr_list)):
        time_arr_list[i] = np.array(time_arr_list[i])[:, :n_epoch]
        loss_arr_list[i] = np.array(loss_arr_list[i])[:, :n_epoch]
        dL_dt_arr_list[i] = np.array(dL_dt_arr_list[i])[:, :n_epoch]
        loss_mean_list.append(np.mean(loss_arr_list[i], axis=0))
        loss_std_list.append(np.std(loss_arr_list[i], axis=0))
        time_mean_list.append(np.mean(time_arr_list[i], axis=0))
        time_std_list.append(np.std(time_arr_list[i], axis=0))

    # plot loss vs epoch
    plt.figure()
    for i in range(len(time_arr_list)):
        plt.plot(loss_mean_list[i], linewidth=3, label=label_list[i])
    plt.gca().set_prop_cycle(None)
    for i in range(len(time_arr_list)):
        plt.fill_between(np.arange(n_epoch), loss_mean_list[i]-loss_std_list[i], loss_mean_list[i]+loss_std_list[i], alpha=0.5)
    plt.xlabel('Epoch', fontsize=26)
    plt.ylabel('Loss', fontsize=26)
    #plt.yscale('log')
    plt.xticks(xticks, fontsize= 20)
    plt.yticks(fontsize= 20)
    plt.legend(fontsize=17)
    if ylim > 0:
        plt.ylim(0, ylim)
    plt.savefig(loc + '/loss_vs_epoch.pdf', bbox_inches='tight')

    # plot loss vs wall-clock time
    plt.figure()
    for i in range(len(time_arr_list)):
        plt.plot(time_mean_list[i], loss_mean_list[i], linewidth=3, label=label_list[i])
        plt.fill_between(time_mean_list[i], loss_mean_list[i]-loss_std_list[i], loss_mean_list[i]+loss_std_list[i], alpha=0.5)
    plt.xlabel('time (s)', fontsize=26)
    plt.ylabel('Loss', fontsize=26)
    max_t = np.max(time_arr_list[0])
    interval = np.round(max_t * 0.3, 2)
    plt.xticks([0, interval, interval * 2, interval * 3], fontsize= 20)
    plt.yticks(fontsize= 20)
    #plt.yscale('log')
    plt.legend(fontsize=17)
    if ylim > 0:
        plt.ylim(0, ylim)
    plt.savefig(loc + '/loss_vs_time.pdf', bbox_inches='tight')

    # plot loss vs dL/dt
    plt.figure()
    for i in range(len(time_arr_list)):
        plt.plot(loss_arr_list[i][-1], dL_dt_arr_list[i][-1], linewidth=3, label=label_list[i])
    plt.xlabel('Loss', fontsize=26)
    plt.ylabel('dL/dt', fontsize=26)
    #plt.yscale('log')
    plt.xscale('log')
    plt.xticks(fontsize= 20)
    plt.yticks([1e1, 1e3, 1e5, 1e7], fontsize= 20)
    plt.legend(fontsize=17)
    if ylim > 0:
        plt.ylim(0, ylim)
    plt.savefig(loc + '/loss_vs_gradient.pdf', bbox_inches='tight')

    # plot dL/dt vs epoch
    plt.figure()
    for i in range(len(time_arr_list)):
        plt.plot(dL_dt_arr_list[i][-1], linewidth=3, label=label_list[i])
    plt.gca().set_prop_cycle(None)
    #for i in range(len(time_arr_list)):
    #    plt.fill_between(np.arange(n_epoch), loss_mean_list[i]-loss_std_list[i], loss_mean_list[i]+loss_std_list[i], alpha=0.5)
    plt.xlabel('Epoch', fontsize=26)
    plt.ylabel('dL/dt', fontsize=26)
    #plt.yscale('log')
    plt.xticks(xticks, fontsize= 20)
    plt.yticks(fontsize= 20)
    plt.legend(fontsize=17)
    plt.savefig(loc + '/gradient_vs_epoch.pdf', bbox_inches='tight')

    return

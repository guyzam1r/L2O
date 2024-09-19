import numpy as np
import time
import torch
from torch.autograd import Variable
from lstm import LSTM_testobj
from teleport_tools import rotate
from obj_tools import sample_obj, sample_x0, fixed_task

def W_list_to_vec(W_list):
    W_vec_all = torch.flatten(W_list[0])
    for i in range(1, len(W_list)):
        W_vec = torch.flatten(W_list[i])
        W_vec_all = torch.concat((W_vec_all, W_vec))
    return W_vec_all

def detach_var(v):
    # make gradient an independent variable that is independent from the rest of the computational graph
    var = Variable(v.data, requires_grad=True)
    var.retain_grad()
    return var

def obj(x, h):
    return torch.norm(h(x))**2

def train_meta_opt(n_run=200, n_epoch=21, unroll=5, hidden_size=200, learn_tele=False, learn_tele_gate=False, input_theta_grad=False, input_iterates=False, input_updates=False, tel_sched=[], exp_weights=-1, safeguard=False, task_dist=""):
    """ Initialize and train the meta optimizer on given task distribution.

    Args:
        n_run: number of training trajectories.
        n_epoch: number of epochs of each run.
        unroll: number of steps before each update of the parameters in the meta-optimizers.
        hidden_size: size of the hidden layer in the LSTM.
        learn_tele: True if meta-optimizers learn teleportation.
        learn_tele_gate: True if meta-opt learns to decide if to teleport or not each epoch.
        input_theta_grad: True if derivative wrt the teleport parameter of the gradient norm is used as input to meta-opt.
        input_iterates: True if iterates are used as input to meta-opt.
        input_updates: True if updates are used as input to meta-opt.
        tel_sched: List of epochs at which to teleport. [] for teleportation on all epochs.
        exp_weights: Specifies weighting coefficients. Default is -1 for equal weights.
        safeguard: True to prevent model from performing updates that increase loss by a lot.
        task_dist: Specifies the task distribution of the test objectives. Options are "elli_fixed", "elli_vary", "ros_fixed", "ros_vary", and "mixed".

    Returns:
        meta_opt: the trained meta_optimizer
    """
    # initialize meta opt
    dim = 2
    meta_opt = LSTM_testobj(dim, hidden_size, learn_tele, learn_tele_gate, input_theta_grad, \
                                input_iterates, input_updates, tel_sched, safeguard)
    optimizer = torch.optim.Adam(meta_opt.parameters(), lr=4e-4)

    # initialize weights for loss function
    weights = np.ones(unroll)
    if exp_weights > 0:
        for i in range(unroll):
            weights[i] = (exp_weights**(unroll - i - 1))
    weights /= np.sum(weights)

    # for each of the n_run training trajectories
    for n in range(n_run):
        if n % 50 == 0:
            print("Now on run", n)

        # initialize objective function and x0 for current training trajectory
        h, h_inv = sample_obj(task_dist)
        x = sample_x0(dim)

        # initialize LSTM hidden and cell
        hidden = torch.zeros((2, 1, hidden_size), requires_grad=True)
        cell = torch.zeros((2, 1, hidden_size), requires_grad=True)
        if input_updates:
            x_prev = x.detach()

        loss_sum = 0.0

        for epoch in range(n_epoch):
            # compute gradient
            if input_theta_grad:
                temp_theta = torch.zeros(1, requires_grad=True)
                x = rotate(x, temp_theta, h, h_inv)
                loss = obj(x, h)
                grad, = torch.autograd.grad(loss, inputs=x, create_graph=True)
                grad_norm = torch.norm(grad)**2
                theta_grad, = torch.autograd.grad(grad_norm, inputs=temp_theta, retain_graph=True)
            else:
                loss = obj(x, h)
                grad, = torch.autograd.grad(loss, inputs=x, retain_graph=True)

            # collect inputs to meta optimizer
            input_list = []
            if input_theta_grad:
                input_list.append(theta_grad)
            if input_iterates:
                input_list.append(x)
            if input_updates:
                input_list.append(x - x_prev)
                x_prev = x.detach()
            input_list.append(grad)
            inputs = W_list_to_vec(input_list)

            # compute update (and teleport, if applicable)
            if learn_tele:
                update, theta, hidden, cell = meta_opt(inputs, hidden, cell, teleport=True)
            else:
                update, hidden, cell = meta_opt(inputs, hidden, cell, teleport=True)

            # perform update
            if not safeguard or obj(x+update, h) <= 100 * loss:
                x = x + update
            else:
                loss_sum += obj(x+update, h)
                print("safeguarded in training run", n, "epoch", epoch)

            # perform teleportation
            if learn_tele and (tel_sched == [] or epoch in tel_sched):
                x = rotate(x, theta, h, h_inv)

            # accumulate loss
            loss_sum += weights[epoch % unroll] * loss

            # unroll and update meta optimizers
            if epoch % unroll == 0 and epoch != 0: 
                optimizer.zero_grad()
                loss_sum.backward(retain_graph=True)
                optimizer.step()

                loss_sum = 0.0
                hidden = detach_var(hidden)
                cell = detach_var(cell)
                update = detach_var(update)
                x = detach_var(x)
    return meta_opt

def test_opts(opts, task_dist, runs, epochs=10):
    """ Test optimizers on a given task distribution.

    Args:
        opts: list of pre-trained meta-optimizers and/or first-order optimizers.
        task_dist: task distribution to test on
        n_epoch: number of optimizer steps to take in testing.
        
    Returns:
        time_arrs: Wall-clock time at each epoch. Dimension len(opts) x runs x epochs.
        loss_arrs: Loss after each epoch. Dimension len(opts) x runs x epochs.
        grad_norm_arrs: Gradient norm at each epoch. Dimension len(opts) x runs x epochs.
        x_arrs: x-coordinate of the iterate at each epoch for the first run. Dimension runs x epochs.
        y_arrs: y-coordinate of the iterate at each epoch for the first run. Dimension runs x epochs.
        x_tele_arrs: x-coordinate of the iterate after a teleportation for the first run. Dimension runs x epochs.
        y_tele_arrs: y-coordinate of the iterate after a teleportation for the first run. Dimension runs x epochs.
        all_x_arrs: x-coordinate of the iterate at each epoch and after teleportations for the first run. Dimension len(opts) x runs x epochs.
        all_y_arrs: y-coordinate of the iterate at each epoch and after teleportations for the first run. Dimension len(opts) x runs x epochs.
    """

    time_arrs = []
    loss_arrs = []
    grad_norm_arrs = []
    x_arrs = []
    y_arrs = []
    x_tele_arrs = []
    y_tele_arrs = []
    all_x_arrs = []
    all_y_arrs = []

    for n in range(runs):
        # initalize h,h_inv, and x
        if n == 0:
            x0, h, h_inv = fixed_task(task_dist)
        else:
            h, h_inv = sample_obj(task_dist)
            x0 = sample_x0(2)

        time_arrs = [[] for i in range(len(opts))]
        loss_arrs = [[] for i in range(len(opts))]
        grad_norm_arrs = [[] for i in range(len(opts))]

        for i in range(len(opts)):
            opt = opts[i]
            x = x0.clone().detach().requires_grad_(True)

            time_arr = []
            loss_arr = []
            grad_norm_arr = []
            if n == 0:
                x_arr = [x.detach().numpy()[0]]
                y_arr = [x.detach().numpy()[1]]
                x_tele_arr = []
                y_tele_arr = []
                all_x_arr = [x.detach().numpy()[0]]
                all_y_arr = [x.detach().numpy()[1]]

            if opt.is_meta_opt:
                # initialize LSTM hidden and cell
                cell = torch.zeros((2, 1, opt.lstm_hidden_dim), requires_grad=True)
                hidden = torch.zeros((2, 1, opt.lstm_hidden_dim), requires_grad=True)
                if opt.input_updates:
                    x_prev = x.detach()    

            t0 = time.time()
            for epoch in range(epochs):
                teleported = False
                safeguarded = False

                if opt.is_meta_opt:
                    if opt.input_theta_grad:
                        temp_theta = torch.zeros(1, requires_grad=True)
                        x = rotate(x, temp_theta, h, h_inv)
                        loss = obj(x, h)
                        grad, = torch.autograd.grad(loss, inputs=x, create_graph=True)
                        grad_norm = torch.norm(grad)**2
                        theta_grad, = torch.autograd.grad(grad_norm, inputs=temp_theta, retain_graph=True)
                    else:
                        loss = obj(x, h)
                        grad, = torch.autograd.grad(loss, inputs=x, retain_graph=True)

                    # collect inputs to meta optimizer
                    input_list = []
                    if opt.input_theta_grad: 
                        input_list.append(theta_grad)
                    if opt.input_iterates:
                        input_list.append(x)
                    if opt.input_updates:
                        input_list.append(x - x_prev)
                        x_prev = x.detach()
                    input_list.append(grad)
                    inputs = W_list_to_vec(input_list)

                    if opt.learn_tele:
                        update, theta, hidden, cell = opt(inputs, hidden, cell, teleport=True)
                    else:
                        update, hidden, cell = opt(inputs, hidden, cell, teleport=True)
                        
                    if opt.safeguard and obj(x + update) > loss:
                        safeguarded = True
                        print("safeguarded!")
                    else:
                        x = x + update

                    if opt.learn_tele and theta != 0.0 and (opt.tel_sched == [] or epoch in opt.tel_sched):
                        teleported = True
                        x_tele = rotate(x, theta, h, h_inv)

                else:
                    loss = obj(x, h)
                    grad, = torch.autograd.grad(loss, inputs=x, retain_graph=True)

                    update = opt.step(grad)
                    x = x + update

                    if opt.teleport:
                        # compute theta_grad
                        loss2 = obj(x, h)
                        if epoch == 0:
                            theta_grad = 0.0
                        else:
                            theta_grad, = torch.autograd.grad(loss2, inputs=opt.theta)
                        
                        opt.theta_step(theta_grad)
                        if opt.theta != 0:
                            teleported = True
                            x_tele = rotate(x, opt.theta, h, h_inv)
                
                # compute norm of gradient
                grad_norm = torch.norm(grad)

                t1 = time.time()
                time_arr.append(t1 - t0)
                loss_arr.append(loss.detach().numpy())
                grad_norm_arr.append(grad_norm.detach().numpy())
                if n == 0:
                    if not safeguarded:
                        x_arr.append(x.detach().numpy()[0])
                        y_arr.append(x.detach().numpy()[1])
                        all_x_arr.append(x.detach().numpy()[0])
                        all_y_arr.append(x.detach().numpy()[1])
                    if teleported:
                        x_tele_arr.append(x_tele.detach().numpy()[0])
                        y_tele_arr.append(x_tele.detach().numpy()[1])
                        all_x_arr.append(x_tele.detach().numpy()[0])
                        all_y_arr.append(x_tele.detach().numpy()[1])
                        x = x_tele
            
            time_arrs[i].append(time_arr)
            loss_arrs[i].append(loss_arr)
            grad_norm_arrs[i].append(grad_norm_arr)
            if n == 0:
                x_arrs.append(x_arr)
                y_arrs.append(y_arr)
                x_tele_arrs.append(x_tele_arr)
                y_tele_arrs.append(y_tele_arr)
                all_x_arrs.append(all_x_arr)
                all_y_arrs.append(all_y_arr)

    iter_arrs = [x_arrs, y_arrs, x_tele_arrs, y_tele_arrs, all_x_arrs, all_y_arrs]
    return time_arrs, loss_arrs, grad_norm_arrs, iter_arrs

import numpy as np
from numpy import random
import time
import torch
from torch.autograd import Variable
from lstm import LSTM_testobj
from first_order_optims import GD_Opt, Momentum_Opt, Adam_Opt
from matplotlib import pyplot as plt
from teleportation import rotate, exact_teleport
import sys

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

def init_obj(func_type, params):
    if func_type == "elli":
        A = params[0]
        b = params[1]
        h, h_inv = init_elli(A,b)
    elif func_type == "ros":
        a = params[0]
        b = params[1]
        f = params[2]
        g = params[3]
        h, h_inv = init_ros(a, b, f, g)
    return h, h_inv

def init_elli(A, b):
    h = lambda x : torch.matmul(A,x) + b
    h_inv = lambda x : torch.matmul(torch.inverse(A), x-b)
    return h, h_inv

def init_ros(a, b, f, g):
    h = lambda x : torch.stack((f(x[0])*x[1] + g(x[0]), a*x[0] + b))
    h_inv = lambda x : torch.stack((  (x[1]-b)/a, (x[0] - g((x[1]-b)/a)) / f((x[1]-b)/a) ))
    return h, h_inv

def obj(x, h):
    return torch.norm(h(x))**2

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

def train_meta_opt(n_run=200, n_epoch=20, unroll=5, learn_tele=False, update_first=False, learn_tele_gate=False, input_iterates=False, input_updates=False, exp_weights=-1, safeguard=False):
    """ Initialize and train the meta optimizer on test objectives.

    Args:
        n_run: number of training trajectories.
        n_epoch: number of epochs of each run.
        unroll: number of steps before each update of the parameters in the meta-optimizers.
        learn_tele: True if meta-optimizers learn teleportation, False if no teleportation is applied.
        update_first: True if to perform update first, then teleport. False if to perform teleport first, then update.
        learn_tele_gate: True if meta-opt learns to decide if to teleport or not each epoch.
        input_iterates: True if iterates are used as input to meta-opt.
        input_updates: True if updates are used as input to meta-opt.
        exp_weights: Loss weighting coefficients. Default is -1 for equal weights.
        safeguard: True to prevent model from performing updates that increase loss by a lot.

    Returns:
        meta_opt: the trained meta_optimizer
    """

    # initialize meta opt
    dim = 2
    input_size = dim
    output_size = dim
    hidden_size = 300
    if input_iterates:
        input_size += dim
    if input_updates:
        input_size += dim
    meta_opt = LSTM_testobj(input_size, hidden_size, output_size, \
                    learn_tele, update_first, learn_tele_gate, input_iterates, input_updates, safeguard)

    optimizer = torch.optim.Adam(meta_opt.parameters(), lr=1e-3)

    # initialize weights for loss function
    weights = np.ones(unroll)
    if exp_weights > 0:
        for i in range(unroll):
            weights[i] = (exp_weights**(unroll - i - 1))
    weights /= np.sum(weights)

    # for each of the n_run training trajectories
    for n in range(n_run):
        if n % 50 == 0:
            print("run", n)

        loss_sum = 0.0
        loss_sum_all = 0.0

        rand_val = random.rand()
        if rand_val > 0.0:
            A = torch.randn(dim, dim)
            while torch.linalg.matrix_rank(A) < dim:
                A = torch.randn(dim, dim)
            b = torch.randn(dim)
            h, h_inv = init_obj("elli", [A,b])
        else:
            a = random.normal()
            b = random.normal()
            c = random.normal()
            d0 = random.normal()
            d1 = random.normal()
            d2 = random.normal()
            d3 = random.normal()
            f = lambda x : c
            if rand_val > 1.0:
                g = lambda x : d3*torch.sin(x) + d2*(x**2) + d1*x + d0
            else:
                g = lambda x : d2*(x**2) + d1*x + d0
            h, h_inv = init_obj("ros", [a,b,f,g])

        x = torch.randn(dim, requires_grad=True)

        # initialize LSTM hidden and cell
        hidden = torch.zeros((2, 1, meta_opt.lstm_hidden_dim), requires_grad=True)
        cell = torch.zeros((2, 1, meta_opt.lstm_hidden_dim), requires_grad=True)
        if input_updates:
            x_prev = x.detach()

        for epoch in range(n_epoch):
            # compute loss and gradients
            loss = obj(x, h)
            grad, = torch.autograd.grad(loss, inputs=x, retain_graph=True)

            # collect inputs to meta optimizer
            input_list = []
            if input_iterates:
                input_list.append(x)
            if input_updates:
                input_list.append(x - x_prev)
                x_prev = x.detach()
            input_list.append(grad)
            inputs = W_list_to_vec(input_list)

            # compute updates from meta optimizer
            if learn_tele:
                if learn_tele_gate:
                    update, theta, gate, hidden, cell = meta_opt(inputs, hidden, cell)
                else:
                    gate = 1
                    update, theta, hidden, cell = meta_opt(inputs, hidden, cell)

                if update_first:
                    if safeguard and obj(x+update, h) > 2*loss:
                        loss_sum += obj(x+update, h)
                        print("safeguarded on run", n, "epoch", epoch, "due to an increase of", (obj(x+update, h)/loss).data)
                    else:
                        x = x + update
                    
                    if gate == 1:
                        x = rotate(x, theta, h, h_inv)
                else:
                    if gate == 1:
                        x = rotate(x, theta, h, h_inv)
                    
                    if safeguard and obj(x+update, h) > 2*loss:
                        loss_sum += obj(x+update, h)
                        print("safeguarded on run", n, "epoch", epoch, "due to an increase of", (obj(x+update, h)/loss).data)
                    else:
                        x = x + update
            else:
                update, hidden, cell = meta_opt(inputs, hidden, cell)
                x = x + update

            # accumulate loss
            loss_sum_all += loss.data
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

def test_opts(opts, objective, x_init=[0,0], n_epoch=10):
    """ Run gradient descent with or without teleportation.

    Args:
        opts: list of pre-trained meta-optimizers and/or first-order optimizers.
        objective: list containing h and h_inv.
        x_init: list containing the initial x.
        n_epoch: number of optimizer steps to take in testing.
        
    Returns:
        time_arr_teleport_n: Wall-clock time at each epoch. Dimension n_run x n_epoch.
        loss_arr_teleport_n: Loss after each epoch. Dimension n_run x n_epoch.
        dL_dt_arr_teleport_n: Squared gradient norm at each epoch. Dimension n_run x n_epoch.
        lr_arr_teleport_n: Learning rate for MLP parameters at each epoch. Dimension n_run x n_epoch.
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

    for i in range(len(opts)):
        opt = opts[i]

        # initalize h,h_inv and initiate x
        x = torch.tensor(x_init, requires_grad=True)
        h = objective[0]
        h_inv = objective[1]

        time_arr = []
        loss_arr = []
        grad_norm_arr = []
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
        for epoch in range(n_epoch):
            safeguarded = False

            # compute loss and gradients
            loss = obj(x, h)
            grad, = torch.autograd.grad(loss, inputs=x, retain_graph=True)
            
            if opt.is_meta_opt:
                # collect inputs to meta optimizer
                input_list = []
                if opt.input_iterates:
                    input_list.append(x)
                if opt.input_updates:
                    input_list.append(x - x_prev)
                    x_prev = x.detach()
                input_list.append(grad)
                inputs = W_list_to_vec(input_list)

                # compute local updates from meta optimizer, then perform updates
                if opt.learn_tele:
                    if opt.learn_tele_gate:
                        update, theta, gate, hidden, cell = opt(inputs, hidden, cell)
                    else:
                        gate = 1
                        update, theta, hidden, cell = opt(inputs, hidden, cell)

                    if opt.update_first:
                        if opt.safeguard and obj(x+update, h) > 2*loss:
                            safeguarded = True
                        else:
                            x = x + update

                        if gate == 1:
                            x_tele = rotate(x, theta, h, h_inv)
                    else:
                        if gate == 1:
                            x_tele = rotate(x, theta, h, h_inv)

                        if opt.safeguard and obj(x+update, h) > 2*loss:
                            safeguarded = True
                        else:
                            x = x_tele + update
                else:
                    update, hidden, cell = opt(inputs, hidden, cell)
                    x = x + update    
            else:
                update = opt.step(grad)
                x = x + update        
            
            # compute norm of gradient
            grad_norm = torch.norm(grad)

            t1 = time.time()
            time_arr.append(t1 - t0)
            loss_arr.append(loss.detach().numpy())
            grad_norm_arr.append(grad_norm.detach().numpy())
            if opt.is_meta_opt and gate == 1 and opt.learn_tele and not opt.update_first:
                x_tele_arr.append(x_tele.detach().numpy()[0])
                y_tele_arr.append(x_tele.detach().numpy()[1])
                all_x_arr.append(x_tele.detach().numpy()[0])
                all_y_arr.append(x_tele.detach().numpy()[1])
            if not safeguarded:
                x_arr.append(x.detach().numpy()[0])
                y_arr.append(x.detach().numpy()[1])
                all_x_arr.append(x.detach().numpy()[0])
                all_y_arr.append(x.detach().numpy()[1])
            if opt.is_meta_opt and gate == 1 and opt.learn_tele and opt.update_first:
                x_tele_arr.append(x_tele.detach().numpy()[0])
                y_tele_arr.append(x_tele.detach().numpy()[1])
                all_x_arr.append(x_tele.detach().numpy()[0])
                all_y_arr.append(x_tele.detach().numpy()[1])
                x = x_tele

        time_arrs.append(time_arr)
        loss_arrs.append(loss_arr)
        grad_norm_arrs.append(grad_norm_arr)
        x_arrs.append(x_arr)
        y_arrs.append(y_arr)
        x_tele_arrs.append(x_tele_arr)
        y_tele_arrs.append(y_tele_arr)
        all_x_arrs.append(all_x_arr)
        all_y_arrs.append(all_y_arr)

    return time_arrs, loss_arrs, grad_norm_arrs, x_arrs, y_arrs, x_tele_arrs, y_tele_arrs, all_x_arrs, all_y_arrs

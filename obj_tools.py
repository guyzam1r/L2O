import numpy as np
from numpy import random
import torch

def init_elli(A, b):
    h = lambda x : torch.matmul(A,x) + b
    h_inv = lambda x : torch.matmul(torch.inverse(A), x-b)
    return h, h_inv

def init_ros(a, b, f, g):
    h = lambda x : torch.stack((f(x[0])*x[1] + g(x[0]), a*x[0] + b))
    h_inv = lambda x : torch.stack((  (x[1]-b)/a, (x[0] - g((x[1]-b)/a)) / f((x[1]-b)/a) ))
    return h, h_inv

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

def init_obj_rand(func_type):
    dim = 2
    if func_type == "elli":
        A = torch.randn(dim, dim)
        while torch.linalg.matrix_rank(A) < dim:
            A = torch.randn(dim, dim)
        b = torch.randn(dim)
        h, h_inv = init_obj("elli", [A,b])
    elif func_type == "ros":
        a = random.normal()
        b = random.normal()
        c = random.normal()
        d0 = random.normal()
        d1 = random.normal()
        d2 = random.normal()
        #d3 = random.normal()
        f = lambda x : c
        #g = lambda x : d3*torch.sin(x) + d2*(x**2) + d1*x + d0
        g = lambda x : d2*(x**2) + d1*x + d0
        h, h_inv = init_obj("ros", [a,b,f,g])
    return h, h_inv

def sample_x0(dim):
    return torch.randn(dim, requires_grad=True)

def sample_obj(task_dist):
    if task_dist == "" or task_dist == "elli_fixed":
        A = torch.tensor([[0.5, 0.], [0., 3.]])
        b = torch.tensor([0., 0.])
        return init_obj("elli", [A,b])
    elif task_dist == "elli_vary":
        return init_obj_rand("elli")
    elif task_dist == "ros_fixed":
        a = 1.0
        b = -1.0
        f = lambda x : -2.0
        g = lambda x : 0.4*(x**2)
        return init_obj("ros", [a,b,f,g])
    elif task_dist == "ros_vary":
        return init_obj_rand("ros")
    elif task_dist == "mixed":
        rand_val = random.rand()
        if rand_val > 0.5:
            return init_obj_rand("elli")
        else:
            return init_obj_rand("ros")
    else:
        print("Use valid task_dist")

def fixed_task(task_dist):
    if task_dist == "" or task_dist == "elli_fixed" or task_dist == "elli_vary":
        h, hinv =  sample_obj("elli_fixed")
        x0 = torch.tensor([-5.0, 3.5])
    elif task_dist == "ros_fixed" or task_dist == "ros_vary" or task_dist == "mixed":
        h, hinv = sample_obj("ros_fixed")
        x0 = torch.tensor([-6.0, -2.0])
    else:
        print("Use valid task_dist")
        return
    return x0, h, hinv
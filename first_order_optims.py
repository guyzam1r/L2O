import numpy as np
import torch
from torch import nn

class GD_Opt:
    def __init__(self, lr=1e-3):
        self.lr = lr
        self.is_meta_opt = False

    def step(self, grad):
    	return -self.lr * grad

class Momentum_Opt:
    def __init__(self, lr=1e-3, decay=0.6):
        self.lr = lr
        self.decay = decay
        self.momentum = 0
        self.is_meta_opt = False

    def step(self, grad):
    	self.momentum = self.decay * self.momentum - self.lr * grad
    	return self.momentum

    def reset_params(self):
    	self.momentum = 0

class Adam_Opt:
    def __init__(self, lr=1e-2, mmtm_decay=0.9, sqgd_decay=0.999, teleport=False, theta=0):
        self.lr = lr
        self.mmtm_decay = mmtm_decay
        self.sqgd_decay = sqgd_decay
        self.mmtm = 0
        self.sqgd = 0
        self.k = 1
        self.teleport=teleport
        self.theta=theta
        self.is_meta_opt = False

    def step(self, grad):
    	self.mmtm = self.mmtm_decay * self.mmtm + (1 - self.mmtm_decay) * grad
    	self.sqgd = self.sqgd_decay * self.sqgd + (1 - self.sqgd_decay) * torch.mul(grad,grad)
    	corrected_mmtm = self.mmtm / (1 - self.mmtm_decay**self.k)
    	corrected_sqgd = self.sqgd / (1 - self.sqgd_decay**self.k)
    	self.k += 1
    	return -self.lr * corrected_mmtm / (1e-8 + torch.sqrt(corrected_sqgd))

    def theta_step(self, theta_grad=0.0):
        pass

    def reset_params(self):
    	self.momentum = 0
    	self.sqgd = 0
    	self.k = 0

class L2TOTF_Opt:
    def __init__(self, lr=0.1, theta_lr=0.01, theta_init=0.0):
        self.lr = lr
        self.theta_lr = theta_lr
        self.theta = torch.tensor(theta_init, requires_grad=True)
        self.teleport = True
        self.is_meta_opt = False

    def step(self, grad):
        return -self.lr * grad

    def theta_step(self, theta_grad=0.0):        
        #update theta
        if theta_grad * self.theta <= 0:
            self.theta = self.theta - self.theta_lr * theta_grad
        else:
            print("Changing direction!")
            self.theta = 0*self.theta - 5 * self.theta_lr * theta_grad
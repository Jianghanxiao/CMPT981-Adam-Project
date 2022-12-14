import os

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import copy
import math

import argparse
from optmizer import Adam, SGD, MSGD, MSVAG, MSSD

def set_seed(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main(args):

    if args.optimizer == 'all':
        optimizers = ['sgd', 'msgd', 'adam', 'msvag', 'mssd']
    elif args.optimizer == 'sgd':
        optimizers = ['sgd']
    elif args.optimizer == 'adam':
        optimizers = ['adam']
    elif args.optimizer == 'msgd':
        optimizers = ['msgd']
    elif args.optimizer == 'msvag':
        optimizers = ['msvag']
    elif args.optimizer == 'mssd':
        optimizers = ['mssd']
    else:
        raise NotImplementedError

    def fn(w):

        x1 = torch.tensor([np.sqrt(3)/2, 1./2], dtype=torch.float32) # [sqrt(3)/2, 1/2]
        x2 = 50 * torch.tensor([-np.sqrt(3)/2, 1./2], dtype=torch.float32) # [-sqrt(3)/2, 1/2]

        a = torch.matmul(-x1, w)
        b = torch.matmul(-x2, w)

        return torch.exp(a) + torch.exp(b)

    def change_lr(optim, k=1):

        for g in optim.param_groups:
            g['lr'] = g['lr'] / np.sqrt(k)

    def get_optimizer(w, name, lr):
        if name == 'sgd':
            opt = SGD([w], lr=lr)
        elif name == 'adam':
            opt = Adam([w], lr=lr)
        elif name == 'msgd':
            opt = MSGD([w], lr=lr)
        elif name == 'msvag':
            opt = MSVAG([w], lr=lr)
        elif name == 'mssd':
            opt = MSSD([w], lr=lr)
        else:
            raise NotImplementedError
        return opt

    def eval_grad(w, name, lr, k):
        
        opt = get_optimizer(w, name, lr)
        change_lr(opt, k)
        
        opt.zero_grad()
        
        loss =  fn(w)
        loss.backward()
        
        gnorm = 0
        grad_ls = []
        for p in opt.param_groups[0]['params']:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                gnorm += param_norm.item() ** 2
                grad_ls.append(p.grad.data)
        gnorm = gnorm ** (1. / 2)
        
        return gnorm, grad_ls

    def eval_smooth(prev_w, w, name, lr, k, num_pts=1):
        alphas = np.arange(1, num_pts+1)/(num_pts+1)
        prev_gnorm, prev_grad_ls = eval_grad(prev_w, name, lr, k)
        update_size = norm_diff([prev_w], [w])
        max_smooth = -1
        for alpha in alphas:
            new_w = copy.deepcopy(prev_w)
            with torch.no_grad():
                new_w = alpha * prev_w + (1-alpha) * w
            new_w = torch.tensor(new_w, requires_grad=True)
                
            _, new_grad_ls = eval_grad(new_w, name, lr, k)
            smooth = norm_diff(new_grad_ls, prev_grad_ls)/ (update_size * (1- alpha))
            max_smooth = max(smooth, max_smooth)
        
        return max_smooth, prev_gnorm
    
    def norm_diff(list1, list2=None):
        if not list2:
            list2 = [0] * len(list1)
        assert len(list1) == len(list2)
        return math.sqrt(sum((list1[i]-list2[i]).norm()**2 for i in range(len(list1))))

    figure, axis = plt.subplots(1, 2, figsize=(12, 5))

    for name in optimizers:
        print(f'-----------------------------------')
        print(name.upper())
        set_seed(args.seed)
        w = torch.randn((2, ), requires_grad=True)

        opt = get_optimizer(w, name, args.lr)

        losses = []
        grad_norm_ls = []
        smoothness_ls = []
        for i in range(args.iters):
            prev_w = copy.deepcopy(w)
            # lr decay
            change_lr(opt, k=i+1)

            # zero grad
            opt.zero_grad()

            # function
            loss =  fn(w)
            print(f'Iter : {i} -- Loss : {loss}')

            loss.backward()
            opt.step()

            losses.append(loss.item())
            smoothness, fn_gradnorm = eval_smooth(prev_w, w, name, args.lr, i+1, num_pts=1)
            grad_norm_ls.append(fn_gradnorm)
            smoothness_ls.append(smoothness)

        losses = np.log10(np.array(losses))
        axis[0].plot(losses[np.logical_not(np.isnan(losses))], label=name)

        grad_norm_ls = np.log10(np.array(grad_norm_ls))
        smoothness_ls = np.log10(np.array(smoothness_ls))
        
        if len(grad_norm_ls[np.logical_not(np.isnan(grad_norm_ls))]) == len(smoothness_ls[np.logical_not(np.isnan(smoothness_ls))]):
            axis[1].scatter(grad_norm_ls[np.logical_not(np.isnan(grad_norm_ls))],
            smoothness_ls[np.logical_not(np.isnan(smoothness_ls))], label=name)
    
    axis[0].legend()
    axis[0].set_xlabel('Iteration')
    axis[0].set_ylabel('Log10 Loss')

    axis[1].legend()
    axis[1].set_xlabel('Log10 Grad Norm')
    axis[1].set_ylabel('Log10 Smoothness')

    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--iters', type=int, default=5)
    parser.add_argument('--optimizer', type=str, default='all', choices=['sgd', 'adam', 'msgd', 'mssd', 'msvag', 'all'])
    args = parser.parse_args()

    main(args)
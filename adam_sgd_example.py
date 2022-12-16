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

    for name in optimizers:
        figure, axis = plt.subplots(1, 2, figsize=(12, 5))
        min_losses = []
        min_gradnorm = []
        for lr in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
            noise = args.noise_std if args.stochastic else 0
            print('\n\n------------------------')
            print(f'OPT: {name} - LR: {lr} - Noise: {noise}')

            set_seed(args.seed)
            w = torch.randn((2, ), requires_grad=True)

            if name == 'sgd':
                opt = SGD([w], lr=lr, stochastic=args.stochastic, noise_std=args.noise_std)
            elif name == 'adam':
                opt = Adam([w], lr=lr, stochastic=args.stochastic, noise_std=args.noise_std)
            elif name == 'msgd':
                opt = MSGD([w], lr=lr, stochastic=args.stochastic, noise_std=args.noise_std)
            elif name == 'msvag':
                opt = MSVAG([w], lr=lr, stochastic=args.stochastic, noise_std=args.noise_std)
            elif name == 'mssd':
                opt = MSSD([w], lr=lr, stochastic=args.stochastic, noise_std=args.noise_std)
            else:
                raise NotImplementedError

            losses = []
            gnorm_ls = []
            for i in range(args.iters):

                # lr decay
                if name == 'sgd':
                    change_lr(opt, k=i+1)

                # zero grad
                opt.zero_grad()

                # function
                loss =  fn(w)
                # print(f'Iter : {i} -- Loss : {loss}')

                loss.backward()

                opt.step()

                losses.append(loss.item())

                gnorm = 0
                grad_ls = []
                for p in opt.param_groups[0]['params']:
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        gnorm += param_norm.item() ** 2
                        grad_ls.append(p.grad.data)
                gnorm = gnorm ** (1. / 2)
                gnorm_ls.append(gnorm)

            min_losses.append(min(losses))
            # print(f'Min loss: {min(losses)}')
            losses = np.log10(np.array(losses))
            axis[0].plot(losses[np.logical_not(np.isnan(losses))], label=name+f"_{lr}")
            
            min_gradnorm.append(min(gnorm_ls))
            # print(f'Min gradnorm: {min(gnorm_ls)}')
            gnorm_ls = np.log10(np.array(gnorm_ls))
            axis[1].plot(gnorm_ls[np.logical_not(np.isnan(gnorm_ls))], label=name+f"_{lr}")
        
        axis[0].legend()
        axis[0].set_title(f'Log10 min Loss: {np.log(min(min_losses))}')
        axis[0].set_xlabel('Iteration')
        axis[0].set_ylabel('Log10 Loss')

        axis[1].legend()
        axis[1].set_title(f'Log10 min Grad Norm: {np.log(min(min_gradnorm))}')
        axis[1].set_xlabel('Iteration')
        axis[1].set_ylabel('Log10 Grad Norm')

        figure.suptitle(f'{name} - noise: {noise}')
        plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--iters', type=int, default=5)
    parser.add_argument('--optimizer', type=str, default='all', choices=['sgd', 'adam', 'msgd', 'mssd', 'msvag', 'all'])
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--noise_std', type=float, default=0.1)
    args = parser.parse_args()

    main(args)
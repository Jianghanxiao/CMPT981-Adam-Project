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
        for lr in [1e-1]:#[1e-7, 1e-6, 1e-5, 9e-4, 3e-4, 1e-4, 9e-3, 3e-3, 1e-3, 9e-2, 3e-2, 1e-2, 0.9, 0.3, 0.1, 1.]:
            
            print('\n\n------------------------')
            print('LR: ', lr)

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
            for i in range(5):

                # lr decay
                if name == 'sgd':
                    change_lr(opt, k=i+1)

                # zero grad
                opt.zero_grad()

                # function
                loss =  fn(w)
                print(f'Iter : {i} -- Loss : {loss}')

                loss.backward()

                opt.step()

                losses.append(loss.item())

            losses = np.log10(np.array(losses))
            plt.plot(losses[np.logical_not(np.isnan(losses))], label=name+f"_{lr}")
        
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Log10 Loss')
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
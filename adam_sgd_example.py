import os

import random
import numpy as np
import torch
import matplotlib.pyplot as plt

import argparse
from optmizer import Adam, SGD, MSGD, MSVAG

def set_seed(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main(args):

    if args.optimizer == 'all':
        optimizers = ['sgd', 'msgd', 'adam', 'msvag']
    elif args.optimizer == 'sgd':
        optimizers = ['sgd']
    elif args.optimizer == 'adam':
        optimizers = ['adam']
    elif args.optimizer == 'msgd':
        optimizers = ['msgd']
    elif args.optimizer == 'msvag':
        optimizers = ['msvag']
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
        
        set_seed(args.seed)
        w = torch.randn((2, ), requires_grad=True)

        if name == 'sgd':
            opt = SGD([w], lr=args.lr)
        elif name == 'adam':
            opt = Adam([w], lr=args.lr)
        elif name == 'msgd':
            opt = MSGD([w], lr=args.lr)
        elif name == 'msvag':
            opt = MSVAG([w], lr=args.lr)
        else:
            raise NotImplementedError

        losses = []
        for i in range(5):

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

        losses = np.log10(np.array(losses))
        plt.plot(losses[np.logical_not(np.isnan(losses))], label=name)
    
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Log10 Loss')
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='msvag', choices=['sgd', 'adam', 'all'])
    args = parser.parse_args()

    main(args)
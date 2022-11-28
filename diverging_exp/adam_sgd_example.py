import os

import random
import numpy as np
import torch

def set_seed(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():

    set_seed(42)
    w = torch.randn((2, ), requires_grad=True)

    def fn(w):

        x1 = torch.tensor([np.sqrt(3)/2, 1./2], dtype=torch.float32) # [sqrt(3)/2, 1/2]
        x2 = 50 * torch.tensor([-np.sqrt(3)/2, 1./2], dtype=torch.float32) # [-sqrt(3)/2, 1/2]

        a = torch.matmul(-x1, w)
        b = torch.matmul(-x2, w)

        return torch.exp(a) + torch.exp(b)

    def change_lr(optim, k=1):

        for g in optim.param_groups:
            g['lr'] = g['lr'] / np.sqrt(k)

    # opt = torch.optim.SGD([w], lr=0.01)
    # opt = torch.optim.Adam([w], lr=0.01)
    opt = torch.optim.SGD([w], lr=0.01, momentum=0.9)

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


if __name__ == '__main__':
    main()
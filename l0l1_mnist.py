import os

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
import torch.nn.functional as F
from torchvision import transforms
from models import ConvNet
import math
import copy
import time

import argparse
from optmizer import Adam, SGD, MSGD, MSVAG, MSSD


transform = transforms.Compose([
        # transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
])

########################
# MNIST dataset (Fast)
print("Load Data ...")
data_root = 'data/MNIST'
train_dataset = datasets.MNIST(data_root, train=True, transform=transform, download=True)
print("Train set size: "+str(len(train_dataset)))
test_dataset = datasets.MNIST(data_root, train=False, transform=transform, download=True)
print("Test set size: "+str(len(test_dataset)))

TRAIN_BS = 1024
TEST_BS = 1024

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BS, shuffle=True)
# testloader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST_BS, shuffle=False)

train_x, train_y = next(iter(trainloader))
# test_x, test_y = next(iter(testloader))

train_x, train_y = train_x.cuda(), train_y.cuda()
# test_x, test_y = train_x.cuda(), train_y.cuda()

def set_seed(seed=2022):
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

    def change_lr(optim, k=1):
        for g in optim.param_groups:
            g['lr'] = g['lr'] / np.sqrt(k)
    
    def get_optimizer(model, name, lr):
        if name == 'sgd':
            opt = SGD(model.parameters(), lr=lr)
        elif name == 'adam':
            opt = Adam(model.parameters(), lr=lr)
        elif name == 'msgd':
            opt = MSGD(model.parameters(), lr=lr)
        elif name == 'msvag':
            opt = MSVAG(model.parameters(), lr=lr)
        elif name == 'mssd':
            opt = MSSD(model.parameters(), lr=lr)
        else:
            raise NotImplementedError
        return opt
    
    def get_model_grads(model):
        return [p.grad.data for _, p in model.named_parameters() if \
                hasattr(p, 'grad') and (p.grad is not None)]

    def get_model_params(model):
        return [p.data for _, p in model.named_parameters() if \
                hasattr(p, 'grad') and (p.grad is not None)]

    def norm_diff(list1, list2=None):
        if not list2:
            list2 = [0] * len(list1)
        assert len(list1) == len(list2)
        return math.sqrt(sum((list1[i]-list2[i]).norm()**2 for i in range(len(list1))))

    def eval_grad(model, name, lr, k):

        opt = get_optimizer(model, name, lr)
        change_lr(opt, k)

        model.zero_grad()

        # function
        out = model(train_x)
        loss = F.nll_loss(out, train_y)

        loss.backward()
        opt.step()

        gradnorm = 0
        grad_ls = []
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            gradnorm += param_norm.item() ** 2
            grad_ls.append(p.grad.data)
        gradnorm = gradnorm ** (1. / 2)
        
        return gradnorm, grad_ls



    def eval_smooth(prev_model, model, name, lr, k, num_pts=1):
        alphas = np.arange(1, num_pts+1)/(num_pts+1)
        gnorm, grad_ls = eval_grad(prev_model, name, lr, k)
        update_size = norm_diff(get_model_params(model), get_model_params(prev_model))
        max_smooth = -1
        for alpha in alphas:
            new_model = copy.deepcopy(prev_model)
            
            for n, p in new_model.named_parameters():
                p.data = alpha * p.data + (1-alpha) * {n:p for n, p in model.named_parameters()}[n].data
                
            _, new_grad_ls = eval_grad(new_model, name, lr, k)
            smooth = norm_diff(get_model_grads(new_model), get_model_grads(prev_model))/ (update_size * (1- alpha))
            max_smooth = max(smooth, max_smooth)
        
        return max_smooth, gnorm

    # figure, axis = plt.subplots(1, 1, figsize=(7, 6))
    for name in optimizers:
        losses_avg = []
        grad_norm_avg = []
        smoothness_avg = []
        for i in range(args.runs):
            train_x, train_y = next(iter(trainloader))
            train_x, train_y = train_x.cuda(), train_y.cuda()
            set_seed(args.seed)
            model = ConvNet(num_classes=10).cuda()

            opt = get_optimizer(model, name, args.lr)

            losses = []
            grad_norm_ls = []
            smoothness_ls = []
            for i in range(args.iters):
                prev_model = copy.deepcopy(model)
                # lr decay
                # change_lr(opt, k=i + 1)

                # zero grad
                opt.zero_grad()

                # function
                out = model(train_x)
                loss = F.nll_loss(out, train_y)
                # print(f'Iter : {i} -- Loss : {loss}')

                loss.backward()
                opt.step()

                losses.append(loss.item())
                smoothness, fn_gradnorm = eval_smooth(prev_model, model, name, args.lr, i+1, args.slice)
                grad_norm_ls.append(fn_gradnorm)
                smoothness_ls.append(smoothness)


            losses_avg.append(losses)
            # axis[1].plot(losses[np.logical_not(np.isnan(losses))], label=name)
            
            grad_norm_avg.append(grad_norm_ls)
            smoothness_avg.append(smoothness_ls)
        
        losses_avg = np.log(np.mean(np.array(losses_avg), axis=0))
        grad_norm_avg = np.log(np.mean(np.array(grad_norm_avg), axis=0))
        smoothness_avg = np.log(np.mean(np.array(smoothness_avg), axis=0))
        
        if len(grad_norm_avg[np.logical_not(np.isnan(grad_norm_avg))]) == len(smoothness_avg[np.logical_not(np.isnan(smoothness_avg))]):
            no_inf_grad = grad_norm_avg[smoothness_avg!= float('inf')]
            no_inf_smooth = smoothness_avg[smoothness_avg!= float('inf')]
            # print(no_inf_grad, no_inf_smooth)
            line = np.polyfit(no_inf_grad, no_inf_smooth, 1)
            fit_line = np.poly1d(line)
            plt.scatter(grad_norm_avg[np.logical_not(np.isnan(grad_norm_avg))],
            smoothness_avg[np.logical_not(np.isnan(smoothness_avg))], label=name)
            step = (max(no_inf_grad) - min(no_inf_grad))/10
            x_range = np.arange(min(no_inf_grad),max(no_inf_grad)+step,step)
            plt.plot(x_range, fit_line(x_range), label=name)

    # axis[0].legend()
    # axis[0].set_xlabel('Iteration')
    # axis[0].set_ylabel('Log10 Loss')

    plt.legend()
    plt.xlabel('Log10 Grad Norm')
    plt.ylabel('Log10 Smoothness')
    plt.title(f'Average over {args.runs} runs of {args.iters} iterations')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--iters', type=int, default=5)
    parser.add_argument('--slice', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default='all', choices=['sgd', 'adam', 'msgd', 'mssd', 'msvag', 'all'])
    parser.add_argument('--runs', type=int, default=5)
    args = parser.parse_args()

    main(args)

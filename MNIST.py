import os

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
import torch.nn.functional as F
from torchvision import transforms
from models import ConvNet

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

    for name in optimizers:

        set_seed(args.seed)
        model = ConvNet(num_classes=10).cuda()

        if name == 'sgd':
            opt = SGD(model.parameters(), lr=args.lr)
        elif name == 'adam':
            opt = Adam(model.parameters(), lr=args.lr)
        elif name == 'msgd':
            opt = MSGD(model.parameters(), lr=args.lr)
        elif name == 'msvag':
            opt = MSVAG(model.parameters(), lr=args.lr)
        elif name == 'mssd':
            opt = MSSD(model.parameters(), lr=args.lr)
        else:
            raise NotImplementedError

        losses = []
        for i in range(5):
            # lr decay
            change_lr(opt, k=i + 1)

            # zero grad
            opt.zero_grad()

            # function
            out = model(train_x)
            loss = F.nll_loss(out, train_y)
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
    parser.add_argument('--optimizer', type=str, default='all', choices=['sgd', 'adam', 'msgd', 'mssd', 'msvag', 'all'])
    args = parser.parse_args()

    main(args)
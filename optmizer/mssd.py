import torch
from torch.optim.optimizer import Optimizer, required


class MSSD(Optimizer):
    def __init__(self, params, lr=required, beta=0.9, stochastic=False, noise_std=0.1):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, beta=beta)
        super(MSSD, self).__init__(params, defaults)

        self.stochastic = stochastic
        self.noise_std = noise_std

    def __setstate__(self, state):
        super(MSSD, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if self.stochastic:
                    grad = grad + torch.randn_like(grad) * self.noise_std

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                
                exp_avg = state['exp_avg']
                beta = group['beta']

                state['step'] += 1

                exp_avg.mul_(beta).add_(1 - beta, grad)
                step_size = group['lr']

                p.data.add_(-step_size, torch.sign(exp_avg))

        return loss
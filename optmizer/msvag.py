import math
import torch
from torch.optim.optimizer import Optimizer


class MSVAG(Optimizer):
    r"""Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, beta=0.9):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, beta=beta)
        super(MSVAG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MSVAG, self).__setstate__(state)

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
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta= group['beta']

                state['step'] += 1
                bias_correction = 1 - beta ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta).add_(1 - beta, grad)
                exp_avg_sq.mul_(beta).addcmul_(1 - beta, grad, grad)

                m = torch.div(exp_avg, bias_correction)
                v = torch.div(exp_avg_sq, bias_correction)

                rho = ((1 - beta) * (1 + beta ** state['step'])) / ((1 + beta) * (1 - beta ** state['step']))

                s = torch.div(torch.add(v, -torch.mul(m, m)), 1 - rho)

                r = torch.div(torch.mul(m ,m), torch.add(torch.mul(m, m), torch.mul(s, rho)))
                
                step_size = group['lr']
                p.data.add_(-step_size, torch.mul(r, m))

        return loss
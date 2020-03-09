import math

import torch
import torch.nn as nn
from torch.autograd import Function

big = 1e20
huge = 1e30
small = 1e-10

def kl_div(x, y):
    # y = y.reshape(x.shape)
    div = torch.div(x, y + small)
    kl = torch.mul(y, div * torch.log(div + small) - div + 1)
    return kl


def wfrcost_matrix(D):
    pi2 = torch.tensor(math.pi / 2, device='cuda:0')
    return - 2 * torch.log(torch.cos(torch.min(D*2, pi2)) + 1e-5)


class WFRSinkhornLossCostFunc(Function):

    @staticmethod
    def forward(ctx, D, epsilon, niter):
        C = wfrcost_matrix(D.detach())
        batchSize, I, J = C.shape
        mu = torch.tensor([1.0] * I).cuda()
        nu = torch.tensor([1.0] * J).cuda()
        dx = torch.tensor([1 / I] * I).cuda()
        dy = torch.tensor([1 / J] * J).cuda()

        p_coef = 1 / (1 + epsilon)

        K_calc = lambda _u, _v: torch.clamp(
            torch.exp((-C + _u.reshape([-1, I, 1]) + _v.reshape([-1, 1, J])) / epsilon),
            0, huge)

        b = torch.ones_like(nu)
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        K = K_calc(u, v)

        for ii in range(niter):
            b = b.reshape([-1, 1, J])
            dy = dy.reshape([-1, 1, J])
            bdy = torch.mul(b, dy)
            s = torch.sum(torch.mul(K, bdy), -1)
            a = torch.clamp(torch.pow(torch.div(mu, torch.exp(u) * s), p_coef), 0, huge)

            a = a.reshape([-1, I, 1])
            dx = dx.reshape([-1, I, 1])
            adx = torch.mul(a, dx)
            s = torch.sum(torch.mul(K, adx), -2)

            b = torch.clamp(torch.pow(torch.div(nu, torch.exp(v) * s), p_coef), 0, huge)
            if torch.max(a) > big or torch.max(b) > big or ii == niter - 1:
                u = epsilon * torch.log(a).squeeze() + u
                v = epsilon * torch.log(b).squeeze() + v
                K = K_calc(u, v)
                b = torch.ones_like(nu)

        dy = dy.reshape([-1, 1, J])
        mu = mu.reshape([-1, I])
        kl1 = kl_div(torch.sum(torch.mul(K, dy), -1), mu)
        assert not torch.isnan(kl1).any()
        dx = dx.reshape([-1, I])
        cons1 = torch.sum(torch.mul(kl1, dx), -1)
        assert not torch.isnan(cons1).any()

        dx = dx.reshape([-1, I, 1])
        nu = nu.reshape([-1, J])
        kl2 = kl_div(torch.sum(torch.mul(K, dx), -2), nu)
        assert not torch.isnan(kl2).any()
        dy = dy.reshape([-1, J])
        cons2 = torch.sum(torch.mul(kl2, dy), -1)
        assert not torch.isnan(cons2).any()

        constrain = cons1 + cons2
        assert not torch.isnan(constrain).any()

        transport = torch.sum(torch.sum(torch.mul(torch.mul(dx, torch.mul(K, C)), dy), -1), -1)
        assert not torch.isnan(transport).any()

        p_opt = constrain + transport
        ctx.save_for_backward(K, D)

        assert not torch.isnan(p_opt).any()

        return p_opt

    @staticmethod
    def backward(ctx, grad_output):
        K, D = ctx.saved_tensors
        batchSize, I, J = K.shape
        dx = torch.tensor([1 / I] * I).reshape([1, I, 1]).cuda()
        dy = torch.tensor([1 / J] * J).reshape([1, 1, J]).cuda()
        grad_output = grad_output.detach().reshape([-1, 1, 1])
        grad_input = torch.mul(torch.mul(dx, K), dy) * grad_output * 2 * torch.tan(D)
        # print(grad_input.shape)
        return grad_input, None, None


class WFRSinkhornLoss(nn.Module):
    def __init__(self, epsilon, niter):
        super(WFRSinkhornLoss, self).__init__()
        self.epsilon = epsilon
        self.niter = niter

    def forward(self, C):
        return WFRSinkhornLossCostFunc.apply(C, self.epsilon, self.niter)

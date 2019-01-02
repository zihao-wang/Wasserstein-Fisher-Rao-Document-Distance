#!/usr/bin/env python
"""
sinkhorn_pointcloud.py

Discrete OT : Sinkhorn algorithm for point cloud marginals.

"""

import math

import torch
from torch.autograd import Variable


def sinkhorn_normalized(x, y, epsilon, n, niter):
    Wxy = sinkhorn_loss(x, y, epsilon, n, niter)
    Wxx = sinkhorn_loss(x, x, epsilon, n, niter)
    Wyy = sinkhorn_loss(y, y, epsilon, n, niter)
    return 2 * Wxy - Wxx - Wyy


def sinkhorn_loss(x, y, epsilon, n, niter):
    """
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    """

    # The Sinkhorn algorithm takes as input three variables :
    C = Variable(lpnorm_matrix(x, y))  # Wasserstein cost function

    # both marginals are fixed with equal weights
    # mu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
    # nu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
    mu = Variable(1. / n * torch.FloatTensor(n).fill_(1), requires_grad=False)
    nu = Variable(1. / n * torch.FloatTensor(n).fill_(1), requires_grad=False)

    # Parameters of the Sinkhorn algorithm.
    rho = 1  # (.5) **2          # unbalanced transport
    tau = -.8  # nesterov-like acceleration
    lam = rho / (rho + epsilon)  # Update exponent
    thresh = 10 ** (-1)  # stopping criterion

    # Elementary operations .....................................................................
    def ave(u, u1):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    def lse(A):
        "log-sum-exp"
        return torch.log(torch.exp(A).sum(1, keepdim=True) + 1e-6)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

    for i in range(niter):
        u1 = u  # useful to check the update
        u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
        # accelerated unbalanced iterations
        # u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze()   ) + u ) )
        # v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze() ) + v ) )
        err = (u - u1).abs().sum()

        actual_nits += 1
        if (err < thresh).data.numpy():
            break
    U, V = u, v
    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    cost = torch.sum(pi * C)  # Sinkhorn cost

    return cost


def WFR_sinkhorn_iteration(C, mu, nu, dx, dy, epsilon, niter, u: torch.Tensor = None, v: torch.Tensor = None, device=None):
    """
    Sinkhorn iteration for entropy regularized
    :param C: cost matrix [I, J]
    :param mu: source measure [1, I]
    :param nu: target measure [J, 1]
    :param dx: reference measure for source
    :param dy: reference measure for target
    :param epsilon: regularization coefficient
    :param niter: number of iteration
    :param u: (optional) potential for source [1, I]
    :param v: (optional) potential for target [J, 1]
    :return:
    """

    with torch.cuda.device(0):
        *_, I, J = C.shape

        p_coef = 1 / (1 + epsilon)
        big = 1e20
        huge = 1e30

        K_calc = lambda _u, _v: torch.clamp(torch.exp((-C + _u.reshape([-1, I, 1]) + _v.reshape([-1, 1, J])) / epsilon), 0, huge)

        b = torch.ones_like(nu)
        u = torch.zeros_like(mu) if u is None else u
        v = torch.zeros_like(nu) if v is None else v
        K = K_calc(u, v)

        for ii in range(niter):
            b = b.reshape([-1, 1, J])
            dy = dy.reshape([-1, 1, J])
            bdy = torch.mul(b, dy)
            s = torch.sum(torch.mul(K, bdy), -1)
            a = torch.clamp(torch.pow(torch.div(mu, torch.exp(u) * s), p_coef), 0, huge)
            check(a)
            a = a.reshape([-1, I, 1])
            dx = dx.reshape([-1, I, 1])
            adx = torch.mul(a, dx)
            s = torch.sum(torch.mul(K, adx), -2)
            b = torch.clamp(torch.pow(torch.div(nu, torch.exp(v) * s), p_coef), 0, huge)
            check(b)
            if torch.max(a) > big or torch.max(b) > big or ii == niter - 1:
                # if ii < niter - 1:
                # print("too big")
                # print('a', a)
                # print('b', b)
                u = epsilon * torch.log(a).squeeze() + u
                check(u)
                v = epsilon * torch.log(b).squeeze() + v
                check(v)
                K = K_calc(u, v)
                check(K)
                b = torch.ones_like(nu)
        # print(u)
        # print(v)
        # print(K)
        return K, u, v


def WFR_sinkhorn(C, mu, nu, tol):
    """
    Calculate WFR by Sinkhorn algorithm
    :param C: cost matrix [I, J]
    :param mu: source measure [I]
    :param nu: target measure [J]
    :param tol: tolerance
    :return:
    """
    epsilon = math.e ** (-2)
    niter = 32
    u, v = None, None
    reg_pd_gap_list = []
    pd_gap_list = []
    dx = torch.ones_like(mu, device='cuda:0')
    dx = dx / torch.sum(dx)
    dy = torch.ones_like(nu, device='cuda:0')
    dy = dy / torch.sum(dy)
    while True:
        # for _ in range(niter):
        K, u, v = WFR_sinkhorn_iteration(C, mu, nu, dx, dy, epsilon, niter, u, v)

        p_opt, reg_p_opt = KP(C, mu, nu, dx, dy, K, epsilon)
        d_opt, reg_d_opt = DP(C, mu, nu, dx, dy, K, u, v, epsilon)

        reg_pd_gap = reg_p_opt - reg_d_opt
        reg_pd_gap_list.append(reg_pd_gap)

        pd_gap = p_opt - d_opt
        pd_gap_list.append(pd_gap)

        if pd_gap < tol:
            print('pd gaps:')
            for vv in pd_gap_list:
                print(vv.data)
            print('entropy gaps:')
            for vv in reg_pd_gap_list:
                print(vv.data)
            return p_opt

        if len(reg_pd_gap_list) > 1:
            if reg_pd_gap_list[-1] > reg_pd_gap_list[-2] * 0.8:  # slow update
                niter *= 2
            elif reg_pd_gap_list[-1] < reg_pd_gap_list[-2] * 0.2:  # fast enough update
                niter = int(niter / 2) if niter > 32 else 32

        if len(pd_gap_list) > 1 and reg_pd_gap_list[-1] / pd_gap_list[-1] < math.e ** 0.5:
            epsilon /= math.e


def KP(C, mu, nu, dx, dy, K, epsilon):
    batchSize, I, J = K.shape
    dy = dy.reshape([-1, 1, J])
    mu = mu.reshape([-1, I])
    kl1 = kl_div(torch.sum(torch.mul(K, dy), -1), mu)
    dx = dx.reshape([-1, I])
    cons1 = torch.sum(torch.mul(kl1, dx), -1)

    dx = dx.reshape([-1, I, 1])
    nu = nu.reshape([-1, J])
    kl2 = kl_div(torch.sum(torch.mul(K, dx), -2), nu)
    dy = dy.reshape([-1, J])
    cons2 = torch.sum(torch.mul(kl2, dy), -1)

    constrain = cons1 + cons2
    transport = torch.sum(torch.sum(
        torch.mul(torch.mul(dx, torch.mul(K, C)), dy),
        -1), -1)

    regular = torch.sum(torch.sum(
        torch.mul(torch.mul(dx, torch.mul(K, torch.log(K + 1e-6) - 1) + torch.exp(-C / epsilon)), dy),
        -1), -1) * epsilon
    p_opt = constrain + transport
    reg_p_opt = p_opt + regular
    return p_opt.squeeze(), reg_p_opt.squeeze()


def DP(C, mu, nu, dx, dy, K, u, v, epsilon):
    d_opt = dot2d(1 - torch.exp(-u), torch.mul(mu, dx)) + dot2d(
        1 - torch.exp(-torch.min(C - u, 1)[0].reshape(dy.shape)), torch.mul(nu, dy))
    constrain = dot2d(1 - torch.exp(-u), torch.mul(mu, dx)) + dot2d(1 - torch.exp(-v), torch.mul(nu, dy))
    regular = epsilon * torch.chain_matmul(dx, torch.exp(-C / epsilon) - K, dy)
    reg_d_opt = constrain + regular
    return d_opt, reg_d_opt.squeeze()


def lpnorm_matrix(x, y, p=2):
    """
    Returns the matrix of $|x_i-y_j|^p$.
    """
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2) ** (1 / p)
    return c


def wfrcost_matrix(D):
    pi = torch.tensor(math.pi / 2, device='cuda:0')
    return - 2 * torch.log(torch.cos(torch.min(D, pi)) + 1e-5)


def dot2d(x, y):
    return torch.sum(torch.mul(x, y))


def kl_div(x, y):
    # y = y.reshape(x.shape)
    div = torch.div(x, y)
    kl = torch.mul(y, div * torch.log(div) - div + 1)
    check(kl)
    return kl


def check(x):
    infjudge = torch.isinf(x)
    nanjudge = torch.isnan(x)
    if torch.max(infjudge):
        print("inf", x)
        assert False
    if torch.max(nanjudge):
        print("Nan", x)
        assert False


def wfr_sinkhorn_dist_loss(D: torch.Tensor, epsilon, niter):
    batchSize, I, J = D.shape
    mu = torch.tensor([1.0] * I).cuda()
    nu = torch.tensor([1.0] * J).cuda()
    dx = torch.tensor([1 / I] * I).cuda()
    dy = torch.tensor([1 / J] * J).cuda()
    D_ = D.clone()
    K, u, v = WFR_sinkhorn_iteration(D_, mu, nu, dx, dy, epsilon, niter)
    kp, _ = KP(D, mu, nu, dx, dy, K, epsilon)
    return kp.reshape([batchSize, 1])


if __name__ == "__main__":
    print("test the evaluation functions")
    C = torch.tensor([[0.5870, 0.5244, 0.6178], [0.5347, 0.9248, 2.2422], [1.6803, 1.2681, 0.1163]], device='cuda:0')
    K = torch.tensor([[0.4709, 1.8036, 0.1174], [1.8458, 0.2491, 0.0000], [0.0001, 0.0040, 2.5938]], device='cuda:0')
    u = torch.tensor([[0.2267, 0.3592, 0.1440]], device='cuda:0')
    v = torch.tensor([[0.2584], [0.3775], [0.1012]], device='cuda:0')
    epsilon = 0.1353
    mu = torch.ones_like(u)
    nu = torch.ones_like(v)
    dx = torch.ones_like(u) / 3
    dy = torch.ones_like(v) / 3
    p_opt, reg_p_opt = KP(C, mu, nu, dx, dy, K, epsilon)
    d_opt, reg_d_opt = DP(C, mu, nu, dx, dy, K, u, v, epsilon)
    print(p_opt, reg_p_opt)
    print(d_opt, reg_d_opt)
    assert abs(p_opt - 0.3702) < 1e-5
    assert abs(reg_p_opt - 0.3265) < 1e-5
    assert abs(d_opt - 0.3430) < 1e-5
    assert abs(reg_d_opt - 0.3265) < 1e-5

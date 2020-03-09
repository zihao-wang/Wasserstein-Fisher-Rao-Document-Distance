import math
import numpy as np
import torch

big = 1e300
huge = 1e300
small = 1e-100

def torch_nan_debuger(tensor):
    assert not torch.isnan(tensor).any()



def zero_mask(M, mask):
    M[mask < small] = 0
    return M

def kl_div(x, y):
    # y = y.reshape(x.shape)
    div = torch.div(x, y + small)
    div[div > big] = big
    torch_nan_debuger(div)
    kl = torch.mul(y, div * torch.log(div + small) - div + 1)
    torch_nan_debuger(kl)
    return kl


def wfrcost_matrix(D, coef):
    pi = torch.tensor(math.pi / 2, device=D.device, dtype=D.dtype)
    C = - 2 * torch.log(torch.cos(torch.min(D * coef, pi)) + small)
    C[D == float('inf')] = float('inf')
    return C


def wfr_sinkhorn_iteration(C, mu, nu, epsilon, niter, u=None, v=None, dx=None, dy=None, device=torch.device("cpu")):
    """
    calculate sinkhorn iteration for list of distance
    :param D: [num, I, J]
    :param mu: [num, I, 1]
    :param nu: [num, 1, J]
    :param epsilon: scala
    :param niter: scala
    :param u: [num, I, 1]
    :param v: [num, 1, J]
    :param dx: [num, I, 1]
    :param dy: [num, 1, J]
    :return: K [num, I, J], u [num, I, 1], v [num, 1, J]
    """
    num, I, J = C.shape
    p_coef = 1 / (1 + epsilon)

    K_calc = lambda _u, _v: torch.clamp(torch.exp((-C + _u + _v) / epsilon), 0, huge)

    b = torch.ones_like(nu, device=device)

    u = torch.zeros_like(mu, device=device) if u is None else u
    dx = torch.ones_like(mu, device=device) / I if dx is None else dx
    v = torch.zeros_like(nu, device=device) if v is None else v
    dy = torch.ones_like(nu, device=device) / J if dy is None else dy

    K = K_calc(u, v)

    for ii in range(niter):
        bdy = torch.mul(b, dy)  # num, 1, J
        s = torch.sum(torch.mul(K, bdy), -1, keepdim=True)  # num, I, 1
        a = torch.clamp(torch.pow(torch.div(mu, torch.exp(u) * s), p_coef), 0, huge)  # num, I, 1
        torch_nan_debuger(a)
        adx = torch.mul(a, dx)  # num, I, 1
        s = torch.sum(torch.mul(K, adx), -2, keepdim=True)  # num, 1, J
        b = torch.clamp(torch.pow(torch.div(nu, torch.exp(v) * s + small), p_coef), 0, huge)  # num, 1, J
        torch_nan_debuger(b)
        if torch.max(a) > big or torch.max(b) > big or ii == niter - 1:
            u = epsilon * torch.log(a + small) + u
            v = epsilon * torch.log(b + small) + v
            K = K_calc(u, v)
            b = torch.ones_like(nu)
            torch_nan_debuger(K)
            torch_nan_debuger(u)
            torch_nan_debuger(v)

    return K, u, v


def wfr_dist_approx_2(C, mu, nu, dx, dy, round, device=torch.device("cpu")):
    epsilon = math.e ** (-2)
    niter = 32
    reg_pd_gap_list = []
    pd_gap_list = []

    u, v = None, None

    for _ in range(round):
        # for _ in range(niter):
        K, u, v = wfr_sinkhorn_iteration(C, mu, nu, epsilon, niter, u, v, dx, dy, device=device)

        p_opt, reg_p_opt = KP(C, mu, nu, dx, dy, K, epsilon)
        d_opt, reg_d_opt = DP(C, mu, nu, dx, dy, K, u, v, epsilon)

        reg_pd_gap = torch.mean(reg_p_opt - reg_d_opt)
        reg_pd_gap_list.append(reg_pd_gap)

        pd_gap = torch.mean(p_opt - d_opt)
        pd_gap_list.append(pd_gap)

        if len(reg_pd_gap_list) > 1:
            if reg_pd_gap_list[-1] > reg_pd_gap_list[-2] * 0.8:  # slow update
                niter *= 2
            elif reg_pd_gap_list[-1] < reg_pd_gap_list[-2] * 0.2:  # fast enough update
                niter = int(niter / 2) if niter > 32 else 32

        if len(pd_gap_list) > 1 and reg_pd_gap_list[-1] / pd_gap_list[-1] < math.e ** 0.5:
            epsilon /= math.e

    torch_nan_debuger(p_opt)

    return p_opt


def wfr_dist_approx(D, mu, nu, dx, dy, round, coef):
    """
    calculate sinkhorn iteration for list of distance
    :param D: [num, I, J]
    :param mu: [num, I, 1]
    :param nu: [num, 1, J]
    :param dx: [num, I, 1]
    :param dy: [num, 1, J]
    :param round: scala, num of round that the iterations do
    :return: [num]
    """
    C = wfrcost_matrix(D.detach(), coef)

    epsilon = math.e ** (-2)
    niter = 32
    reg_pd_gap_list = []
    pd_gap_list = []

    u, v = None, None

    for _ in range(round):
        # for _ in range(niter):
        K, u, v = wfr_sinkhorn_iteration(C, mu, nu, epsilon, niter, u, v, dx, dy)

        p_opt, reg_p_opt = KP(C, mu, nu, dx, dy, K, epsilon)
        d_opt, reg_d_opt = DP(C, mu, nu, dx, dy, K, u, v, epsilon)

        reg_pd_gap = torch.mean(reg_p_opt - reg_d_opt)
        reg_pd_gap_list.append(reg_pd_gap)

        pd_gap = torch.mean(p_opt - d_opt)
        pd_gap_list.append(pd_gap)

        if len(reg_pd_gap_list) > 1:
            if reg_pd_gap_list[-1] > reg_pd_gap_list[-2] * 0.8:  # slow update
                niter *= 2
            elif reg_pd_gap_list[-1] < reg_pd_gap_list[-2] * 0.2:  # fast enough update
                niter = int(niter / 2) if niter > 32 else 32

        if len(pd_gap_list) > 1 and reg_pd_gap_list[-1] / pd_gap_list[-1] < math.e ** 0.5:
            epsilon /= math.e

    torch_nan_debuger(p_opt)

    return p_opt


def KP(C, mu, nu, dx, dy, K, epsilon):
    kl1 = kl_div(torch.sum(torch.mul(K, dy), -1, keepdim=True), mu)  # [num, I, 1]
    cons1 = torch.sum(torch.mul(kl1, dx), -2).squeeze()  # num
    torch_nan_debuger(cons1)

    kl2 = kl_div(torch.sum(torch.mul(K, dx), -2, keepdim=True), nu)  # [num, 1, J]
    cons2 = torch.sum(torch.mul(kl2, dy), -1).squeeze()  # num
    torch_nan_debuger(cons2)

    constrain = cons1 + cons2
    torch_nan_debuger(constrain)

    transport = torch.sum(torch.sum(
        torch.mul(torch.mul(dx, zero_mask(torch.mul(K, C), K)), dy),
        -1), -1)
    torch_nan_debuger(transport)

    regular = torch.sum(torch.sum(
        torch.mul(torch.mul(dx, torch.mul(K, torch.log(K + small) - 1) + torch.exp(-C / epsilon)), dy),
        -1), -1) * epsilon

    p_opt = constrain + transport

    reg_p_opt = p_opt + regular

    torch_nan_debuger(regular)

    return p_opt.reshape((-1)), reg_p_opt.reshape((-1))


def DP(C, mu, nu, dx, dy, K, u, v, epsilon):
    v_, _ = torch.min(C - u, 1, keepdim=True)  # [num, 1, J]

    d_opt = dot2d(1 - torch.exp(-u), torch.mul(mu, dx)) + dot2d(1 - torch.exp(-v_), torch.mul(nu, dy))

    constrain = dot2d(1 - torch.exp(-u), torch.mul(mu, dx)) + dot2d(1 - torch.exp(-v), torch.mul(nu, dy))

    regular = epsilon * torch.sum(torch.sum(
        torch.mul(torch.mul(dx, torch.exp(-C / epsilon) - K), dy),
        -1), -1)

    reg_d_opt = constrain + regular

    torch_nan_debuger(d_opt)
    torch_nan_debuger(constrain)
    torch_nan_debuger(regular)

    return d_opt, reg_d_opt.reshape(-1)


def dot2d(x, y):
    return torch.sum(torch.sum(torch.mul(x, y), -1), -1)

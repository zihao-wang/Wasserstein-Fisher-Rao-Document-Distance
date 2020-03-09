import sys
import math
import pickle
import time
import os
import argparse

sys.path.append("..")

import torch
import numpy as np

from dist_toolbox import wfrcost_matrix, wfr_sinkhorn_iteration, KP, DP, wfr_dist_approx_2
from knn_dataset_helper import KNNHeldoutEvaluation, availableDatasetIDs

argParser = argparse.ArgumentParser()
argParser.add_argument("--cuda", default='-1')
argParser.add_argument("--coef", type=float, default=1.0)
argParser.add_argument("--testratio", type=float, default=1.0)
argParser.add_argument("--trainratio", type=float, default=1.0)
argParser.add_argument("--distgran", type=int, default=1)
argParser.add_argument("--wfrgran", type=int, default=1)
argParser.add_argument("--dataset", default='twitter')

args = argParser.parse_args()
args.device = None
if int(args.cuda) < 0 or not torch.cuda.is_available():
    args.device = torch.device("cpu")
    print("use cpu", args.device)
else:
    args.device = torch.device("cuda:%s" % args.cuda)
    print("use gpu", args.device)



knnk = 20
def prepare_bow_dist(test_sample, train_sample, gran, coef=1, device=torch.device("cpu")):
    costmats = []
    mus = []
    nus = []
    yts = []

    xs, bows, ys = test_sample
    bows = np.asarray(bows/np.sum(bows)).reshape((-1,))
    I = len(bows.tolist())
    if I == 0:
        return None, None, None, None
    head = 0
    J = len(train_sample[0][1].tolist())
    tail = head + max(int(gran / (I * J)), 1)
    assert head < tail
    while head < len(train_sample):
        crt_sample = train_sample[head: tail]
        block_size = len(crt_sample)
        # prepare source block
        xsArray = np.repeat(xs.reshape([1, I, 1, -1]), block_size, axis=0)
        xtlist, bowtlist, ytlist = [], [], []
        tllist = []
        for xt, bowt, yt in crt_sample:
            # prepare target block
            bowt = np.asarray(bowt / np.sum(bowt)).reshape((-1,))
            xtlist.append(xt)
            bowtlist.append(bowt)
            tllist.append(len(bowt.tolist()))
            ytlist.append(yt)
        J = max(tllist)
        xtArray = np.zeros([block_size, 1, J, 300])
        for ii in range(block_size):
            xtArray[ii, 0, :tllist[ii], :] = xtlist[ii]
        xsTensor = torch.from_numpy(xsArray).to(device)
        xtTensor = torch.from_numpy(xtArray).to(device)
        costMatrix = wfrcost_matrix(torch.sqrt(torch.sum((xsTensor - xtTensor)**2, -1)).cpu(), coef).numpy()
        del xsTensor, xtTensor
        torch.cuda.empty_cache()
        costmats += np.split(costMatrix, range(1, block_size), axis=0)
        mus += [bows] * block_size
        nus += bowtlist
        yts += ytlist
        head = tail
        J = len(train_sample[tail if tail < len(train_sample) else -1][1].tolist())
        tail = head + max(int(gran / (I * J)), 1)

    return mus, nus, costmats, yts


def one_round_iteration(bows_list, bowt_list, costmat_list, gran, epsilon, niter, u_list=None, v_list=None, device=torch.device("cpu")):
    new_u_list = []
    new_v_list = []
    new_p_opt = []
    new_d_opt = []

    bows = bows_list[0]
    I = len(bows.tolist())
    head = 0
    J = len(bowt_list[0].tolist())
    tail = head + max(int(gran / (I * J)), 1)
    assert head < tail
    while head < len(bowt_list):

        bowsl = bows_list[head:tail]
        bowtl = bowt_list[head:tail]
        costmat_l = costmat_list[head:tail]
        if u_list is not None:
            ul = u_list[head:tail]
        if v_list is not None:
            vl = v_list[head:tail]

        num = len(bowsl)
        sllist = [len(bows) for bows in bowsl]
        tllist = [len(bowt) for bowt in bowtl]

        C = np.ones([num, I, J]) * np.inf
        mu = np.zeros([num, I, 1])
        nu = np.zeros([num, 1, J])
        u = np.zeros([num, I, 1])
        v = np.zeros([num, 1, J])

        for numi in range(num):
            C[numi, :sllist[numi], :tllist[numi]] = costmat_l[numi][0, :sllist[numi], :tllist[numi]]
            mu[numi, :sllist[numi], 0] = bowsl[numi]
            nu[numi, 0, :tllist[numi]] = bowtl[numi]
            if u_list is not None:
                u[numi, :sllist[numi], 0] = ul[numi][:, :sllist[numi], 0]
            if v_list is not None:
                *_, j = u_list[numi].shape
                # assert j == tllist[numi]
                v[numi, 0, :tllist[numi]] = vl[numi][:, 0, :tllist[numi]]

        dx = np.ones([num, I, 1])
        dx[mu == 0] = 0
        dy = np.ones([num, 1, J])
        dy[nu == 0] = 0

        CTensor = torch.from_numpy(C).to(device)
        muTensor = torch.from_numpy(mu).to(device)
        nuTensor = torch.from_numpy(nu).to(device)
        dxTensor = torch.from_numpy(dx).to(device)
        dyTensor = torch.from_numpy(dy).to(device)
        uTensor = torch.from_numpy(u).to(device)
        vTensor = torch.from_numpy(v).to(device)

        new_K, new_u, new_v = wfr_sinkhorn_iteration(CTensor, muTensor, nuTensor, epsilon, niter, uTensor, vTensor, dxTensor, dyTensor, device=device)
        p_opt, reg_p_opt = KP(CTensor, muTensor, nuTensor, dxTensor, dyTensor, new_K, epsilon)
        d_opt, reg_d_opt = DP(CTensor, muTensor, nuTensor, dxTensor, dyTensor, new_K, uTensor, vTensor, epsilon)
        new_u_list += np.split(new_u.cpu().numpy(), range(1, num), axis=0)
        new_v_list += np.split(new_v.cpu().numpy(), range(1, num), axis=0)
        try:
            new_p_opt += p_opt.cpu().numpy().tolist()
            new_d_opt += d_opt.cpu().numpy().tolist()
        except TypeError:
            new_p_opt.append(p_opt)
            new_d_opt.append(d_opt)

        head = tail
        J = len(bowt_list[tail if tail < len(bowt_list) else -1].tolist())
        tail = head + max(int(gran / (I * J)), 1)
        assert head < tail
    return new_u_list, new_v_list, new_p_opt, new_d_opt


def calc_sample(bowsl, bowtl, costmat_l, round=3, device=torch.device("cpu")):
    num = len(bowsl)
    sllist = [len(bows) for bows in bowsl]
    I = max(sllist)
    tllist = [len(bowt) for bowt in bowtl]
    J = max(tllist)

    C = np.ones([num, I, J]) * np.inf
    mu = np.zeros([num, I, 1])
    nu = np.zeros([num, 1, J])

    for numi in range(num):
        C[numi, :sllist[numi], :tllist[numi]] = costmat_l[numi][0, :sllist[numi], :tllist[numi]]
        mu[numi, :sllist[numi], 0] = bowsl[numi]
        nu[numi, 0, :tllist[numi]] = bowtl[numi]

    dx = np.ones([num, I, 1])
    dx[mu == 0] = 0
    dy = np.ones([num, 1, J])
    dy[nu == 0] = 0

    CTensor = torch.from_numpy(C).to(device)
    muTensor = torch.from_numpy(mu).to(device)
    nuTensor = torch.from_numpy(nu).to(device)
    dxTensor = torch.from_numpy(dx).to(device)
    dyTensor = torch.from_numpy(dy).to(device)

    p_opts = wfr_dist_approx_2(CTensor, muTensor, nuTensor, dxTensor, dyTensor, round, device=device).cpu().numpy().tolist()
    return p_opts


def knn_classifier(test_sample, train_sample, coef, dist_mat_gran=50, wfr_mat_gran=2000, device=torch.device("cpu")):
    """
    knn classifier
    :param test_sample: one tuple of vbowy
    :param train_sample: list of tuple of vbowy
    :param dist_mat_gran: calculate the distance
    :param wfr_mat_gran: calculate the distance
    :return:
    """
    # dist mat calculation

    bows_list, bowt_list, costmat_list, yt_list = prepare_bow_dist(test_sample, train_sample, dist_mat_gran, coef, device=device)

    if bows_list is None:
        return None

    # wfr calculation

    def head_filter(bows_list, bowt_list, costmat_list, u_list, v_list, yt_list, dp_list, kp_list, upper_bound):
        new_package = [(bows, bowt, costmat, u, v, yt, dp, kp)
                       for bows, bowt, costmat, u, v, yt, dp, kp in zip(bows_list, bowt_list, costmat_list, u_list, v_list, yt_list, dp_list, kp_list)
                       if not dp > upper_bound]

        new_package = sorted(new_package, key=lambda x_: x_[-1])
        (bows_list, bowt_list, costmat_list, u_list, v_list, yt_list, dp_list, kp_list) = [
            [bows for bows, bowt, costmat, u, v, yt, dp, kp in new_package],
            [bowt for bows, bowt, costmat, u, v, yt, dp, kp in new_package],
            [costmat for bows, bowt, costmat, u, v, yt, dp, kp in new_package],
            [u for bows, bowt, costmat, u, v, yt, dp, kp in new_package],
            [v for bows, bowt, costmat, u, v, yt, dp, kp in new_package],
            [yt for bows, bowt, costmat, u, v, yt, dp, kp in new_package],
            [dp for bows, bowt, costmat, u, v, yt, dp, kp in new_package],
            [kp for bows, bowt, costmat, u, v, yt, dp, kp in new_package]
        ]
        return bows_list, bowt_list, costmat_list, u_list, v_list, yt_list, dp_list, kp_list

    def sort_filter(bows_list, bowt_list, costmat_list, u_list, v_list, yt_list, dp_list, kp_list):
        if u_list is not None:
            new_package = [(bows, bowt, costmat, u, v, yt, dp, kp)
                           for bows, bowt, costmat, u, v, yt, dp, kp in zip(bows_list, bowt_list, costmat_list, u_list,
                                                                            v_list, yt_list, dp_list, kp_list)]
            new_package = sorted(new_package, key=lambda x_: len(x_[1]), reverse=True)
            (bows_list, bowt_list, costmat_list, u_list, v_list, yt_list, dp_list, kp_list) = [
                [bows for bows, bowt, costmat, u, v, yt, dp, kp in new_package],
                [bowt for bows, bowt, costmat, u, v, yt, dp, kp in new_package],
                [costmat for bows, bowt, costmat, u, v, yt, dp, kp in new_package],
                [u for bows, bowt, costmat, u, v, yt, dp, kp in new_package],
                [v for bows, bowt, costmat, u, v, yt, dp, kp in new_package],
                [yt for bows, bowt, costmat, u, v, yt, dp, kp in new_package],
                [dp for bows, bowt, costmat, u, v, yt, dp, kp in new_package],
                [kp for bows, bowt, costmat, u, v, yt, dp, kp in new_package]
            ]
            return bows_list, bowt_list, costmat_list, u_list, v_list, yt_list, dp_list, kp_list
        else:
            new_package = [(bows, bowt, costmat, yt)
                           for bows, bowt, costmat, yt in zip(bows_list, bowt_list, costmat_list, yt_list)]
            new_package = sorted(new_package, key=lambda x_: len(x_[1]), reverse=True)
            (bows_list, bowt_list, costmat_list, yt_list) = [
                [bows for bows, bowt, costmat, yt in new_package],
                [bowt for bows, bowt, costmat, yt in new_package],
                [costmat for bows, bowt, costmat, yt in new_package],
                [yt for bows, bowt, costmat, yt in new_package],
            ]
            return bows_list, bowt_list, costmat_list, None, None, yt_list, None, None

    u_list, v_list = None, None
    dp_list, kp_list = None, None

    for r in range(1, 6):
        # init with out any special
        # if device.type is "cpu":
        p_opts = []
        if int(args.cuda) < 0:
            for i in range(knnk):
                p_op = calc_sample(bows_list[i:i+1], bowt_list[i:i+1], costmat_list[i:i+1], r)
                p_opts += p_op
        else:
            p_opts = calc_sample(bows_list[:knnk], bowt_list[:knnk], costmat_list[:knnk], r)
        # count the first 40 samples
        crt_upper_bound = max(p_opts)
        # calculate the full list
        (bows_list, bowt_list, costmat_list,
         u_list, v_list, yt_list, dp_list, kp_list) = sort_filter(bows_list, bowt_list, costmat_list, u_list, v_list,
                                                                  yt_list, dp_list, kp_list)

        u_list, v_list, kp_list, dp_list = one_round_iteration(bows_list, bowt_list, costmat_list, wfr_mat_gran,
                                                               1/math.e**(r+1), 32*r, u_list, v_list)
        # filter the sequence by dp, sort by KP
        (bows_list, bowt_list, costmat_list,
         u_list, v_list, yt_list, dp_list, kp_list) = head_filter(bows_list, bowt_list, costmat_list, u_list, v_list,
                                                                  yt_list, dp_list, kp_list, crt_upper_bound)
        # print(len(bows_list))
        assert len(bows_list) >= knnk

    return yt_list[:knnk]

def get_knn_results(label, yknn):
    if yknn is None:
        return np.ones((knnk,))
    else:
        yknn = [int(y) for y in yknn]
    nlabels = max(yknn)
    vote_box = np.zeros((nlabels+1,))
    knn_label = np.zeros((knnk,))
    for i in range(knnk):
        y = yknn[i]
        vote_box[y] += 1
        maxCount = np.amax(vote_box)
        if np.sum(maxCount == vote_box) > 1:
            knn_label[i] = knn_label[i-1]
        else:
            knn_label[i] = np.argmax(vote_box)

    return knn_label == label


def run_knn(dataset, coef, trainratio, testratio, distgran, wfrgran, j=9999999, device=torch.device("cpu")):
    he = KNNHeldoutEvaluation(dataset)
    test_vbowy = he.get_test_set(ratio=testratio)
    test_size = len(test_vbowy)
    knn_res = np.zeros((test_size, knnk))
    # evaluate each test point
    train_vbowys = he.get_random_train_split(ratio=trainratio)
    init_time = time.time()
    old_time = time.time()
    for i, tvbowy in enumerate(test_vbowy):
        if i >= j:
            return
        *_, ytest = tvbowy
        yknn = knn_classifier(tvbowy, train_vbowys, coef, dist_mat_gran=distgran, wfr_mat_gran=wfrgran, device=device)
        knn_res[i,:] = get_knn_results(ytest, yknn)
        error_rates = 1 - np.mean(knn_res[:i+1, :], axis=0)
        min_error_rates = np.min(error_rates)
        new_time = time.time()
        print("{:5} of {:5} evaluated, error_rates {:.4f}, elapse {:4f}, total_elapse {:4f}".format(
              i, test_size, min_error_rates, new_time-old_time, new_time-init_time), end="\n" if i % 100 == 0 else "\r")
        old_time = new_time

    print("random split evaluated, error_rates {:.4f}, total_elapse {:4f}".format(
          min_error_rates, new_time-init_time))

def prune_stat(coef, trainratio, testratio, distgran, wfrgran, device=torch.device("cpu")):
    for datasetID in availableDatasetIDs:
        print(datasetID)
        run_knn(datasetID, coef, trainratio, testratio, distgran, wfrgran, j=5, device=device)


if __name__=="__main__":

    # print(datasetID, coef)
    datasetID = args.dataset
    coef = args.coef
    trainratio = args.trainratio
    testratio = args.testratio
    distgran = args.distgran
    wfrgran = args.wfrgran
    # prune_stat(coef, trainratio, testratio, distgran, wfrgran, device=args.device)
    run_knn(datasetID, coef, trainratio, testratio, distgran, wfrgran)

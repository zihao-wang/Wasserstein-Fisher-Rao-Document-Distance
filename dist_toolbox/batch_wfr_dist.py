import numpy as np
import torch

from .wfr_dist import wfr_dist_approx


# batch handle the wfr distance
def batch_wfr_dist_adhoc(bowsl, xsl, bowtl, xtl, round=3, coef=1):
    """
    this adhoc method calculates the distance matrices to reduce the IO cost
    :param bowsl: lsit of source bow
    :param xsl: list of source embedding representation
    :param bowtl: list of target bow
    :param xtl: list of target embedding representation
    :return: wfrdl for the list of wfr distance
    """
    assert len(bowsl) == len(bowtl)
    assert len(xsl) == len(xtl)
    assert len(bowsl) == len(xsl)
    assert len(bowtl) == len(xtl)

    num = len(bowsl)
    dim = xsl[0].shape[1]

    sllist = [len(list(bows.tolist())) for bows in bowsl]
    tllist = [len(list(bowt.tolist())) for bowt in bowtl]

    I, J = max(sllist), max(tllist)

    # calculate bow tensor and dxdy tensor

    bowsArray = np.zeros([num, I, 1])
    for i in range(num):
        bowsArray[i, :sllist[i], 0] = bowsl[i]

    bowtArray = np.zeros([num, 1, J])
    for i in range(num):
        bowtArray[i, 0, :tllist[i]] = bowtl[i]

    bowsTensor = torch.from_numpy(bowsArray).cuda()
    bowtTensor = torch.from_numpy(bowtArray).cuda()

    dxArray = np.ones([num, I, 1])  # / np.asarray(sllist).reshape([num, 1, 1])
    dxArray[bowsArray == 0] = 0
    dyArray = np.ones([num, 1, J])  # / np.asarray(tllist).reshape([num, 1, 1])
    dyArray[bowtArray == 0] = 0

    dxTensor = torch.from_numpy(dxArray).cuda()
    dyTensor = torch.from_numpy(dyArray).cuda()

    # calculate distance tensor


    DArray = np.ones([num, I, J]) * np.inf
    for i in range(num):
        torch.cuda.empty_cache()
        xsTensor = torch.from_numpy(xsl[i].reshape(sllist[i], 1, dim).astype(np.double)).cuda()
        xtTensor = torch.from_numpy(xtl[i].reshape(1, tllist[i], dim).astype(np.double)).cuda()
        DArray[i, :sllist[i], :tllist[i]] = torch.sqrt(torch.sum((xsTensor - xtTensor) ** 2, -1)).cpu().numpy()

    DTensor = torch.from_numpy(DArray).cuda()

    wfrdl = wfr_dist_approx(D=DTensor, mu=bowsTensor, nu=bowtTensor, dx=dxTensor, dy=dyTensor, round=round, coef=coef)
    return wfrdl

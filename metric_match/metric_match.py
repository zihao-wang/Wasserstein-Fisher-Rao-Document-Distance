import os
import json
import pickle
import sys
sys.path.append("..")
import numpy as np
from scipy.spatial import distance_matrix
import ot

from metric_match_data_helper import DatasetLoader

from dist_toolbox import batch_wfr_dist_adhoc

tmp_dir = 'tmp'



def save_case(bowsl, xsl, bowtl, xtl, ret_label, i):
    with open(os.path.join(tmp_dir, "case{}.pickle".format(i)), 'wb') as f:
        pickle.dump((bowsl, xsl, bowtl, xtl, ret_label), f)


def load_case(i):
    with open(os.path.join(tmp_dir, "case{}.pickle".format(i)), 'rb') as f:
        bowsl, xsl, bowtl, xtl, ret_label = pickle.load(f)
    return bowsl, xsl, bowtl, xtl, ret_label


def batch_wmd_dist_adhoc(bowsl, xsl, bowtl, xtl):
    dlist = []
    for bows, xs, bowt, xt in zip(bowsl, xsl, bowtl, xtl):
        bows = np.asarray(bows)
        a = bows / np.sum(bows)
        bowt = np.asarray(bowt)
        b = bowt / np.sum(bowt)
        D = distance_matrix(xs, xt)
        d = ot.emd2(a, b, D)
        dlist.append(d)
    return dlist


def eval(coef):
    dataset = DatasetLoader()
    dataset.load()
    bowsl, xsl, bowtl, xtl, ret_label = dataset.get_pairs()
    assert bowsl is not None
    assert xsl is not None
    assert bowtl is not None
    assert xtl is not None
    assert ret_label is not None
    step = 600
    wfr_list = []
    for i in range(0, len(bowsl), step):
        bows, xs, bowt, xt = bowsl[i: i+step], xsl[i: i+step], bowtl[i: i+step], xtl[i: i+step]
        nbows = [(bow+1)/np.sum(bow+1) for bow in bows]
        nbowt = [(bow+1)/np.sum(bow+1) for bow in bowt]
        print(i)
        wfr_list += batch_wfr_dist_adhoc(nbows, xs, nbowt, xt, 5, coef=coef).cpu().numpy().tolist()
    print(wfr_list)
    out = sorted(zip(wfr_list, ret_label), key=lambda x_: x_[0])
    with open("wfr{}.out".format(coef), 'wt') as f:
        for d, l in out:
            f.write("{:.4f}\t{}\n".format(d,l))

def calc_prf(coef):
    with open("wfr{}.out".format(coef), 'rt') as f:
        ordered_l = []
        for line in f.readlines():
            d, l = line.strip().split()
            ordered_l.append(int(l))
    prf1 = []
    maxp = 0
    best_f1 = 0
    ones = np.sum(ordered_l)
    for i in range(len(ordered_l)):
        p = np.sum(ordered_l[:i])/i
        r = np.sum(ordered_l[:i])/ones
        f1 = 2 * p * r / (p + r)
        prf1.append((p, r, f1))
        if p > maxp:
            maxp = p
        if f1 > best_f1:
            best_f1 = f1
    with open("prf1{}.out".format(coef), 'wb') as f:
        pickle.dump(prf1, f)
    print(best_f1, maxp)

def draw_prf(coef):
    with open("prf1{}.out".format(coef), 'rb') as f:
        prf1 = pickle.load(f)


if __name__ == "__main__":
    for c in [0.125, 0.25, 0.5, 1, 1.5, 2]:
        print(c)
        eval(c)
        calc_prf(c)


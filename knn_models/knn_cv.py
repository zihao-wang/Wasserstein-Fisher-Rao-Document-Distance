import pickle
import numpy as np
import argparse

import scipy.io as sio

from knn_dataset_helper import KNNCrossValidation

argParser = argparse.ArgumentParser()
argParser.add_argument("--coef", type=float, default=1.0)
argParser.add_argument("--dataset", default='twitter')

def knn_classifier(target_idy, known_idys, dist_matrix):
    tid, ty = target_idy
    dist_rank_list = []
    for kid, ky in known_idys:
        d = dist_matrix[min(tid, kid), max(tid, kid)]
        dist_rank_list.append((d, ky))

    dist_rank_list = sorted(dist_rank_list, key=lambda x: x[0])
    knn_sorted_list = dist_rank_list[:40]

    return [int(y) for d, y in knn_sorted_list]


def find_most_common(vec):
    l = len(vec)
    while True:
        vec = vec[:l]
        vote_count = {}
        for y in vec:
            if y in vote_count:
                vote_count[y] += 1
            else:
                vote_count[y] = 1
        vote_sort = sorted(vote_count.items(), key=lambda x: x[1], reverse=True)  # key, counts
        if len(vote_sort) > 1 and vote_sort[0][1] == vote_sort[1][1]:
            l -= 1
            continue
        else:
            return vote_sort[0][0]


def eval_knn(datasetID, coef):
    cv = KNNCrossValidation(datasetID)
    with open("{}-{:.4f}.pickle".format(datasetID, coef), 'rb') as f:
        dist_matrix = pickle.load(f)

    # dist_matrix = sio.loadmat('wmd_d_bbcsport.mat')['WMD_D']
    correct_count = np.zeros([cv.num_folds, 40, cv.fold_size])
    for i in range(cv.num_folds):
        train, test = cv.get_cross_validation(i)
        knn_vec = []
        y_vec = []
        for tidy in test:
            knn_vec.append(knn_classifier(tidy, train, dist_matrix))
            y_vec.append(tidy[1])
        for k in range(1, 41):
            mc_vec = []
            for ii, (y, vec) in enumerate(zip(y_vec, knn_vec)):
                mc_keys = find_most_common(vec[:k])
                mc_vec.append(mc_keys)
                if y == mc_keys:
                    correct_count[i, k-1, ii] = 1

    min_err_rate = 1
    opt_k = 0
    for k in range(1, 41):
        k_correct_count = correct_count[:, k-1, :]
        each_err = np.mean(k_correct_count, -1)
        std = np.std(each_err)
        k_err_rate = 1 - np.mean(each_err)
        if k_err_rate < min_err_rate:
            min_err_rate = k_err_rate
            opt_k = k
        print("k {:2d} | err {:.4f} ,std {:.4f} | min err {:.4f} @ k = {} ".format(k, k_err_rate, std, min_err_rate, opt_k))
        if std > 5:
            print(each_err)


if __name__ == "__main__":
    args = argParser.parse_args()
    datasetID = args.dataset
    coef = args.coef
    eval_knn(datasetID, coef)

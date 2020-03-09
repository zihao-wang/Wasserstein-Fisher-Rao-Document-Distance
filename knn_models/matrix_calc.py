import sys
import pickle
import time
import os
import argparse

sys.path.append("..")

import numpy as np

from dist_toolbox import batch_wfr_dist_adhoc
from knn_dataset_helper import KNNCrossValidation, availableDatasetIDs

argParser = argparse.ArgumentParser()
argParser.add_argument("--cuda", default='0')
argParser.add_argument("--coef", type=float, default=1.0)
argParser.add_argument("--dataset", default='twitter')


def computeMatrix(datasetID, coef):
    cv = KNNCrossValidation(datasetID)

    vbowys = cv.get_all()
    size = cv.size
    step = size // 4 + 1

    print(size)

    wfr_dist_mat = np.zeros([size, size])

    last_cursor_time = -1
    last_batch_time = -1

    start_time = time.time()
    for ii in range(size):

        this_cursor_time = time.time()
        cursor_elapse = this_cursor_time - last_cursor_time
        last_cursor_time = this_cursor_time
        total_consume = this_cursor_time - start_time
        estimate_left = total_consume / (ii + 1) * (size - ii)

        iivbowy = vbowys[ii]
        # bow = np.asarray(bow).reshape([-1])
        for jj in range(ii + 1, size, step):
            jjvbowys = vbowys[jj: jj + step]
            # set meta number
            num = len(jjvbowys)

            #################### prepare for lists #################################
            xs, bows, _ = iivbowy
            bowsl = [bows / np.sum(bows)] * num
            xsl = [xs] * num
            bowtl, xtl = [], []
            for xt, bowt, _ in jjvbowys:
                bowtl.append(bowt / np.sum(bowt))
                xtl.append(xt)
            ########################################################################

            gpu_start = time.time()
            wfr_dist_mat[ii, jj: jj + num] = batch_wfr_dist_adhoc(bowsl, xsl, bowtl, xtl, coef=coef).cpu().numpy()
            gpu_elapse = time.time() - gpu_start

            this_batch_time = time.time()
            batch_elapse = this_batch_time - last_batch_time
            last_batch_time = this_batch_time

            print("cursor position: {:4d} / {:4d}, range ({:4d}, {:4d}) / {:4d} "
                  "batch elapse {:.4f} | gpu elapse {:.4f} | cursor elapse {:.4f} |"
                  " total_consume {:.4f} | estimate time left {:.4f}".format(
                ii, size, jj, jj + num, size, batch_elapse, gpu_elapse, cursor_elapse, total_consume, estimate_left),
                end='\r')

    with open("{}-{:.4f}.pickle".format(datasetID, coef), "wb") as f:
        pickle.dump(wfr_dist_mat, f)

    return wfr_dist_mat


if __name__ == "__main__":
    args = argParser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    datasetID = args.dataset
    coef = args.coef
    print(datasetID, coef)
    computeMatrix(datasetID, coef)

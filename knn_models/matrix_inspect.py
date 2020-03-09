import pickle
import numpy as np
import argparse

import scipy.io as sio

argParser = argparse.ArgumentParser()
argParser.add_argument("--coef", type=float, default=1.0)
argParser.add_argument("--dataset", default='twitter')

if __name__ == "__main__":
    args = argParser.parse_args()
    datasetID = args.dataset
    coef = args.coef
    with open("{}-{:.4f}.pickle".format(datasetID, coef), 'rb') as f:
        dist_matrix = pickle.load(f)
    for i in range(10):
        for j in range(10):
            print(dist_matrix[i, j], end=', ')
        print()



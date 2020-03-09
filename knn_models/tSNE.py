import argparse
import pickle
import random
import numpy as np
from sklearn.preprocessing import normalize
from sklearn import manifold
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import numpy

from knn_dataset_helper import KNNCrossValidation

argParser = argparse.ArgumentParser()
argParser.add_argument("--coef", type=float, default=1.5)
argParser.add_argument("--dataset", default='bbcsport')


if __name__=="__main__":
    args = argParser.parse_args()
    datasetID = args.dataset
    coef = args.coef

    cv = KNNCrossValidation(dataset_id=datasetID)
    labels = cv.get_all_label()
    num_label = max(labels)
    with open("{}-{:.4f}.pickle".format(datasetID, coef), 'rb') as f:
        dist_matrix = pickle.load(f)
    D = dist_matrix + dist_matrix.T
    label = labels

    # index = random.sample(range(len(labels)), 1000)
    # D = D[index, :][:, index]
    # label = labels[index]

    model = manifold.TSNE(
        n_components=2,
        # n_iter=250,
        perplexity=50,
        metric='precomputed')
    cmap = plt.get_cmap('Set1')
    colors = [cmap(l) for l in label]
    print("fitting ...")
    coords = model.fit_transform(D)
    print("fitted")
    plt.scatter(coords[:,0], coords[:,1], c=colors, s=0.5)
    plt.show()



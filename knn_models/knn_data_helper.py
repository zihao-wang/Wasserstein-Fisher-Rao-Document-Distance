import argparse
import os
import scipy.io as sio
import pickle
from random import shuffle, seed
from collections import OrderedDict

import torch
from torch.autograd import Variable

import numpy as np


seed(19950820)
argParser = argparse.ArgumentParser()
argParser.add_argument("--dataset", default='twitter')

dataDir = "../data"
datasetName = "knn_icml15"
datasetDir = os.path.join(dataDir, datasetName)

availableDatasetNames = [
    "20ng2_500-emd_tr_te.mat",
    "amazon-emd_tr_te_split.mat",
    "bbcsport-emd_tr_te_split.mat",
    "classic-emd_tr_te_split.mat",
    "ohsumed-emd_tr_te_ix.mat",
    "r8-emd_tr_te3.mat",
    "recipe2-emd_tr_te_split.mat",
    "twitter-emd_tr_te_split.mat"
]

singleSplit = ["amazon-emd_tr_te_split.mat",
               "bbcsport-emd_tr_te_split.mat",
               "classic-emd_tr_te_split.mat",
               "recipe2-emd_tr_te_split.mat",
               "twitter-emd_tr_te_split.mat"]

availableDatasetIDs = [
    "20ng2_500",
    "amazon",
    "bbcsport",
    "classic",
    "ohsumed",
    "r8",
    "recipe2",
    "twitter"
]

id2file = {k: v for k, v in zip(availableDatasetIDs, availableDatasetNames)}


class WordIdEmbedding:
    def __init__(self):
        self.wordIdDict = OrderedDict()
        self.word2DF = {}
        self.wordEmbDict = OrderedDict()

    def add_word_X_pair(self, word, X):
        print("add dict")
        numPointClouds = len(word)
        numPointClouds_ = len(X)
        # assert numPointClouds == numPointClouds_
        wi, xj = 0, 0

        matched_word = []
        matched_x = []
        while wi < numPointClouds and xj < numPointClouds_:
            try:
                wordseq = word[wi]
                word_seq = wordseq.tolist()[0]
            except:
                print("jump empty")
                wi += 1
                wordseq = word[wi]
                word_seq = wordseq.tolist()[0]

            xseq = X[xj]
            # assert len(word_seq) == xseq.shape[1]
            for i, w_ in enumerate(word_seq):
                w = w_[0]
                if w == 0:
                    continue
                if w not in self.wordIdDict:
                    self.wordIdDict[w] = len(self.wordIdDict)
                    self.wordEmbDict[w] = xseq[:, i]
            assert len(self.wordIdDict) == len(self.wordEmbDict)

            matched_word.append(wordseq)
            matched_x.append(xseq)

            wi += 1
            xj += 1
        return matched_word, matched_x

    def dump_matrix(self):
        print("calculate matrix")
        emb_mat = np.zeros([len(self.wordEmbDict), 300])
        for i, v in enumerate(self.wordEmbDict.values()):
            emb_mat[i, :] = v

        emb_mat = torch.from_numpy(emb_mat).cuda()

        def row_pairwise_distances(x, y=None, dist_mat=None):
            if y is None:
                y = x
            if dist_mat is None:
                dtype = x.data.type()
                dist_mat = Variable(torch.Tensor(x.size()[0], y.size()[0]).type(dtype))

            for i, row in enumerate(x.split(1)):
                r_v = row.expand_as(y)
                sq_dist = torch.sum((r_v - y) ** 2, 1)
                dist_mat[i] = torch.sqrt(sq_dist.view(1, -1))
            return dist_mat

        matrix = row_pairwise_distances(emb_mat).cpu().numpy()
        return matrix

    def words2ids(self, words):
        ids = [self.wordIdDict[w[0]] for w in words.tolist()[0] if not w[0] == 0]
        return ids


class KNNCrossValidation:
    def __init__(self, dataset_id):
        assert dataset_id in availableDatasetIDs
        with open(self.get_pickle_file(dataset_id), 'rb') as f:
            vbowy = pickle.load(f)
        self.vbowy = vbowy

        classes = set()
        for *_, y in vbowy:
            classes.add(y)
        self.num_classes = len(classes)

        self.size = len(self.vbowy)
        self.num_folds = 5
        self.batch_size = 64
        self.indexes = np.arange(self.size)
        # np.random.seed(820)
        np.random.shuffle(self.indexes)

        self.fold_size = self.size // self.num_folds

    def __len__(self):
        return self.size

    @staticmethod
    def get_pickle_file(dataset_id):
        return os.path.join(datasetDir, dataset_id + '.pickle')

    def cross_validation_set(self, num_folds, batch_size):
        self.num_folds = num_folds
        self.batch_size = batch_size
        self.fold_size = self.size // self.num_folds

    def get_cross_validation(self, index):
        assert index < self.num_folds

        test_start = index * self.fold_size
        test_end = (index + 1) * self.fold_size
        train = []
        test = []

        for i, index in enumerate(self.indexes):
            *_, y = self.vbowy[index]
            if i < test_start or i >= test_end:
                train.append((index, y))
            else:
                test.append((index, y))

        return train, test

    def get_all(self):
        return self.vbowy


def get_dataset_id(dataset_file_name):
    return dataset_file_name.split('-')[0]


def get_pickle_file_path(dataset_file_name):
    dataset_id = get_dataset_id(dataset_file_name)
    return os.path.join(datasetDir, dataset_id + '.pickle')


def get_npy_file_path(dataset_file_name):
    dataset_id = get_dataset_id(dataset_file_name)
    return os.path.join(datasetDir, dataset_id + '.npy')


def single_split_parser(dataset_file_name):

    dataset_file_path = os.path.join(datasetDir, dataset_file_name)
    file_contents = sio.loadmat(dataset_file_path)
    bow = file_contents['BOW_X'].tolist()[0]
    X = file_contents['X'].tolist()[0]
    Y = file_contents['Y'].tolist()[0]
    try:
        words = file_contents['words'].tolist()[0]
    except:
        words = file_contents['the_words'].tolist()[0]

    vbowy = []

    for x, b, y in zip(X, bow, Y):
        vbowy.append((x.T, b.reshape((-1)), y))

    with open(get_pickle_file_path(dataset_file_name), 'wb') as f:
        vbowy = sorted(vbowy, key=lambda x_: len(x_[0]), reverse=True)
        pickle.dump(vbowy, f)

    return


def train_test_split_parser(dataset_file_name):

    dataset_file_path = os.path.join(datasetDir, dataset_file_name)
    file_contents = sio.loadmat(dataset_file_path)

    # xtr = file_contents['xtr'].tolist()[0]
    # BOW_xtr = file_contents['BOW_xtr'].tolist()[0]
    # ytr = file_contents['ytr'].tolist()[0]

    xte = file_contents['xte'].tolist()[0]
    BOW_xte = file_contents['BOW_xte'].tolist()[0]
    yte = file_contents['yte'].tolist()[0]
    words_te = file_contents['words_te'].tolist()[0]

    vbowy = []

    for x, b, y in zip(xte, BOW_xte, yte):
        vbowy.append((x.T, b.reshape((-1)), y))

    with open(get_pickle_file_path(dataset_file_name), 'wb') as f:
        vbowy = sorted(vbowy, key=lambda x_: len(x_[0]), reverse=True)
        pickle.dump(vbowy, f)

    return


def parse_dataset(datasetID):
    d = id2file[datasetID]
    if d in singleSplit:
        single_split_parser(d)
    else:
        train_test_split_parser(d)
    return


if __name__ == "__main__":
    args = argParser.parse_args()
    datasetID = args.dataset
    print("parse the dataset {} into pickle".format(datasetID))
    parse_dataset(datasetID)

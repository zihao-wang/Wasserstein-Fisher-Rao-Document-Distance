import argparse
import os
import scipy.io as sio
import pickle
import random

import numpy as np

argParser = argparse.ArgumentParser()
argParser.add_argument("--dataset", default='twitter')

dataDir = "../data"
datasetName = "knn_icml15"
datasetDir = os.path.join(dataDir, datasetName)

# There are two kinds of pipelines
# 1. for data sample generation
# 2. for set generation

# Each sample is a tuple (sampleID, bow, wordID, label)

# provide calling interfaces for following tasks.
# 1. full sample cross validation
# 2. full and lack-sample held-out validation

# Notes: the variables contains inside different datasets
# 20ng_2: not use ICML2015 split, use the sort by date split, evaluate lastly


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


class KNNHeldoutEvaluation:
    def __init__(self, dataset_id):
        assert dataset_id in availableDatasetIDs
        datasetFile = id2file[dataset_id]
        if datasetFile in singleSplit:
            with open(get_pickle_file_path(dataset_id), 'rb') as f:
                vbowy = pickle.load(f)
            random.shuffle(vbowy)
            total_num = len(vbowy)
            train_size = int(0.7 * total_num)
            test_size = total_num - train_size
            self.train_vbowy = vbowy[:train_size]
            self.test_vbowy = vbowy[-test_size:]

        else:
            with open(get_pickle_file_path(dataset_id, "train"), 'rb') as f:
                vbowy = pickle.load(f)
            self.train_vbowy = vbowy
            with open(get_pickle_file_path(dataset_id, "test"), 'rb') as f:
                vbowy = pickle.load(f)
            self.test_vbowy = vbowy

    def get_random_train_split(self, ratio=1):
        if ratio < 1:
            total_num = len(self.train_vbowy)
            sample_num = int(ratio * total_num)
            sampled_index = random.sample(range(total_num), sample_num)
            ret_vbowy = [self.train_vbowy[i] for i in sampled_index]
        else:
            ret_vbowy = self.train_vbowy
        ret_vbowy = sorted(ret_vbowy, key=lambda x_: len(np.asarray(x_[1]).reshape((-1,)).tolist()), reverse=True)
        return ret_vbowy

    def get_test_set(self, ratio=1):
        # random.shuffle(self.test_vbowy)
        ret_vbowy = self.test_vbowy[:int(ratio * len(self.test_vbowy))]
        ret_vbowy = sorted(ret_vbowy, key=lambda x_: len(np.asarray(x_[1]).reshape((-1,)).tolist()))
        return ret_vbowy


class KNNCrossValidation:
    def __init__(self, dataset_id):
        assert dataset_id in availableDatasetIDs
        with open(get_pickle_file_path(dataset_id), 'rb') as f:
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
        np.random.seed(820)
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

    def get_all_label(self):
        return [y for *_, y in self.vbowy]


def get_dataset_id(dataset_file_name):
    return dataset_file_name.split('-')[0]


def get_pickle_file_path(dataset_file_name, str=""):
    dataset_id = get_dataset_id(dataset_file_name)
    return os.path.join(datasetDir, dataset_id + str + '.pickle')


def get_npy_file_path(dataset_file_name):
    dataset_id = get_dataset_id(dataset_file_name)
    return os.path.join(datasetDir, dataset_id + '.npy')


def single_split_all_parse(dataset_file_name):
    dataset_file_path = os.path.join(datasetDir, dataset_file_name)
    file_contents = sio.loadmat(dataset_file_path)
    bow = file_contents['BOW_X'].tolist()[0]
    X = file_contents['X'].tolist()[0]
    Y = file_contents['Y'].tolist()[0]

    vbowy = []

    for x, b, y in zip(X, bow, Y):
        bow = b.reshape((-1,))
        vbowy.append((x.T, bow, y))

    with open(get_pickle_file_path(dataset_file_name), 'wb') as f:
        vbowy = sorted(vbowy, key=lambda x_: len(x_[0]), reverse=True)
        pickle.dump(vbowy, f)

    return


def pre_split_test_parse(dataset_file_name):
    dataset_file_path = os.path.join(datasetDir, dataset_file_name)
    file_contents = sio.loadmat(dataset_file_path)

    # xtr = file_contents['xtr'].tolist()[0]
    # BOW_xtr = file_contents['BOW_xtr'].tolist()[0]
    # ytr = file_contents['ytr'].tolist()[0]

    xte = file_contents['xte'].tolist()[0]
    BOW_xte = file_contents['BOW_xte'].tolist()[0]
    yte = file_contents['yte'].tolist()[0]

    test_vbowy = []

    for x, b, y in zip(xte, BOW_xte, yte):
        bow = b.reshape((-1,))
        test_vbowy.append((x.T, bow, y))

    with open(get_pickle_file_path(dataset_file_name), 'wb') as f:
        test_vbowy = sorted(test_vbowy, key=lambda x_: len(x_[0]), reverse=True)
        pickle.dump(test_vbowy, f)
    return


def pre_split_train_test_parse(dataset_file_name):
    dataset_file_path = os.path.join(datasetDir, dataset_file_name)
    file_contents = sio.loadmat(dataset_file_path)

    xtr = file_contents['xtr'].tolist()[0]
    BOW_xtr = file_contents['BOW_xtr'].tolist()[0]
    ytr = file_contents['ytr'].tolist()[0]

    train_vbowy = []

    for x, b, y in zip(xtr, BOW_xtr, ytr):
        bow = b.reshape((-1,))
        train_vbowy.append((x.T, bow, y))

    with open(get_pickle_file_path(dataset_file_name, "train"), 'wb') as f:
        train_vbowy = sorted(train_vbowy, key=lambda x_: len(x_[0]), reverse=True)
        pickle.dump(train_vbowy, f)


    xte = file_contents['xte'].tolist()[0]
    BOW_xte = file_contents['BOW_xte'].tolist()[0]
    yte = file_contents['yte'].tolist()[0]

    test_vbowy = []

    for x, b, y in zip(xte, BOW_xte, yte):
        bow = b.reshape((-1,))
        test_vbowy.append((x.T, bow, y))

    with open(get_pickle_file_path(dataset_file_name, "test"), 'wb') as f:
        test_vbowy = sorted(test_vbowy, key=lambda x_: len(x_[0]), reverse=True)
        pickle.dump(test_vbowy, f)

    return


def parse_dataset(datasetID):
    d = id2file[datasetID]
    if d in singleSplit:
        single_split_all_parse(d)
    else:
        pre_split_test_parse(d)
        pre_split_train_test_parse(d)
    return


def dataset_report():
    for datasetID in availableDatasetIDs:
        print("===")
        if datasetID is not '20ng2_500':
            parse_dataset(datasetID)
        print(datasetID)
        he = KNNHeldoutEvaluation(datasetID)
        train_size = len(he.train_vbowy)
        train_lengths = [len(np.asarray(bow).reshape((-1,))) for _, bow, y in he.train_vbowy]
        test_size = len(he.test_vbowy)
        test_lengths = [len(np.asarray(bow).reshape((-1,))) for _, bow, y in he.test_vbowy]
        print(train_size, np.mean(train_lengths), np.std(train_lengths))
        print(test_size, np.mean(test_lengths), np.std(test_lengths))


if __name__ == "__main__":
    args = argParser.parse_args()
    datasetID = args.dataset
    parse_dataset(datasetID)


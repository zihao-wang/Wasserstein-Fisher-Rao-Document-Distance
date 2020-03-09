import os
import pickle

import numpy as np
import scipy.io as sio
from gensim.models import KeyedVectors

datasetDir = "..\\data\\concept-project"
conceptsFile = os.path.join(datasetDir, 'concepts.txt')
projectsFile = os.path.join(datasetDir, 'projects.txt')
annotateFile = os.path.join(datasetDir, 'annotations.txt')

cout = "concept_out.mat"
pout = "project_out.mat"


class word2id:
    def __init__(self):
        self.word2id = {}
        self.vocab = {}

    def save_vocabulary(self):
        with open('vocabulary.txt', 'wt', encoding='utf8') as f:
            f.write("{}\n".format(len(self.word2id)))
            word_id_list = sorted(self.word2id.items(), key=lambda x_: x_[1])
            for word, id in word_id_list:
                f.write("{}\n".format(word))

    def get_vocab(self, kv):
        self.vocab = {}
        for word, id in self.word2id.items():
            try:
                self.vocab[id] = kv[word]
            except:
                print(word)
                self.vocab[id] = np.zeros((300,))
        return self.vocab

    def add_word_seq(self, word_seq):
        for w in word_seq:
            if w not in self.word2id:
                self.word2id[w] = len(self.word2id)

    def map_word_seq(self, word_seq):
        id_seq = [self.word2id[w] for w in word_seq]
        bow_dict = {}
        for i in id_seq:
            if i in bow_dict:
                bow_dict[i] += 1
            else:
                bow_dict[i] = 0

        bow = []
        idv = []
        for k, v in bow_dict.items():
            bow.append(v)
            idv.append(k)

        ret_bow = np.asarray(bow, dtype=np.int)
        ret_idv = np.asarray(idv, dtype=np.int)
        return ret_bow, ret_idv

class DatasetLoader:
    def __init__(self):
        self.vocab = None
        self.cbow = None
        self.pbow = None
        self.cidv = None
        self.cidv = None

    def parse(self):
        w2i = word2id()
        concepts, projects, annotate = [], [], []
        with open(conceptsFile, 'rt', encoding='utf8') as cf:
            for l in cf.readlines():
                content = l.strip().split(':')[-1].split()
                concepts.append(content)
                w2i.add_word_seq(content)

        with open(projectsFile, 'rt', encoding='utf8') as pf:
            for l in pf.readlines():
                content = l.strip().split('\t||\t')[-1].split()[1:]
                projects.append(content)
                w2i.add_word_seq(content)

        with open(annotateFile, 'rt', encoding='utf8') as af:
            for l in af.readlines():
                label = int(l.strip())
                annotate.append(int(label))

        kv = KeyedVectors.load_word2vec_format("../../WFRlib/src/matlab/wiki-news-300d-1M.vec")
        self.vocab = w2i.get_vocab(kv)
        self.size = len(annotate)
        self.anno = annotate
        self.cbow, self.cidv = [], []
        for concept in concepts:
            bow, idv = w2i.map_word_seq(concept)
            self.cbow.append(bow)
            self.cidv.append(idv)

        self.pbow, self.pidv = [], []
        for project in projects:
            bow, idv = w2i.map_word_seq(project)
            self.pbow.append(bow)
            self.pidv.append(idv)

    def save(self):
        with open("dataset.pickle", 'wb') as f:
            pickle.dump((self.vocab, self.cbow, self.cidv, self.pbow, self.pidv, self.anno), f)

    def load(self):
        with open("dataset.pickle", 'rb') as f:
            self.vocab, self.cbow, self.cidv, self.pbow, self.pidv, self.anno = pickle.load(f)

    def get_pairs(self):
        self.cx = []
        self.px = []
        for cidv, pidv in zip(self.cidv, self.pidv):
            cxx = np.concatenate([self.vocab[i].reshape((1, -1)) for i in cidv])
            self.cx.append(cxx)
            pxx = np.concatenate([self.vocab[i].reshape((1, -1)) for i in pidv])
            self.px.append(pxx)
        return self.cbow, self.cx, self.pbow, self.px, self.anno


if __name__ == "__main__":
    dl = DatasetLoader()
    print("begin parse ... ")
    dl.parse()
    print("dataset parsed")
    dl.save()
    print("save")

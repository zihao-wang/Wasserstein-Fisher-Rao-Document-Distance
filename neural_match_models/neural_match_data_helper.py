from random import shuffle, seed

import torch

seed(6918)


class DatasetLoader:
    def __init__(self, record, batchSize):
        shuffle(record)
        self.record = record
        self.batchSize = batchSize
        self.train = record[0: int(len(record) * 0.6)]
        self.trainBatchNum = len(self.train) // self.batchSize + 1
        self.validate = record[int(len(record) * 0.6): -1]
        # self.test = record[int(len(record) * 1): len(record)]

    def _pack_matrix(self, releventRecords):

        length = len(releventRecords)

        packed = []

        for i in range(length // self.batchSize):
            if (i + 1) * self.batchSize < length:
                _releventRecords = releventRecords[i * self.batchSize: (i + 1) * self.batchSize]
            else:
                _releventRecords = releventRecords[i * self.batchSize: -1]
            annotationArray = torch.tensor([a for p, c, a in _releventRecords], dtype=torch.float)
            projectList = [p for p, c, a in _releventRecords]
            batchMaxProjectLength = max([len(p) for p in projectList])
            projectMatrix = torch.tensor([p + [0] * (batchMaxProjectLength - len(p)) for p in projectList])
            conceptList = [c for p, c, a in _releventRecords]
            batchMaxConceptLength = max([len(c) for c in conceptList])
            conceptMatrix = torch.tensor([c + [0] * (batchMaxConceptLength - len(c)) for c in conceptList])
            packed.append((projectMatrix, conceptMatrix, annotationArray))

        return packed

    def get_validation(self):
        return self._pack_matrix(self.validate)

    def get_test(self):
        return self._pack_matrix(self.test)

    def get_train(self):
        # get each list first, and must padding it into matrix
        # projectMatrix = None  # [batchSize, batchMaxProjectLength]
        # conceptMatrix = None  # [batchSize, batchMaxConceptLength]
        # annotationArray = None  # [batchSize]
        # print("get batch", i)

        return self._pack_matrix(self.train)

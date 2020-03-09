import os
import json
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from uotarcii import UotArcII, UotGuidedArcII
from arcii import ArcII
from neural_match_data_helper import DatasetLoader

# user specific
batchSize = 4
epochNum = 40
datasetName = 'concept-project'
# datasetName = 'mrpc'
dimWordEmbedding = 10

# automotive
dataDir = "data"
dataset = os.path.join(dataDir, datasetName)

with open(os.path.join(dataset, 'dict.json'), 'rt') as f:
    dictionary = json.load(f)

numWordEmbedding = len(dictionary)

with open(os.path.join(dataset, 'record.pickle'), 'rb') as f:
    records = pickle.load(f)

with torch.cuda.device(0):
    dataset = DatasetLoader(records, batchSize)
    losses = []
    loss_function = nn.BCELoss()
    model = UotGuidedArcII(numWordEmbedding, dimWordEmbedding)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # model = ArcII(numWordEmbedding, dimWordEmbedding)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.cuda('cuda:0')

    for e in range(epochNum):
        total_loss = 0

        for index, (projectMatrix, conceptMatrix, annotationArray) in enumerate(dataset.get_train()):
            projectMatrix = projectMatrix.cuda('cuda:0')
            conceptMatrix = conceptMatrix.cuda('cuda:0')
            annotationArray = annotationArray.cuda('cuda:0')
            model.zero_grad()
            logits = model(conceptMatrix, projectMatrix)
            loss = loss_function(logits, annotationArray) + 0.1 * torch.mean(model.regular * (annotationArray-0.5))
            loss.backward()

            # model.parameters().grad.data.clamp_(-1, 1)

            optimizer.step()
            batch_loss = loss.item()
            total_loss += batch_loss
            print("batch {:4d}/{:4d} | loss {:.4f}\r".format(index + 1, dataset.trainBatchNum, batch_loss), end="")
            del projectMatrix, conceptMatrix, annotationArray
        print("epoch {:4d}/{:4d} | loss {:.4f}".format(e + 1, epochNum, total_loss / dataset.trainBatchNum), end="")
        losses.append(total_loss)
        torch.cuda.empty_cache()
        # epoch validation
        preds = []
        labels = []
        for projectMatrix, conceptMatrix, annotationArray in dataset.get_validation():
            projectMatrix = projectMatrix.cuda('cuda:0')
            conceptMatrix = conceptMatrix.cuda('cuda:0')
            annotationArray = annotationArray.cuda('cuda:0')
            vLogits = model(conceptMatrix, projectMatrix)
            pred = torch.ge(vLogits, 0.5).cpu().numpy().tolist()
            label = annotationArray.cpu().numpy().tolist()
            preds += pred
            labels += label

        predsArr = np.asarray(preds)
        labelArr = np.asarray(labels)
        assert predsArr.shape == labelArr.shape

        frac1label = np.sum(labelArr) / len(labelArr)
        accuracy = np.sum(predsArr == labelArr) / np.size(predsArr)
        precision = np.sum((predsArr == labelArr) * predsArr) / np.sum(predsArr)
        recall = np.sum((predsArr == labelArr) * labelArr) / np.sum(labelArr)
        f1score = 2 / (1 / precision + 1 / recall)

        print("| label 1 frac {:.4f} | accuracy {:.4f} | precision {:.4f} | recall {:.4f} | f1score {:.4f} |".format(
            frac1label, accuracy, precision, recall, f1score))

        del projectMatrix, conceptMatrix, annotationArray
        torch.cuda.empty_cache()

    print(total_loss)

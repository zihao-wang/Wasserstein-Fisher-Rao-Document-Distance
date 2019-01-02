import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcII(nn.Module):
    def __init__(self, numWordEmbedding, dimWordEmbedding,):
        super(ArcII, self).__init__()

        # declare embedding layer
        self.embedding = nn.Embedding(numWordEmbedding, dimWordEmbedding).cuda()
        # declare the intra-block self attention (optional)

        # declare cnn layer of cost matrix (optional)
        self.cnn1 = nn.Conv2d(dimWordEmbedding, 16, kernel_size=(3, 3), stride=(1, 1))
        self.pooling1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1))
        self.cnn2 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
        self.pooling2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1))
        # self.cnn3 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
        # self.pooling3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1))

        self.linear = nn.Linear(16, 1)
        self.regular = 0


    def forward(self, concept, project):
        conceptEmb = self.embedding(concept)
        projectEmb = self.embedding(project)
        matrix = self.matrixConstruct(conceptEmb, projectEmb)
        matrix = self.matrixConstruct(conceptEmb, projectEmb)
        matrix = F.relu(self.cnn1(matrix).squeeze())
        matrix = self.pooling1(matrix)
        matrix = F.relu(self.cnn2(matrix).squeeze())
        matrix = self.pooling2(matrix)
        maxByRow, *_ = torch.max(matrix, -1)
        maxByKernel, *_ = torch.max(maxByRow, -1)
        logits = torch.sigmoid(self.linear(maxByKernel))
        return logits.squeeze()

    #
    # def matrixConstruct(self, conceptEmb: torch.Tensor, projectEmb: torch.Tensor):
    #     batchSizeC, maxConceptLength, dimWordEmbeddingC = conceptEmb.shape
    #     batchSizeP, maxProjectLength, dimWordEmbeddingP = projectEmb.shape
    #     assert batchSizeC == batchSizeP
    #     assert dimWordEmbeddingC == dimWordEmbeddingP
    #     conceptExtend = torch.reshape(conceptEmb, [batchSizeC, maxConceptLength, 1, dimWordEmbeddingC])
    #     conceptRepeat = conceptExtend.repeat(1, 1, maxProjectLength, 1)
    #     projectExtend = torch.reshape(projectEmb, [batchSizeP, 1, maxProjectLength, dimWordEmbeddingP])
    #     projectRepeat = projectExtend.repeat(1, maxConceptLength, 1, 1)
    #     matrix = torch.cat([conceptRepeat, projectRepeat], 3)
    #     matrix = torch.transpose(matrix, 1, 3)
    #     # matrix.shape = [batchSize, maxConceptLength, maxProjectLength, 2*dimWordEmbedding]
    #     return matrix


    def matrixConstruct(self, conceptEmb: torch.Tensor, projectEmb: torch.Tensor):
        batchSizeC, maxConceptLength, dimWordEmbeddingC = conceptEmb.shape
        batchSizeP, maxProjectLength, dimWordEmbeddingP = projectEmb.shape
        assert batchSizeC == batchSizeP
        assert dimWordEmbeddingC == dimWordEmbeddingP

        conceptEmb = conceptEmb.reshape([batchSizeC, 1, maxConceptLength, dimWordEmbeddingC])
        projectEmb = projectEmb.reshape([batchSizeP, maxProjectLength, 1, dimWordEmbeddingP])

        matrix = conceptEmb - projectEmb
        matrix = torch.transpose(matrix, 1, 3)
        # matrix.shape = [batchSize, maxConceptLength, maxProjectLength, 2*dimWordEmbedding]
        return matrix

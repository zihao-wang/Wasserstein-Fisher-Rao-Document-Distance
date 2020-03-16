# Wasserstein-Fisher-Rao-Document-Distance

requirement:
- pytorch
- numpy
- scipy

Test KNN Heldout evaluation:

In the knn_models directory

1. `python knn_dataset_helper --dataset=bbcsport`
2. `python knn_he.py --dataset=bbcsport --cuda=0 --distgran=200000 --wfrgran=10000000`

python matrix_calc.py --cuda=0 --dataset=ohsumed --coef=1 && python knn_cv.py --dataset=ohsumed --coef=1
python matrix_calc.py --cuda=0 --dataset=ohsumed --coef=0.5 && python knn_cv.py --dataset=ohsumed --coef=0.5
python matrix_calc.py --cuda=0 --dataset=ohsumed --coef=0.25 && python knn_cv.py --dataset=ohsumed --coef=0.25
python matrix_calc.py --cuda=0 --dataset=ohsumed --coef=0.125 && python knn_cv.py --dataset=ohsumed --coef=0.125

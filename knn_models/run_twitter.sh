python knn_data_helper.py
python matrix_calc.py --coef=1 && python knn_cv.py --coef=1
python matrix_calc.py --coef=0.5 && python knn_cv.py --coef=0.5
python matrix_calc.py --coef=0.25 && python knn_cv.py --coef=0.25
python matrix_calc.py --coef=0.125 && python knn_cv.py --coef=0.125
python matrix_calc.py --coef=0.0625 && python knn_cv.py --coef=0.0625
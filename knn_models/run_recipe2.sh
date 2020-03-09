python knn_data_helper.py --dataset=recipe2
python matrix_calc.py --dataset=recipe2 --coef=1 && python knn_cv.py --dataset=recipe2 --coef=1
python matrix_calc.py --dataset=recipe2 --coef=0.5 && python knn_cv.py --dataset=recipe2 --coef=0.5
python matrix_calc.py --dataset=recipe2 --coef=0.25 && python knn_cv.py --dataset=recipe2 --coef=0.25
python matrix_calc.py --dataset=recipe2 --coef=0.125 && python knn_cv.py --dataset=recipe2 --coef=0.125
python matrix_calc.py --dataset=recipe2 --coef=0.0625 && python knn_cv.py --dataset=recipe2 --coef=0.0625
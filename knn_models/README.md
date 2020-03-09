# Dataset Evaluations

We use KNN method to evaluate different datasets

We have two kinds of datasets

1. datasets that are not splitted so we use cross-validation methods
    1. bbcsport
    2. twitter
    3. recipe
    4. classic
    5. amazon
2. datasets that have been splitted so that we could use random sampling methods to do the classification
    1. ohsumed
    2. reuters
    3. 20news
    
Two tasks will be conducted

1. 5-fold cross validaton that should be conducted on 1st group of dataset
2. random drop training set and do the knn classification

## bbcsport

- WMD: 04.18 % ~ 0.99 %

- WFR: 03.40 % ~ 1.14 %  @ D        | Gain: +0.78  %

## Twitter

- WMD: 28.86 % ~ 1.80 %

- WFR: 31.30 % ~ 1.34 %  @ D        | Gain: -2.44  %
- WFR: 27.38 % ~ 2.41 %  @ D/2      | Gain: +1.48  %
- WFR: 27.67 % ~ 1.16 %  @ D/4      | Gain: +1.19  %
- WFR: 26.76 % ~ 1.48 %  @ D/8      | Gain: +2.10  %    BEST GAIN
- WFR: 28.70 % ~ 1.36 %  @ D/16     | Gain: +0.16  %

## Recipe2

- WMD: 45.69 % ~ 1.70 %

- WFR: 43.07 % ~ 1.91 %  @ 2D       | Gain: +2.65  %
- WFR: 42.68 % ~ 1.52 %  @ 1.5D     | Gain: +3.11  %
- WFR: 43.25 % ~ 2.50 %  @ D        | Gain: +2.44  %    BEST GAI
- WFR: 58.63 % ~ 1.78 %  @ 0.25D    

## Ohsumed

- WFR: 45.77 % ~ 2.29 %  @ 2D
- WFR: 44.99 % ~ 2.55 %  @ 1.75D
- WFR: 44.31 % ~ 2.32 %  @ 1.5D
- WFR: 45.83 % ~ 2.52 %  @ D
- WFR: 46.78 % ~ 2.25 %  @ 0.5D
- WFR: 56.14 % ~ 2.11 %  @ 0.25D


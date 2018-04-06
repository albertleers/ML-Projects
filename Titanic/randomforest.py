import numpy as np
import pandas as pd
import learning_curve as lc
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

train = pd.read_csv('data/randomforest/data_train.csv').as_matrix()
test = pd.read_csv('data/randomforest/data_test.csv').as_matrix()

rf = RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_leaf=2,
                            max_features='sqrt', verbose=0)
lc.plot_learning_curve(rf, train[:, 1:], train[:, 0])



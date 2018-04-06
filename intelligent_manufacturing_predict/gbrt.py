import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import cross_val

train = pd.read_csv('data/03train.csv')
testA = pd.read_csv('data/03testA.csv')
testA_answer = testA[['ID']].copy()

label = train['Y'].copy()
train.drop(['ID', 'Y'], axis=1, inplace=True)
testA.drop('ID', axis=1, inplace=True)
train.sort_index(axis=1, inplace=True)
testA.sort_index(axis=1, inplace=True)


param_grid = [
    {'n_estimators': np.linspace(50, 150, 6, dtype='int'),
     'max_depth': np.linspace(4, 20, 9, dtype='int'),
     'learning_rate': [0.05, 0.1]
     }
]
gbrt = GridSearchCV(GradientBoostingRegressor(n_jobs=-1), param_grid=param_grid,
                   cv=5, scoring='neg_mean_squared_error')
gbrt.fit(train.values, label.values)
print(gbrt.best_params_)


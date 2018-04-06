import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn import preprocessing
import cross_val

train = pd.read_csv('data/05train.csv')
testA = pd.read_csv('data/05testA.csv')
testA_answer = testA[['ID']].copy()

label = train['Y'].copy()
train.drop(['ID', 'Y'], axis=1, inplace=True)
testA.drop('ID', axis=1, inplace=True)
train.sort_index(axis=1, inplace=True)
testA.sort_index(axis=1, inplace=True)

cols = []
for col in train.columns:
    if abs(train[col].corr(label)) < 0.1:
        cols.append(col)

data = pd.concat([train, testA], axis=0)
data.drop(cols, axis=1, inplace=True)
print(data.shape)
data_scaled = preprocessing.StandardScaler().fit_transform(data.values)

# param_grid = {
#     'alpha': np.linspace(10, 100, 10, dtype='int')
# }
# svr = GridSearchCV(SVR(), param_grid=param_grid, cv=5,
#                    scoring='neg_mean_squared_error', verbose=1)
# svr.fit(data_scaled[:500], label.values)
# print(svr.best_params_)

# svr = SVR(**svr.best_params_)
cross_val.cross_validation(SVR(), data_scaled[:500], label.values)
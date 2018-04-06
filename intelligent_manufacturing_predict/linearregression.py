import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
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
    if abs(train[col].corr(label)) < 0.2:
        cols.append(col)

data = pd.concat([train, testA], axis=0)
data.drop(cols, axis=1, inplace=True)
print(data.shape)
train_data = preprocessing.StandardScaler().fit_transform(data.values)

lr = LinearRegression()
cross_val.cross_validation(lr, data[:500], label.values)
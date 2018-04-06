import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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

# param_grid = [
#     {'n_estimators': np.linspace(80, 180, 6, dtype='int'),
#      'max_depth': np.linspace(6, 28, 12, dtype='int'),
#      }
# ]
# rfr = GridSearchCV(RandomForestRegressor(n_jobs=-1), param_grid=param_grid,
#                    cv=5, scoring='neg_mean_squared_error', verbose=1)
rfr = RandomForestRegressor(n_estimators=150, max_depth=3, n_jobs=-1)
cross_val.cross_validation(rfr, train_data[:500], label.values)

# rfr.fit(train.values, label.values)
# joblib.dump(rfr, 'model/04rfr01.pkl')
#
#
# testA_answer['Y'] = rfr.predict(testA.values)
# testA_answer.to_csv('data/04rfr01_testA_answer.csv')



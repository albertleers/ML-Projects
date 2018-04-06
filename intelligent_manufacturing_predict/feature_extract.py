import numpy as np
import pandas as pd
from itertools import chain
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler


train_data = pd.read_csv('data/train.csv')
train_label = train_data[['ID', 'Y']].copy()
train_data.drop('Y', axis=1, inplace=True)
test_data_A = pd.read_csv('data/testA.csv')

# data = pd.concat([train_data, test_data_A], axis=0)

# 删除日期列
date_cols = []
def is_date(s):
    s = str(s)
    if len(s) == 8 and s.startswith('2017'):
        return True
    else:
        return False
for col in train_data.columns:
    if train_data.loc[train_data[col].notnull(), col].apply(is_date).all():
        date_cols.append(col)
train_data.drop(date_cols, axis=1, inplace=True)
test_data_A.drop(date_cols, axis=1, inplace=True)

# 删除值取值唯一的列
val_only_cols = []
for col in train_data.columns:
    if len(train_data[col].unique()) == 1:
        val_only_cols.append(col)
train_data.drop(val_only_cols, axis=1, inplace=True)
test_data_A.drop(val_only_cols, axis=1, inplace=True)

# 将缺失值过多的列的值设为是否缺失
null_cols = []
for col in train_data.columns[1:]:
    if train_data[col].isnull().sum() > 100:
        null_cols.append(col)
        train_data.loc[train_data[col].notnull(), col] = 'yes'
        train_data.loc[train_data[col].isnull(), col] = 'no'
        test_data_A.loc[test_data_A[col].notnull(), col] = 'yes'
        test_data_A.loc[test_data_A[col].isnull(), col] = 'no'

data = pd.concat([train_data, test_data_A], axis=0)
# one-hot
dummies_data = [data]
drop_cols = []
for col in train_data.columns[1:]:
    ser_data = train_data.loc[:, col]
    if ser_data.dtype == np.object:
        if col not in null_cols:
            drop_cols.append(col)
        dummy_data = pd.get_dummies(ser_data, prefix=col)
        dummies_data.append(dummy_data)

data.drop(drop_cols, axis=1, inplace=True)


# 删除缺失值过多的列
data.drop(null_cols, axis=1, inplace=True)

# knn填充缺失值
knn_train_cols = []; knn_label_cols = []

for col in data.columns[1:]:
    if data[col].isnull().any():
        knn_label_cols.append(col)
    else:
        knn_train_cols.append(col)

def fill_knn(df, train_cols, label_cols):
    # del_cols = []
    # for col in train_cols:
    #     if (df[col] == df[col][0]).all():
    #         del_cols.append(col)
    # for col in del_cols:
    #     train_cols.remove(col)
    data = StandardScaler().fit_transform(df.loc[:, train_cols].values)
    data = pd.DataFrame(data, columns=train_cols)
    data.dropna(axis=1, how='any',inplace=True)
    knn_train_cols = [col for col in train_cols if col in data.columns]
    for col in label_cols:
        data_train = data.loc[df[col].notnull().tolist(), knn_train_cols].values
        data_label = df.loc[df[col].notnull().tolist(), col].values
        knn = neighbors.KNeighborsRegressor()
        knn.fit(data_train, data_label)
        df.loc[df[col].isnull(), col] = knn.predict(data.loc[df[col].isnull().tolist(), knn_train_cols].values)

fill_knn(data, knn_train_cols, knn_label_cols)
train_data = data.iloc[0:500].copy()
test_data_A = data.iloc[500:].copy()

train_data['Y'] = train_label['Y']

print(train_data.shape)
print(test_data_A.shape)

train_data.to_csv('data/train.csv', index=None)
test_data_A.to_csv('data/testA.csv', index=None)
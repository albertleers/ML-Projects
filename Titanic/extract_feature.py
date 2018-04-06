import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import re

data_train = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')
data_test['Fare'].fillna(data_test['Fare'].mode()[0], inplace=True)

PassengerId = data_test['PassengerId']

full_data = [data_train, data_test]

# fill Age
age_df_train = data_train[['Pclass', 'SibSp', 'Parch', 'Fare', 'Age']]
age_known_train = age_df_train[age_df_train.Age.notnull()].as_matrix()
age_df_test = data_test[['Pclass', 'SibSp', 'Parch', 'Fare', 'Age']]
age_known_test = age_df_test[age_df_test.Age.notnull()].as_matrix()
age_known = np.vstack((age_known_train, age_known_test))
rfr = RandomForestRegressor(random_state=1, n_estimators=2000, n_jobs=-1)
rfr.fit(age_known[:, :-1], age_known[:, -1])

age_unknown_train = age_df_train[age_df_train.Age.isnull()].as_matrix()
data_train.loc[data_train.Age.isnull(), 'Age'] = rfr.predict(age_unknown_train[:, :-1])
age_unknown_test = age_df_test[age_df_test.Age.isnull()].as_matrix()
data_test.loc[data_test.Age.isnull(), 'Age'] = rfr.predict(age_unknown_test[:, :-1])

# some functions
def get_title(name):
    title_search = re.search(' ([A-Za_z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ''


for dataset in full_data:
    # one-hot encoding for some features
    pclass_dummies = pd.get_dummies(dataset['Pclass'], prefix='Pclass')
    sex_dummies = pd.get_dummies(dataset['Sex'], prefix='Sex')
    embarked_dummies = pd.get_dummies(dataset['Embarked'], prefix='Embarked')
    # Family Size
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    # Is Alone
    dataset['IsAlone'] = dataset['FamilySize'].apply(lambda x: 1 if x == 1
    else 0)
    # Cabin
    dataset['Cabin_Known'] = dataset.Cabin.notnull().astype(int)
    dataset['Cabin_Unknown'] = dataset.Cabin.isnull().astype(int)
    # Age
    dataset['Age_Known'] = dataset.Age.notnull().astype(int)
    dataset['Age_Unknown'] = dataset.Age.isnull().astype(int)
    # Title
    dataset['Title'] = dataset['Name'].apply(get_title)
    dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
                              'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare', inplace=True)
    dataset['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'}, inplace=True)
    dataset['Title'].replace('', np.nan, inplace=True)
    title_dummies = pd.get_dummies(dataset['Title'], prefix='Title')
    # select features
    drop_elements = ['PassengerId', 'Name', 'Pclass', 'Sex', 'Embarked', 'Ticket', 'Cabin', 'Title']
    dataset.drop(drop_elements, axis=1, inplace=True)
    dataset = pd.concat([dataset, pclass_dummies, sex_dummies, embarked_dummies], axis=1)

data_train.to_csv('data/randomforest/data_train.csv', index=None)
data_test.to_csv('data/randomforest/data_test.csv', index=None)
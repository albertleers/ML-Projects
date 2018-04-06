import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from collections import Iterable

# learning_curve
def learning_curve_plot(model, train, label, train_sizes=[0.1, 0.25, 0.5, 0.75, 1],
                        cv=5, scoring='neg_mean_squared_error'):
    train_sizes, train_loss, test_loss = learning_curve(model, train, label,
                                                        cv=cv, scoring=scoring,
                                                        train_sizes=train_sizes)
    train_loss_mean = -np.mean(train_loss, axis=1)
    test_loss_mean = -np.mean(test_loss, axis=1)
    plt.plot(train_sizes, train_loss_mean, 'o-', color='r', label='Training')
    plt.plot(train_sizes, test_loss_mean, 'o-', color='g', label='Cross-Validation')

    plt.xlabel('Training data size')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()

# cross_validation
def cross_validation(models, train, label):
    if not isinstance(models, list):
        models = [models]
    for m in models:
        scores = cross_val_score(m, train, label,
                                 scoring='neg_mean_squared_error', cv=5)
        print(scores.mean(), scores.std()*2)
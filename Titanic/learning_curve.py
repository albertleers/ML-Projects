import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimater, X, y, title='learning curve',
                        cv=None, train_sizes=np.linspace(0.05, 1, 20),
                        verbose=0):
    train_sizes, train_scores, test_scores = learning_curve(
        estimater, X, y, cv=cv, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.xlabel('sample number')
    plt.ylabel('score')
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean-train_scores_std,
                     train_scores_mean+train_scores_std, alpha=0.1, color='b')
    plt.fill_between(train_sizes, test_scores_mean-test_scores_std,
                     test_scores_mean+test_scores_std, alpha=0.1, color='r')
    plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label='traing score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label='test score')

    plt.legend(loc='best')
    plt.show()



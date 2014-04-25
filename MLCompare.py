#-*- coding: utf-8 -*-

import scipy as sp
import numpy as np
from sklearn import datasets, cross_validation, svm, neighbors, linear_model

from multiprocessing import Process, Manager

DEBUG = True


class Learner():

    """
    A helper class, to train and compare classifiers in parallel.
    """

    def __init__(self, algorithm, dataset, targets):
        self.algorithm = algorithm[1]
        self.name = algorithm[0]
        self.dataset = dataset
        self.targets = targets
        self.train = Process(target=self.parallelTrain, args=(self.dataset,))

    def parallelTrain(self, data):
        self.algorithm.fit(self.dataset, self.targets)
        scores = cross_validation.cross_val_score(
            self.algorithm, self.dataset, self.targets, n_jobs=15)
        scores = np.mean(scores)
        if DEBUG:
            print 'Fitted %s with training scores %s' % (self.name, scores)

    def printResult(self, dataset, targets):
        score = self.algorithm.score(dataset, targets)
        print '%s:      Correct: %s     Time: 0     Parameters: ()' % (self.name, score)

# TODO:
# - By only calling MLCompare, you view of algorithm is performing better.
# - The learner should do an average of 10-15 rounds of learning/prediction.
# - Each learning phase should be done in parallel.
# - Use KFold for training and averages.
# - Try almost all parameters for each algorithm. (Check Grid-Search)
# - Finish by showing a  nice overview for each algorithm.
# - Optional: Could be nice to also have some graphs ?
#
# For each classifier, output this example line:
# SVM   Correct: 0.87  Time: 5.46s  Parameters: C=0.5, M=4.67, Kernel='rbf'


def MLCompare(dataset, targets):
    algorithms = [
        ('SVM', svm.SVC()),
        ('kNN', neighbors.KNeighborsClassifier()),
        ('Logistic Classifier', linear_model.LogisticRegression()),
    ]

    train_set = dataset[10:]
    test_set = dataset[:10]
    train_targets = targets[10:]
    test_targets = targets[:10]

    learners = [Learner(x, train_set, train_targets) for x in algorithms]

    [l.train.start() for l in learners]

    [l.train.join() for l in learners]

    [l.printResult(test_set, test_targets) for l in learners]

# For testing purposes:
data = datasets.load_iris()
MLCompare(data.data, data.target)

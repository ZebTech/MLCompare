#-*- coding: utf-8 -*-

import scipy as sp
import numpy as np
from sklearn import datasets, svm, neighbors, linear_model

from multiprocessing import Process, Manager

"""
A helper class, to train and compare classifiers in parallel.
"""


class Learner():

    def __init__(self, algorithm, dataset, targets):
        self.algorithm = algorithm[0]
        self.name = algorithm[1]
        self.dataset = dataset
        self.targets = targets
        self.process = Process(target=self.parallel_train, args=())

    def parallel_train(self, data):
        print 'Not sure if useful or not yet.'

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
        ('SVM', svm.SVC),
        ('kNN', neighbors.KNeighborsClassifier),
        ('Logistic Classifier', linear_model.LogisticRegression),
    ]

    learners = [Learner(x, dataset, targets) for x in algorithms]
    print learners

# For testing purposes:
data = datasets.load_iris()
MLCompare(data.data, data.target)

#-*- coding: utf-8 -*-

import scipy as sp
import numpy as np
from sklearn import datasets, svm, neighbors, linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold, cross_val_score

from multiprocessing import Process, Manager

# Show the debug messages ?
DEBUG = False  # True

# Computer Settings
NB_THREADS = 4

# Application settings
TEST_SET_PERCENTAGE = 0.1
GRID_SEARCH_CV = 4


class Learner():

    """
    A helper class, to train and compare classifiers in parallel threads.
    """

    def __init__(self, algorithm, features, targets):
        self.name = algorithm[0]
        self.algorithm = algorithm[1]
        self.parameters = algorithm[2]
        (self.train_data,
         self.train_targets,
         self.test_data,
         self.test_targets) = self.buildSets(features, targets)
        self.optimal_algo = self.algorithm()
        self.train = Process(
            target=self.parallelTrain, args=(self.train_data, self))

    def parallelTrain(self, data, Learner):
        # self.algorithm.fit(self.train_data, self.train_targets)
        # scores = cross_val_score(
        #     self.algorithm, self.train_data, self.train_targets, n_jobs=NB_THREADS)
        # scores = np.mean(scores)
        Learner.optimal_algo = Learner.findBestParams()
        Learner.averaged_score = Learner.findAverageCVResults()
        if DEBUG:
            message = 'Fitted %s with training scores %s'
            print message % (Learner.name, Learner.averaged_score)

    def findBestParams(self):
        clf = GridSearchCV(
            self.algorithm(), self.parameters, n_jobs=NB_THREADS, cv=GRID_SEARCH_CV)
        clf.fit(self.train_data, self.train_targets)
        if DEBUG:
            message = 'The optimal algorithm for %s is: %s'
            print message % (self.name, self.optimal_algo)
        return clf.best_estimator_

    def findAverageCVResults(self):
        print 'To implement'
        return 0.20

    def buildSets(self, features, targets):
        # Shuffling the data
        shuffled = np.asarray(features)
        shuffled = np.hstack(
            (shuffled, np.zeros((shuffled.shape[0], 1), dtype=shuffled.dtype)))
        shuffled[:, -1] = targets
        np.random.shuffle(shuffled)
        features = shuffled[:, :-1]
        targets = shuffled[:, -1]
        # Creating test and train sets
        test_length = int(TEST_SET_PERCENTAGE * len(targets))
        train_data, train_targets = features[
            0:test_length], targets[0:test_length]
        test_data, test_targets = features[
            test_length:], targets[test_length:]
        if DEBUG:
            print 'Train set size: %s examples.' % len(train_data)
            print 'Test set size: %s inputs.' % len(test_data)
        return (train_data, train_targets, test_data, test_targets)

    def score(self):
        self.optimal_algo.fit(self.train_data, self.train_targets)
        score = self.optimal_algo.score(self.test_data, self.test_targets)
        message = '%s:      Correct: %s     Time: 0     Parameters: %s'
        print ''
        print ''
        print message % (self.name, score, self.optimal_algo)
        print ''
        print ''

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


def MLCompare(features, targets):
    algorithms = [
        ('SVM', svm.SVC, [{'C': [1, 10, 100, 1000, 10000]}]),
        # ('kNN', neighbors.KNeighborsClassifier,
        #  [{'C': [1, 10, 100, 1000, 10000]}]),
        ('Logistic Classifier', linear_model.LogisticRegression,
         [{'C': [1, 10, 100, 1000, 10000]}]),
    ]

    learners = [Learner(x, features, targets) for x in algorithms]

    [l.train.start() for l in learners]

    [l.train.join() for l in learners]

    print ''
    print ''
    print 'RESULTS:'

    [l.score() for l in learners]

# For testing purposes:
data = datasets.load_iris()
MLCompare(data.data, data.target)

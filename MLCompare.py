#-*- coding: utf-8 -*-

import scipy as sp
import numpy as np
from sklearn import datasets, svm, neighbors, linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold, cross_val_score

from multiprocessing import Process, Manager

# Show the debug messages ?
DEBUG = False #True

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
        self.optimal_params = 1.0
        self.train = Process(
            target=self.parallelTrain, args=(self.train_data,))

    def parallelTrain(self, data):
        # self.algorithm.fit(self.train_data, self.train_targets)
        # scores = cross_val_score(
        #     self.algorithm, self.train_data, self.train_targets, n_jobs=NB_THREADS)
        # scores = np.mean(scores)
        self.optimal_params = self.findBestParams()
        self.averaged_score = self.findAverageCVResults()
        if DEBUG:
            message = 'Fitted %s with training scores %s'
            print message % (self.name, self.averaged_score)

    def findBestParams(self):
        clf = GridSearchCV(
            self.algorithm(), self.parameters, n_jobs=NB_THREADS, cv=GRID_SEARCH_CV)
        clf.fit(self.train_data, self.train_targets)
        print(clf.best_estimator_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() / 2, params))
        print()

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

    def scoreLearner(self):
        # self.algorithm(self.optimal_params).fit(
        #     self.train_data, self.train_targets)
        self.algorithm().fit(self.train_data, self.train_targets)
        score = 0.34  # self.algorithm.score(self.test_data, self.test_targets)
        message = '%s:      Correct: %s     Time: 0     Parameters: %s'
        print message % (self.name, score, self.optimal_params)

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

    [l.scoreLearner() for l in learners]

# For testing purposes:
data = datasets.load_iris()
MLCompare(data.data, data.target)

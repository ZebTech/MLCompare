#-*- coding: utf-8 -*-

import scipy as sp
import numpy as np
from sklearn import datasets, svm, neighbors, linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold

from multiprocessing import Process, Manager

# Show the debug messages ?
DEBUG = True

# Application settings
TEST_SET_PERCENTAGE = 0.1

class Learner():

    """
    A helper class, to train and compare classifiers in parallel threads.
    """

    def __init__(self, algorithm, features, targets):
        self.algorithm = algorithm[1]
        self.name = algorithm[0]
        self.train_data, self.train_targets,
        self.test_data, self.test_targets = self.buildSets(
            features, targets)

        self.train = Process(
            target=self.parallelTrain, args=(self.train_data,))

    def parallelTrain(self, data):
        self.algorithm.fit(self.train_data, self.train_targets)
        scores = cross_validation.cross_val_score(
            self.algorithm, self.train_data, self.train_targets, n_jobs=15)
        scores = np.mean(scores)
        if DEBUG:
            print 'Fitted %s with training scores %s' % (self.name, scores)

    def findBestParams(self):
        print 'to implement'

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
        test_length = TEST_SET_PERCENTAGE * len(targets)
        train_data, train_targets = features[
            :, test_length], targets[:, test_length]
        test_data, test_targets = features[
            :, test_length], targets[:, test_length]
        return (train_data, train_targets, test_data, test_targets)

    def printResult(self, features, targets):
        score = 0.34  # self.algorithm.score(features, targets)
        message = '%s:      Correct: %s     Time: 0     Parameters: ()'
        print message % (self.name, score)

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
        ('SVM', svm.SVC()),
        ('kNN', neighbors.KNeighborsClassifier()),
        ('Logistic Classifier', linear_model.LogisticRegression()),
    ]

    learners = [Learner(x, train_set, train_targets) for x in algorithms]

    [l.train.start() for l in learners]

    [l.train.join() for l in learners]

    [l.printResult(test_set, test_targets) for l in learners]

# For testing purposes:
data = featuress.load_iris()
MLCompare(data.data, data.target)

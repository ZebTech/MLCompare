#-*- coding: utf-8 -*-

import scipy as sp
import numpy as np

from multiprocessing import Process, Manager


class Learner():

    def __init__(self, algorithm, trainset):
        self.algorithm = algorithm
        self.trainset =
        self.process = Process(target=parallel_train, args=())

    def parallel_train(self, data):
        print 'Not sure if useful or not yet.'


linear = Learner('asdf')

# TODO:
# - By only calling MLCompare, you view of algorithm is performing better.
# - The learner should do an average of 10-15 rounds of learning/prediction.
# - Each learning phase should be done in parallel.
# - Use KFold for training and averages.
# - Try almost all parameters for each algorithm. (Check Grid-Search)
# - Finish by showing a  nice overview for each algorithm.
# - Optional: Could be nice to also have some graphs ?


def MLCompare(dataset, targets):
    print 'To be executed to compare the ML Algorithms on a given dataset'

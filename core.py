#!/usr/bin/env python

from datetime import datetime
import csv


TIMESTAMP = datetime.today().strftime('%m%d%y%H%M%S')
DATADIR = 'bytecup2016data/'


class Solver(object):
    def __init__(self, output=None):
        self.output = output if output else '%s_%s.csv' % (self.__class__.__name__, TIMESTAMP)

    def solve(self):
        """
        Create an output file with the predictions for the question, user pairs in test or validation file.
        """
        pass

    def predict(self, qid, uid):
        """
        Predict the probability user with ID uid will answer question with ID quid.

        param {string} qid - Question id
        param {string} uid - User id

        return {number} Probability of answering the question
        """
        pass

    def file_iter(self, filename, delimiter='\t'):
        """
        A generator over the rows in the file named filename.

        This method allows you to easily iterate over the rows of a file without having to worry about using the `open`
        or `csv.reader` functions.

        Example:
            for row in self.file_iter('myfile.csv', delimiter=','):
                ...

        param {string} filename - Name of  file to read. Files should be relative to path that script is run.
        param {string} delimiter (Optional) - Delimiter of elements in file. Most files for ByteCup are tab delimited
            (\t), so \t is the default delimiter.

        return {iterator} Iterator over file
        """
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                yield(row)

    def feature_vector(self, qid, uid, binary=False, n_features=None):
        """
        Generate a feature vector for the question, user pair.

        param {string} qid - Question id
        param {string} uid - User id
        param {bool} binary (Optional) - If true, only binary (0,1) features are returned in the feature vector.
            Defaults to False.
        param {number} n_features (Optional) - Number of features to return in feature vector. If None, all features
            are returned. Defaults to None

        return {numpy vector} Feature vector
        """
        # TODO: Implement
        pass
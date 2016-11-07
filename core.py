#!/usr/bin/env python

from datetime import datetime
import csv
from itertools import izip, combinations_with_replacement
import numpy as np


TIMESTAMP = datetime.today().strftime('%m%d%y%H%M%S')
DATADIR = 'bytecup2016data/'


class Solver(object):
    def __init__(self, output=None):
        self.output = output if output else '%s_%s.csv' % (self.__class__.__name__, TIMESTAMP)

        # Load all user and question data
        self.users = self.load_users()
        self.questions = self.load_questions()

        # machine learning solver (typically from skit-learn
        self.solver = None

        # Initialize tag matrix
        max_qtags = 19
        max_utags = 142
        self.tag_matrix = np.zeros([max_qtags + 1, max_utags + 1], dtype=int)


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

    def _id_sequence_to_list(self, sequence):
        """
        Convert a string character or word ID sequence into a list of floats.

        This is a convenience method to convert a sequence like '0/1/2/3/10' into [0.0, 1.0, 2.0, 3.0, 10.0]. The word
        and character ID sequences in the user and question input data appear in this '/'-separated string format.

        Note: The order of the numbers in the input is preserved in the output (i.e. '0/1' won't return [1.0, 0.0]).
        Duplicates are also preserved.

        param {string}: word or character ID sequence string

        return {float list} input sequence converted into a list of floats
        """
        # Need try/except for case where word or character sequence is null (i.e. '/')
        try:
            return map(float, sequence.split('/'))
        except ValueError:  # sequence == '/'
            return []

    def _question_to_dict(self, question):
        """
        Convert the list form of a question into dictionary form.

        Input is a row of the question_info.txt file when reading it using the csv.reader method. Python's csv.reader
        converts each entry in the question_info.txt into a list of strings. This method will convert that list into a
        dictionary that's easier to manipulate/handle later on.

        Input:
            [ID, tag, word ID seq, char ID seq, upvotes, answers, top answers]  # list of strings

        Output:
            {
                ID: {
                        tag: {float} question tag
                        words: {float list} list of word ids
                        characters: {float list} list of character ids
                        upvotes: {float} number of upvotes
                        answers: {float} number of answers
                        top_answers: {float} number of top quality answers
                    }
            }
        """
        return {
            question[0]: {
                'tag': float(question[1]),
                'words': self._id_sequence_to_list(question[2]),
                'characters': self._id_sequence_to_list(question[3]),
                'upvotes': float(question[4]),
                'answers': float(question[5]),
                'top_answers': float(question[6])
            }
        }

    def _user_to_dict(self, user):
        """
        Convert the list form of a user into dictionary form.

        Input is a row of the user_info.txt file when reading it using the csv.reader method. Python's csv.reader
        converts each entry in user_info.txt into a list of strings. This method will convert that list into a
        dictionary that's easier to manipulate/handle later on.

        Input:
            [ID, tag sequence, word ID seq, char ID seq]  # list of strings

        Output:
            {
                ID: {
                        tags: {float list} user tags
                        words: {float list} list of word ids
                        characters: {float list} list of character ids
                    }
            }
        """
        return {
            user[0]: {
                'tags': self._id_sequence_to_list(user[1]),
                'words': self._id_sequence_to_list(user[2]),
                'characters': self._id_sequence_to_list(user[3]),
            }
        }

    def load_questions(self):
        """
        Load all 8095 questions into memory as a nested dictionary.

        Read the Bytecup file 'question_info.txt' line by line, constructing a nested dictionary
        for each entry. Each question will have the following entry in the dictionary:

            {
                id: {
                        tag: {float} question tag
                        words: {float list} list of word ids
                        characters: {float list} list of character ids
                        upvotes: {float} number of upvotes
                        answers: {float} number of answers
                        top_answers: {float} number of top quality answers
                    }
            }

        The id refers to the unique question identifier.

        return {dict} The dictionary of all 8095 questions
        """
        questions = {}
        for row in self.file_iter(DATADIR + 'question_info.txt'):
            questions.update(self._question_to_dict(row))
        return questions

    def load_users(self):
        """
        Load all 28763 users into memory as a nested dictionary.

        Read the Bytecup file 'user_info.txt' line by line, constructing a nested dictionary for
        each entry. Each user will have the following entry in the dictionary:

            {
                id: {
                        tags: {float list} user tags
                        words: {float list} list of word ids
                        characters: {float list} list of character ids
                    }
            }

        The id refers to the unique user identifier.

        return {dict} The dictionary of all 28763 users
        """
        users = {}
        for row in self.file_iter(DATADIR + 'user_info.txt'):
            users.update(self._user_to_dict(row))
        return users

    def _update_tag_matrix(self, qtag, utags, label):
        """
        Update tag matrix with passed in question and user tags.

        Tag matrix shows tag matches between question and user in answered questions (label = 1)
        Question tags along rows, user tags along columns
        """
        if not label is 1:
            return

        for t in utags:
            self.tag_matrix[int(qtag), int(t)] += 1

    def _popularity_features(self, question, user):
        """
        Return the features that correspond to the popularity of a question.

        The features returned are:
            - number of upvotes
            - total number of answers
            - total number of top quality answers
        
        return {list} list of integers corresponding to popularity features
        """
        return [
            question['upvotes'],
            question['answers'],
            question['top_answers']
        ]

    def _number_common_words(self, question, user):
        common_words = set(question['words']) & set(user['words'])
        return len(common_words)

    def _number_common_characters(self, question, user):
        common_chars = set(question['characters']) & set(user['characters'])
        return len(common_chars)

    def _has_n_common_words(self, question, user, nmax=5):
        """
        Return number of words in common between question and user description as binary feature

        return {list} of 0s with 1 corresponding to number of words in common
        """
        num_common_words = self._number_common_words(question, user)
        has_n_words = [0 for i in range(nmax+1)]
        if num_common_words <= nmax:
            has_n_words[num_common_words] = 1
        return has_n_words

    def _has_n_common_characters(self, question, user, nmax=9):
        """
        Return number of characters in common between question and user description as binary feature

        return {list} of 0s with 1 corresponding to number of characters in common
        """
        num_common_chars = self._number_common_characters(question, user)
        has_n_chars = [0 for i in range(nmax+1)]
        if num_common_chars <= nmax:
            has_n_chars[num_common_chars] = 1
        return has_n_chars

    def _has_n_most_popular_tags(self, question, user, tagmat, num_tags=5):
        """
        Return whether user has tags that most commonly answered question tag as binary feature

        return {list} of 0s with 1 corresponding to popularity index of tags that users have
        """
        qtag = int(question['tag'])
        num_answers = tagmat[qtag]
        sorted_utags = np.argsort(num_answers) # sorted list from least common to most common

        has_tags = [0 for i in range(num_tags)]
        for i in range(num_tags):
            if sorted_utags[-1 - i] in user['tags']:
                has_tags[i] = 1

        return has_tags


    def _percent_top_answers(self, question, user):
        """
        Return percentage of answers which are ranked as top answers
        If the question has no answers, return 0 (because none are top answers)
        """
        if question['answers'] > 0:
            return question['top_answers']/question['answers']
        else:
            return 0

    def feature_vector(self, question, user, binary=False, poly_expansion=True):
        """
        Generate a feature vector for the question, user pair.

        param {dict} question - Nested dictionary representation of question
        param {dict} user - Nested dictionary representation of user
        param {bool} binary (Optional) - If true, only binary (0,1) features are returned in the feature vector.
            Defaults to False.
        param {bool} poly_expansion (optional) = If true, perform polynomial expansion on the feature vector.
            Defaults to False

        return {numpy vector} Feature vector
        """
        features = []

        # Process non-binary (0,1) features
        if not binary:
            features.append(question['tag'])
            features += self._popularity_features(question, user)
            features.append(self._number_common_words(question, user))
            features.append(self._number_common_characters(question, user))
            features.append(self._percent_top_answers(question, user))

        # binary features
        features += (self._has_n_common_words(question, user))
        features += (self._has_n_common_characters(question, user))
        features += (self._has_n_most_popular_tags(question, user, self.tag_matrix))

        if poly_expansion:
            features = polynomial_expansion(features)

        return np.array(features)

    def load_training(self, filename=DATADIR+'invited_info_train.txt'):
        """
        Load all 245752 rows of training data into memory as numpy arrays.

        Each row of invited_info_train.txt is converted into a feature vector and added to the feature matrix X. The
        label vector y is a binary label indicating if the user answered the question.

        param {str} filename (Optional) - Relative path and filename of training data file. By default, the file
            bytecup2016data/invited_info_train.txt is read. However, you can pass your own path if it differs.

        return {tuple - numpy array, numpy array} - The first element of the tuple is the feature matrix X as a numpy
            array. The second element of the return tuple is the label vector y as a numpy array.
        """
        x = []
        y = []

        # Iterate over input file once to compute anything that requires all data to make feture vectors
        # E.g. most popular tag features
        for row in self.file_iter(filename):
            qtag = self.questions[row[0]]['tag']
            utags = self.users[row[1]]['tags']
            label = int(row[2])
            self._update_tag_matrix(qtag, utags, label)

        # Iterate over row of input file
        for row in self.file_iter(filename):
            question = self.questions[row[0]]
            user = self.users[row[1]]
            # Convert question,user pair into feature vector
            x.append(self.feature_vector(question, user))
            y.append(int(row[2]))
        return np.array(x), np.array(y)

    def validation_iter(self, filename=DATADIR+'validate_nolabel.txt'):
        for row in self.file_iter(filename, delimiter=','):
            # Skip header
            if 'qid' in row:
                continue
            yield(row)

    def load_validation(self, filename=DATADIR+'validate_nolabel.txt'):
        x = []
        for row in self.validation_iter(filename):
            question = self.questions[row[0]]
            user = self.users[row[1]]
            x.append(self.feature_vector(question, user))

            # Move this
        return np.array(x)

    def train(self):
        x, y = self.load_training()
        self.solver.fit(x,y)

    def solve(self, filename=DATADIR+'validate_nolabel.txt'):
        """
        Create an output file with the predictions for the question, user pairs in test or validation file.
        """
        with open(self.output, 'w') as outfile:
            outfile.write('qid,uid,label\n')
            for row in self.validation_iter():
                prediction = self.predict(qid=row[0], uid=row[1])
                outfile.write('%s,%s,%s\n' % (row[0], row[1], prediction))

    def predict(self, qid, uid):
        """
        Predict the probability user with ID uid will answer question with ID quid.

        param {string} qid - Question id
        param {string} uid - User id

        return {number} Probability of answering the question
        """
        question = self.questions[qid]
        user = self.users[uid]
        x = self.feature_vector(question, user).reshape(1, -1)  # reshape needed to avoid sklearn warning
        return self.solver.predict(x)[0]


def polynomial_expansion(vect):
    """
    Expand the list vect with each element raised to each power in powers

    param {numerical list} vect - list of numbers
    param {numerical list} powers - list of exponents

    return {list}
    """
    for xi, xj in combinations_with_replacement(vect, 2):
        vect.append(xi * xj)
    return vect

import csv
from collections import defaultdict
import numpy as np

from core import Solver, DATADIR

class DescrChars(Solver):
    # index to data about number of words in common
    ANSWERED = 0
    TOTAL = 1

    def __init__(self):
        super(DescrChars, self).__init__()
        self.probability = None
        self.questions = {}
        self.users = {}
        self.load_questions()
        self.load_users()

    def load_questions(self):
        for row in self.file_iter(DATADIR + 'question_info.txt'):
            self.questions[row[0]] = row[3].split('/')

    def load_users(self):
        for row in self.file_iter(DATADIR + 'user_info.txt'):
            self.users[row[0]] = row[3].split('/')

    def chars_in_common(self, qid, uid):
        common = set(self.questions[qid]) & set(self.users[uid])
        return len(common)

    def train(self):
        max_chars = 10
        chars = np.zeros((max_chars+1,2))
        for row in self.file_iter(DATADIR + 'invited_info_train.txt'):
            qid = row[0]
            uid = row[1]
            answered = row[2]
            # Number of characters in common
            numchars = self.chars_in_common(qid, uid)
            chars[numchars, self.TOTAL] += 1
            if answered is '1':
                chars[numchars, self.ANSWERED] += 1

        self.probability = np.zeros(max_chars+1)
        for i in range(0, max_chars+1):
            if chars[i,self.TOTAL] == 0:
                self.probability[i] = 0
            else:
                self.probability[i] = chars[i,self.ANSWERED] / chars[i,self.TOTAL]

    def predict(self, qid, uid):
        return self.probability[self.chars_in_common(qid, uid)]

    def solve(self):
        with open(self.output, 'w') as outfile:
            outfile.write('qid,uid,label\n')
            for row in self.file_iter(DATADIR + 'validate_nolabel.txt', delimiter=','):
                # Skip header row in validate_nolabel.txt file
                if 'qid' in row:
                    continue
                qid = row[0]
                uid = row[1]
                answered = self.predict(qid, uid)
                outfile.write('%s,%s,%s\n' % (qid, uid, answered))


if __name__ == '__main__':
    solver = DescrChars()
    solver.train()
    solver.solve()
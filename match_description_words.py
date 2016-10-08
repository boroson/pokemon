import csv
from collections import defaultdict
import numpy as np

from core import Solver, DATADIR

class DescrWords(Solver):
    # index to data about number of words in common
    ANSWERED = 0
    TOTAL = 1

    def __init__(self):
        super(DescrWords, self).__init__()
        self.probability = None
        self.questions = {}
        self.users = {}
        self.load_questions()
        self.load_users()

    def load_questions(self):
        for row in self.file_iter(DATADIR + 'question_info.txt'):
            self.questions[row[0]] = row[2].split('/')

    def load_users(self):
        for row in self.file_iter(DATADIR + 'user_info.txt'):
            self.users[row[0]] = row[2].split('/')

    def words_in_common(self, qid, uid):
        common = set(self.questions[qid]) & set(self.users[uid])
        return len(common)

    def train(self):
        max_words = 10
        words = np.zeros((max_words+1,2))
        for row in self.file_iter(DATADIR + 'invited_info_train.txt'):
            qid = row[0]
            uid = row[1]
            answered = row[2]
            # Number of words in common
            numwords = self.words_in_common(qid, uid)
            print numwords
            words[numwords, self.TOTAL] += 1
            if answered is '1':
                words[numwords, self.ANSWERED] += 1

        self.probability = np.zeros(max_words+1)
        for i in range(0, max_words+1):
            if words[i,self.TOTAL] == 0:
                self.probability[i] = 0
            else:
                self.probability[i] = words[i,self.ANSWERED] / words[i,self.TOTAL]

    def predict(self, qid, uid):
        return self.probability[self.words_in_common(qid, uid)]

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
    solver = DescrWords()
    solver.train()
    solver.solve()
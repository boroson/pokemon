import csv
from collections import defaultdict

from core import Solver, DATADIR


class NaiveTag(Solver):
    def __init__(self):
        super(NaiveTag, self).__init__()
        self.probability = None
        self.questions = {}
        self.users = {}
        self.load_questions()
        self.load_users()

    def load_questions(self):
        for row in self.file_iter(DATADIR + 'question_info.txt'):
            self.questions[row[0]] = row[1]

    def load_users(self):
        for row in self.file_iter(DATADIR + 'user_info.txt'):
            self.users[row[0]] = row[1].split('/')

    def tag_in_common(self, qid, uid):
        try:
            if self.questions[qid] in self.users[uid]:
                return True
        except KeyError:
            pass
        finally:
            return False

    def train(self):
        common_tag = {'total': 0, 'answered': 0}
        no_common_tag = {'total': 0, 'answered': 0}
        for row in self.file_iter(DATADIR + 'invited_info_train.txt'):
            qid = row[0]
            uid = row[1]
            answered = row[2]
            # Common Tag b/w user and question
            if self.tag_in_common(qid, uid):
                common_tag['total'] += 1
                if answered is '1':
                    common_tag['answered'] += 1
            # No common tag b/w user and question
            else:
                no_common_tag['total'] += 1
                if answered is '1':
                    no_common_tag['answered'] += 1

        self.probability = {
            'common_tag': common_tag['answered'] / float(common_tag['total']) if common_tag['total'] else 0,
            'no_common_tag': no_common_tag['answered'] / float(no_common_tag['total']) if no_common_tag['total'] else 0
        }

    def predict(self, qid, uid):
        if self.tag_in_common(qid, uid):
            return self.probability['common_tag']
        return self.probability['no_common_tag']

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
    solver = NaiveTag()
    solver.train()
    solver.solve()
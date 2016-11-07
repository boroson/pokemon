import numpy as np

from core import Solver, DATADIR

MAX_WORD = 37810
MAX_CHAR = 4022

def vector_cos_similarity_words(user1, user2):
    user1_words = np.zeros(MAX_WORD)
    user2_words = np.zeros(MAX_WORD)
    for word in user1['words']:
        user1_words[word] += 1
    for word in user2['words']:
        user2_words[word] += 1

    w = np.dot(user1_words, user2_words)/(np.linalg.norm(user1_words)*np.linalg.norm(user2_words))
    return w


def vector_cos_similarity_chars(user1, user2):
    user1_chars = np.zeros(MAX_CHAR)
    user2_chars = np.zeros(MAX_CHAR)
    for character in user1['characters']:
        user1_chars[character] += 1
    for character in user2['characters']:
        user2_chars[character] += 1

    w = np.dot(user1_chars, user2_chars) / (np.linalg.norm(user1_chars) * np.linalg.norm(user2_chars))
    return w



class Similarity(Solver):
    # index to data about number of words in common
    ANSWERED = 0
    TOTAL = 1

    def __init__(self):
        super(Similarity, self).__init__()
        self.qu_pairs = {}

    def load_training(self, filename=DATADIR+'invited_info_train.txt'):
        # pairs of questions and all users that looked at them
        qu_pairs = {}
        for row in self.file_iter(filename):
            question = row[0]
            user = row[1]
            answer = float(row[2])
            if question in qu_pairs:
                qu_pairs[question]['users'].append(user)
                qu_pairs[question]['answers'].append(answer)
            else:
                qu_pairs.update({question: {
                    'users': [user],
                    'answers': [answer]
                 }
                })

        return qu_pairs

    def train(self):
        self.qu_pairs = self.load_training()

    def predict(self, qid, uid):
        # For all users who looked at question, compute similarity weights
        if qid in self.qu_pairs:
            total_weight = 0
            total_ans_weight = 0
            users = self.qu_pairs[qid]['users']
            answers = self.qu_pairs[qid]['answers']
            for i in range(0, len(users)):
                sim = vector_cos_similarity_chars(self.users[users[i]], self.users[uid])
                if answers[i]:
                    total_ans_weight += sim
                total_weight += sim

            if total_weight > 0:
                return total_ans_weight/total_weight
            else:
                return 0.0
        else:
            return 0.0


if __name__ == '__main__':
    solver = Similarity()
    solver.train()
    solver.solve()
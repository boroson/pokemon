#!/usr/bin/env python

from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import argparse
import numpy as np
from multiprocessing import Process

from collaborative_filter import CollabFilter, N_USERS
from kmeans_cf import KMeansCF
from core import file_iter, DATADIR

MAX_WORD = 37810
N_CORES = 16

class HybridCF(KMeansCF):
    def __init__(self, k=8, data=None):
        super(HybridCF, self).__init__(data=data)
        self.corrs = {}

    def populate_rating_matrix(self, data):
        """
        Use the training data to populate the rating matrix
        """
        if data is None:
            data = self.file_iter(DATADIR + 'invited_info_train.txt')

        # Load all user and question data
        self.users = self.load_users()
        self.questions = self.load_questions()

        self.num_answers = {}

        for qid, uid, label in data:
            uindex = self.u_index[uid]
            qindex = self.q_index[qid]
            # Scale to [-1,1]
            if int(label) == 0:
                self.R_mat[uindex, qindex] = -1.0
            else:
                self.R_mat[uindex, qindex] = 1.0
            self.total_ratings += 1
            if uid in self.num_answers:
                self.num_answers[uid] += 1
            else:
                self.num_answers[uid] = 1

        # test matrix dump
        # self.R_mat.dump("project_data/r_mat_test.dat")

        # Split up R_mat into pieces
        # Run separate processes for each piece
        procs = []
        q_pieces = []
        q_size = len(self.q_ids)/N_CORES + 1
        for i in range(0,N_CORES):
        #     R_mat_piece = np.copy(self.R_mat)
            q_pieces.append(self.q_ids[q_size*i: q_size*(i+1)])
        #     print("Starting thread {}".format(i))
        #     procs.append(Process(target=self.proc_rmat_piece, args=(R_mat_piece, q_pieces[i], i)))
        #     procs[i].start()
        #
        for i in range(0,N_CORES):
        #     procs[i].join()
            print("Thread {} finished".format(i))
            R_mat_piece = np.load("project_data/r_mat_svm_{}.dat".format(i))
            for qid in q_pieces[i]:
                print self.q_index[qid], np.nonzero(np.isnan(R_mat_piece[:, self.q_index[qid]]))
                for uid in self.u_ids:
                    self.R_mat[self.u_index[uid],self.q_index[qid]] = R_mat_piece[self.u_index[uid],self.q_index[qid]]


        # print("Computing pseudo-ratings")
        # # compute content-boosted pseudo-ratings
        # for qid in self.q_index.keys():
        #     nb, numus = self.train_question_predictor(qid)
        #     # We can't deal with users who answered no questions.
        #     # We should be using predictions from user descriptions too.
        #     if numus > 0:
        #         for uid in self.u_index.keys():
        #             if np.isnan(self.R_mat[self.u_index[uid], self.q_index[qid]]):
        #                 # question = self.questions[qid]
        #                 # # Convert question into feature vector
        #                 # question_words = np.zeros(MAX_WORD)
        #                 # for word in question['words']:
        #                 #     question_words[word] += 1
        #                 # x = question_words.reshape(1, -1)  # reshape needed to avoid sklearn warning
        #                 user = self.users[uid]
        #                 # Convert question into feature vector
        #                 desc_words = np.zeros(MAX_WORD)
        #                 for word in user['words']:
        #                     desc_words[word] += 1
        #                 x = desc_words.reshape(1, -1)  # reshape needed to avoid sklearn warning
        #                 p_rating = nb.predict(x)
        #                 if int(p_rating) == 0:
        #                     self.R_mat[self.u_index[uid], self.q_index[qid]] = -1.0
        #                 else:
        #                     self.R_mat[self.u_index[uid], self.q_index[qid]] = 1.0
        #     if (self.q_index[qid] % 10) == 0:
        #         print self.q_index[qid]
        # print("Done")

    def train_question_predictor(self, qid):
        # nb = MultinomialNB()
        mysvm = svm.SVR(kernel='rbf')
        x, y, numus = self.load_question_training(qid)
        if numus > 0:
            mysvm.fit(x, y)
            # nb.fit(x, y)
        # return nb, numus
        return mysvm, numus

    def load_user_training(self, uid, filename=DATADIR + 'invited_info_train.txt'):
        """
        Training data for user is converted to feature vector using bag-of-words of questions

        param {str} filename (Optional) - Relative path and filename of training data file. By default, the file
            bytecup2016data/invited_info_train.txt is read. However, you can pass your own path if it differs.

        return {tuple - numpy array, numpy array} - The first element of the tuple is the feature matrix X as a numpy
            array. The second element of the return tuple is the label vector y as a numpy array.
        """
        x = []
        y = []

        # Iterate over row of input file
        for row in self.file_iter(filename):
            if row[1] == uid:
                question = self.questions[row[0]]
                # Convert question,user pair into feature vector
                question_words = np.zeros(MAX_WORD)
                for word in question['words']:
                    question_words[word] += 1
                x.append(question_words)
                y.append(int(row[2]))
        return np.array(x), np.array(y), len(y)

    def load_question_training(self, qid, filename=DATADIR + 'invited_info_train.txt'):
        """
        Training data for user is converted to feature vector using bag-of-words of questions

        param {str} filename (Optional) - Relative path and filename of training data file. By default, the file
            bytecup2016data/invited_info_train.txt is read. However, you can pass your own path if it differs.

        return {tuple - numpy array, numpy array} - The first element of the tuple is the feature matrix X as a numpy
            array. The second element of the return tuple is the label vector y as a numpy array.
        """
        x = []
        y = []

        question = self.questions[qid]

        # Iterate over row of input file
        for row in self.file_iter(filename):
            if row[0] == qid:
                user = self.users[row[1]]
                # Convert question,user pair into feature vector
                # desc_words = np.zeros(MAX_WORD)
                # for word in user['words']:
                #     desc_words[word] += 1
                x.append(self.feature_vector(question=question, user=user, binary=False, poly_expansion=False))
                y.append(int(row[2]))
        return np.array(x), np.array(y), len(y)

    def solve(self, filename=DATADIR+'validate_nolabel.txt'):
        """
        Create an output file with the predictions for the question, user pairs in test or validation file.
        """
        qids = []
        uids = []
        for row in self.validation_iter():
            qids.append(row[0])
            uids.append(row[1])

        procs = []
        q_size = len(qids) / N_CORES + 1
        for i in range(0, N_CORES):
            print("Starting thread {}".format(i))
            procs.append(Process(target=self.solve_piece, args=(qids[q_size * i: q_size * (i + 1)], uids[q_size * i: q_size * (i + 1)], i)))
            procs[i].start()

        results = {}
        for i in range(0, N_CORES):
            procs[i].join()
            print("Thread {} finished".format(i))
            for row in self.file_iter("project_data/predictions_svm_weight_{}.csv".format(i), delimiter=','):
                key = (row[0], row[1])
                results[key] = row[2]

        with open(self.output, 'w') as outfile:
        # with open('invited_pseudoratings.csv', 'w') as outfile:
            outfile.write('qid,uid,label\n')
            for row in self.validation_iter():
                key = (row[0], row[1])
                if key in results:
                # prediction = self.R_mat[self.u_index[row[1]],self.q_index[row[0]]]
                    prediction = results[key]
                    outfile.write('%s,%s,%s\n' % (row[0], row[1], prediction))
                else:
                    outfile.write('\n')

    def predict(self, uid, qid):
        """
        Returns the predicted rating for user UID and question QID
        """
        ui = self.u_index[uid]
        qi = self.q_index[qid]

        prediction = self.umeans[ui]
        if uid in self.num_answers:
            mi = min(float(self.num_answers[uid])/20.0, 1.0)
            print uid, mi

            sim_total = 0.0
            neighbor_total = 0.0
            for uj in self.neighborhood(ui, qi):
                uid_j = self.u_ids[uj]
                if uid_j in self.num_answers:
                    mj = min(float(self.num_answers[uid_j]) / 20.0, 1.0)
                    if ui < uj:
                        key = (ui, uj) #.sort()
                    else:
                        key = (uj, ui)
                    # print("ui = {}, uj = {}, key = {}".format(ui, uj, key))
                    if key in self.corrs:
                        sim = self.corrs[key]
                        # print("Found key {}".format(key))
                    else:
                        # sim = self.similarity(ui, uj)
                        sim = self.pearson_corr(ui, uj)
                        self.corrs[key] = sim
                    if np.isnan(sim):
                        print key

                    neighbor_total += sim * (self.R_mat[uj, qi] - self.umeans[uj]) * (2*mi*mj/(mi + mj))
                    sim_total += abs(sim) * (2*mi*mj/(mi + mj))

            # Add content prediction
            # currently no weighting for number of questions answered
            neighbor_total += mi * (self.R_mat[ui, qi] - self.umeans[ui])
            sim_total += mi
            if sim_total > 0.0:
                prediction += (neighbor_total / float(sim_total))
        # Scale back to [0,1]
        prediction = (prediction + 1) / 2
        if prediction > 1.0:
            print("Prediction above 1.0")
            prediction = 1.0
        elif prediction < 0.0:
            print("Prediction below 0.0")
            prediction = 0.0
        elif np.isnan(prediction):
            print("Prediction is NaN")
            prediction = 0.0
        return prediction

    def proc_rmat_piece(self, R_mat_piece, q_index_piece, i):
        for qid in q_index_piece:
            nb, numus = self.train_question_predictor(qid)
            # We can't deal with users who answered no questions.
            # We should be using predictions from user descriptions too.
            if numus > 0:
                for uid in self.u_index.keys():
                    if np.isnan(R_mat_piece[self.u_index[uid], self.q_index[qid]]):
                        question = self.questions[qid]
                        # # Convert question into feature vector
                        # question_words = np.zeros(MAX_WORD)
                        # for word in question['words']:
                        #     question_words[word] += 1
                        # x = question_words.reshape(1, -1)  # reshape needed to avoid sklearn warning
                        user = self.users[uid]
                        # # Convert question into feature vector
                        # desc_words = np.zeros(MAX_WORD)
                        # for word in user['words']:
                        #     desc_words[word] += 1
                        # x = desc_words.reshape(1, -1)  # reshape needed to avoid sklearn warning
                        x = self.feature_vector(question=question, user=user, binary=False, poly_expansion=False).reshape(1, -1)  # reshape needed to avoid sklearn warning
                        p_rating = nb.predict(x)
                        # print self.u_index[uid], self.q_index[qid], p_rating
                        if int(p_rating) == 0:
                            R_mat_piece[self.u_index[uid], self.q_index[qid]] = -1.0
                        else:
                            R_mat_piece[self.u_index[uid], self.q_index[qid]] = 1.0
            if (self.q_index[qid] % 10) == 0:
                print self.q_index[qid]
            print self.q_index[qid], np.nonzero(np.isnan(R_mat_piece[:, self.q_index[qid]]))

        R_mat_piece.dump("project_data/r_mat_svm_{}.dat".format(i))


    def solve_piece(self, qids, uids, i):
        with open("project_data/predictions_svm_weight_{}.csv".format(i), 'w') as outfile:
            # corrs = {}
            for j in range(0, len(qids)):
                prediction = self.predict(qid=qids[j], uid=uids[j])
                outfile.write('%s,%s,%s\n' % (qids[j], uids[j], prediction))
                if (j % 10) == 0:
                    print j

# def cross_validation():
#     # Load training data
#     X = np.array([row for row in file_iter(DATADIR + 'invited_info_train.txt')])
#     # Test k values from 2 to 11
#     k_values = range(2, 11)
#     # Create K-fold cross validation iterator
#     folds = 15
#     kf = KFold(n_splits=folds)
#
#     score = {}
#     for k in k_values:
#         print("\nTesting k = %d..." % k)
#         print("-------------------")
#         score[k] = 0
#         for train_index, test_index in kf.split(X):
#             # Split data into training and test set
#             training, test = X[train_index], X[test_index]
#             classifier = KMeansCF(k=k, data=training)
#             score[k] += classifier.score(test)
#         score[k] /= float(folds)
#     for k, mae in score.iteritems():
#         print("K = %d | MAE = %f" % (k, mae))
#     best_k = min(score, key=lambda x: score[x])
#     print("\nBest K = %d" % best_k)
#     return best_k

#
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-k', help='Number of clusters (default: 8)', type=int, default=8)
#     parser.add_argument('-cv', help='Perform cross validation', action='store_true')
#     return parser.parse_args()

if __name__ == '__main__':
    # args = parse_args()
    # if args.cv:
    #     cross_validation()
    # else:
    hybrid = HybridCF()
    print("Solving...")
    hybrid.solve()
    print("\tDone")

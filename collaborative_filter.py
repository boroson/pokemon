from __future__ import print_function

from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr
import sys

from core import Solver, DATADIR, TIMESTAMP, all_nan


N_QUESTIONS = 8095
N_USERS = 28763


class CollabFilter(Solver):
    """
    User-model based Collaborative filter
    """
    def __init__(self, data=None):
        self.output = '%s_%s.csv' % (self.__class__.__name__, TIMESTAMP)
        # Dictionary assigning each user an index and each question an index - this is mainly for convenience
        self.q_index = self.create_index_map(filename=DATADIR + 'question_info.txt')
        self.u_index = self.create_index_map(filename=DATADIR + 'user_info.txt')

        # Initialize rating matrix (rows = user indices, columns are question indices)
        self.R_mat = np.full(shape=(N_USERS, N_QUESTIONS), fill_value=np.NaN)
        self.total_ratings = 0
        self.populate_rating_matrix(data)

        # Precompute user rating means
        print("Precomputing user rating means...")
        self.umeans = self.compute_means()
        print("\tDone")

    def create_index_map(self, filename):
        """
        Create a dictionary mapping question or user ID's to integer indexes

        Ex:
            {
                'asd24kfe0204...': 0,
                ...
            }
        """
        indices = {}
        index = 0
        for row in self.file_iter(filename):
            uuid = row[0]
            indices[uuid] = index
            index += 1
        return indices

    def populate_rating_matrix(self, data):
        """
        Use the training data to populate the rating matrix
        """
        if data is None:
            data = self.file_iter(DATADIR + 'invited_info_train.txt')

        for qid, uid, label in data:
            uindex = self.u_index[uid]
            qindex = self.q_index[qid]
            # Scale to [-1,1]
            if int(label) == 0:
                self.R_mat[uindex, qindex] = -1.0
            else:
                self.R_mat[uindex, qindex] = 1.0
            self.total_ratings += 1

    def compute_means(self):
        """
        Iterate over elements in it, and compute the means of there row/column in the ratings matrix
        """
        means = []
        for i in xrange(N_USERS):
            ratings = self.R_mat[i]
            avg = np.nanmean(ratings) if not all_nan(ratings) else 0.0
            means.append(avg)
        return means

    def compute_similarities(self):
        """
        Create a NxN matrix of all pairwise similarities
        """
        S_mat = np.zeros(shape=(N_USERS, N_USERS))
        for i in xrange(N_USERS):
            print(i)
            for j in xrange(N_USERS):
                S_mat[i, j] = self.similarity(i, j)
        return S_mat
                
    def rating(self, uid, qid):
        """
        Return the rating user UID gave to question QID. If no rating given, np.nan returned
        """
        ui = self.u_index[uid]
        qi = self.q_index[qid]
        return self.R_mat[ui, qi]

    def user_ratings(self, uid):
        return self.R_mat[self.u_index[uid]]

    def question_ratings(self, qid):
        return self.R_mat[:, self.q_index[qid]]

    def similarity(self, i, j):
        """
        Computes the similarity between uid1 and uid2
        """
        ri = np.nan_to_num(self.R_mat[i])
        normi = np.linalg.norm(ri)

        rj = np.nan_to_num(self.R_mat[j])
        normj = np.linalg.norm(rj)

        if normi == 0.0 or normj == 0.0:
            return 0.0
        return np.dot(ri, rj) / (normi * normj)

    def pearson_corr(self, i, j):
        """
        Computes the Pearson correlation between uid1 and uid2
        """
        # Find elements rated by both users
        qi_ind = np.nonzero(np.isfinite(self.R_mat[i]))
        qj_ind = np.nonzero(np.isfinite(self.R_mat[j]))
        common_qs = np.intersect1d(qi_ind, qj_ind)
        if common_qs.size == 0:
            # users are not correlated
            return 0.0

        ri = self.R_mat[i, common_qs]
        rj = self.R_mat[j, common_qs]

        # normalize ratings
        # normalized_ri = ri - np.mean(ri)
        # normalized_rj = rj - np.mean(rj)
        normalized_ri = ri - self.umeans[i]
        normalized_rj = rj - self.umeans[j]

        normi = np.linalg.norm(normalized_ri)
        normj = np.linalg.norm(normalized_rj)

        if normi == 0.0 or normj == 0.0:
            return 0.0
        return np.dot(normalized_ri, normalized_rj) / (normi * normj)

    def neighborhood(self, ui, qi):
        """
        Returns the user neighborhood of a user (index ui) and question (index qi)
        """
        neighbors = []
        for uj in xrange(N_USERS):
            if ui == uj or np.isnan(self.R_mat[uj, qi]):
                continue
            neighbors.append(uj)
        sorted_neighbors = sorted(neighbors, key=lambda uj: self.similarity(ui, uj), reverse=True)  # sort descending order by similarity
        return sorted_neighbors

    def predict(self, uid, qid):
        """
        Returns the predicted rating for user UID and question QID
        """
        ui = self.u_index[uid]
        qi = self.q_index[qid]

        prediction = self.umeans[ui]
        sim_total = 0
        neighbor_total = 0
        for uj in self.neighborhood(ui, qi):
            #sim = self.similarity(ui, uj)
            sim = self.pearson_corr(ui, uj)
            neighbor_total += sim * (self.R_mat[uj, qi] - self.umeans[uj])
            sim_total += abs(sim)
        if sim_total > 0.0:
            prediction += (neighbor_total / float(sim_total))
            # Scale back to [0,1]
            prediction = (prediction + 1) / 2
        if prediction > 1.0:
            prediction = 1.0
        elif prediction < 0.0:
            prediction = 0.0
        return prediction

    def score(self, data):
        print("Computing score...")
        return self._mean_absolute_error(data)

    def _mean_absolute_error(self, data):
        """
        Computes MAE of data
        """
        score = 0
        iteration = 0
        for qid, uid, rating in data:
            prediction = self.predict(uid=uid, qid=qid)
            score += abs(prediction - int(rating))
            print("\r\tIteration %d" % iteration, end='')
            sys.stdout.flush()
            iteration += 1
        print("\n\tDone")
        return score / float(self.total_ratings)


if __name__ == '__main__':
    cfilter = CollabFilter()
    print("Solving...")
    cfilter.solve()

# Next:
#  - Use different correlations
#  - Scale ratings to [-1, 1] then scale back linearly
#  - Scale back to [0, 1] with sigmoid (maybe)
#  - Use/don't use means

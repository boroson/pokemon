from collections import defaultdict
import numpy as np
from scipy.stats import pearsonr

from core import Solver, DATADIR, TIMESTAMP, all_nan


N_QUESTIONS = 8095
N_USERS = 28763


class CollabFilter(Solver):
    """
    User-model based Collaborative filter
    """
    def __init__(self):
        self.output = '%s_%s.csv' % (self.__class__.__name__, TIMESTAMP)
        # Dictionary assigning each user an index and each question an index - this is mainly for convenience
        self.q_index = self.create_index_map(filename=DATADIR + 'question_info.txt')
        self.u_index = self.create_index_map(filename=DATADIR + 'user_info.txt')

        # Initialize rating matrix (rows = user indices, columns are question indices)
        self.R_mat = np.full(shape=(N_USERS, N_QUESTIONS), fill_value=np.NaN)
        self.populate_rating_matrix()

        # Precompute user rating means
        print("Computing means...")
        self.umeans = self.compute_means()

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

    def populate_rating_matrix(self):
        """
        Use the training data to populate the rating matrix
        """
        for qid, uid, label in self.file_iter(DATADIR + 'invited_info_train.txt'):
            uindex = self.u_index[uid]
            qindex = self.q_index[qid]
            self.R_mat[uindex, qindex] = float(label)

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
            sim = self.similarity(ui, uj)
            neighbor_total += sim * (self.R_mat[uj, qi] - self.umeans[uj])
            sim_total += abs(sim)
        if sim_total > 0.0:
            prediction += (neighbor_total / float(sim_total))
        if prediction > 1.0:
            print("Bigger than 1.0")
        if prediction < 0.0:
            print("Smaller than 0.0")
        return prediction

if __name__ == '__main__':
    cfilter = CollabFilter()
    print("Solving...")
    cfilter.solve()

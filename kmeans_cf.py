#!/usr/bin/env python

from sklearn.cluster import KMeans as _KMeans
from sklearn.model_selection import KFold
import argparse
import numpy as np

from collaborative_filter import CollabFilter, N_USERS
from core import file_iter, DATADIR


class KMeansCF(CollabFilter):
    def __init__(self, k=8, data=None):
        super(KMeansCF, self).__init__(data=data)
        self.k = k

        # David: Maybe we should try different parameters for KMeans
        print("Clustering...")
        kmeans = _KMeans(
            n_clusters=k
        )
        self.assignments = kmeans.fit_predict(np.nan_to_num(self.R_mat))
        self.clusters = self.create_clusters()
        print("\tDone")

    def create_clusters(self):
        """
        Precompute the cluster each user is in to avoid having to compute it every time we want a user's neighborhood
        """
        clusters = {}
        for k in xrange(self.k):
            clusters[k] = [i for i, cluster in enumerate(self.assignments) if cluster == k]
        return clusters

    def neighborhood(self, ui, qi):
        assignment = self.assignments[ui]
        for uj in self.clusters[assignment]:
            if ui == uj or np.isnan(self.R_mat[uj, qi]):
                continue
            yield uj

    def save_clusters(self):
        with open('clusters.csv', 'w') as f:
            for uid, ui in self.u_index.iteritems():
                cluster = self.assignments[ui]
                row = '%s,%d\n' % (uid, cluster)
                f.write(row)


def cross_validation():
    # Load training data
    X = np.array([row for row in file_iter(DATADIR + 'invited_info_train.txt')])
    # Test k values from 2 to 11
    k_values = range(2, 11)
    # Create K-fold cross validation iterator
    folds = 15
    kf = KFold(n_splits=folds)

    score = {}
    for k in k_values:
        print("\nTesting k = %d..." % k)
        print("-------------------")
        score[k] = 0
        for train_index, test_index in kf.split(X):
            # Split data into training and test set
            training, test = X[train_index], X[test_index]
            classifier = KMeansCF(k=k, data=training)
            score[k] += classifier.score(test)
        score[k] /= float(folds)
    for k, mae in score.iteritems():
        print("K = %d | MAE = %f" % (k, mae))
    best_k = min(score, key=lambda x: score[x])
    print("\nBest K = %d" % best_k)
    return best_k


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', help='Number of clusters (default: 8)', type=int, default=8)
    parser.add_argument('-cv', help='Perform cross validation', action='store_true')
    parser.add_argument('--test', action='store_true', help='Solve against the test data rather than the validation data')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.cv:
        cross_validation()
    else:
        dataset = 'test_nolabel.txt' if args.test else 'validate_nolabel.txt'
        kmeans = KMeansCF(k=args.k)
        print("Solving...")
        #kmeans.save_clusters()
        kmeans.solve(filename=DATADIR+dataset)
        print("\tDone")

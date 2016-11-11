#!/usr/bin/env python

from sklearn.cluster import KMeans as _KMeans
import argparse
import numpy as np

from collaborative_filter import CollabFilter, N_USERS

class KMeansCF(CollabFilter):
    def __init__(self, k=8):
        super(KMeansCF, self).__init__()
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', help='Number of clusters (default: 8)', type=int, default=8)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    kmeans = KMeansCF(k=args.k)
    print("Solving...")
    kmeans.solve()
    print("\tDone")

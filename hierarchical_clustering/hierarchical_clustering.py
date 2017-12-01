import numpy as np


class HierarchicalClustering:
    def __init(self, affine, linkage, n_clusters):
        self.affine = affine
        self.linkage = linkage
        self.n_clusters = n_clusters
        self._points = None
        self._labels = None
        self._cluster_list = None
        self._initial_distance = None
        self._current_distance = None

    def fit(self, X):
        self._points = X
        self._labels = np.arange(len(X))
        pass

    def fit_predict(self, X, y=None):
        self._points = X
        self._labels = np.arange(len(X))
        pass
        y = self._labels
        return y

    def get_params(self, deep=True):
        pass

    def set_params(self, **params):
        pass

    def _step(self):
        pass

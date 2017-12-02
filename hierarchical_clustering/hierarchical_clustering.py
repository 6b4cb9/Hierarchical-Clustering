import numpy as np


class StepInfo:
    def __init__(self):
        self.cluster_list = None
        self.initial_distance = None
        self.current_distance = None


class HierarchicalClustering:
    def __init(self, affine, linkage, n_clusters):
        self.affine = affine
        self.linkage = linkage
        self.n_clusters = n_clusters
        self._points = None
        self._labels = None
        self._step_info = StepInfo()

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
        p,q=np.unravel_index(self._step_info.current_distance.argmin(), self._step_info.current_distance.shape)
        self._step_info.cluster_list[p] = np.concatenate(self._step_info.cluster_list[p],self._step_info.cluster_list[q])



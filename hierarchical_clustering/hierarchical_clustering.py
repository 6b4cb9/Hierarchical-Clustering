import numpy as np


class StepInfo:
    def __init__(self):
        self.cluster_list = np.array([])
        self.initial_distance = np.array([])
        self.current_distance = np.array([])


class HierarchicalClustering:
    def __init__(self, affine, linkage, n_clusters):
        self.affine = affine
        self.linkage = linkage
        self.n_clusters = n_clusters
        self._points = None
        self._labels = None
        self._step_info = StepInfo()

    def fit(self, X):
        self._points = X
        self._labels = np.arange(len(X))
        #cluster list ma byc tyle elementow ile w n_clusters
        while self._step_info.cluster_list > self.n_clusters:
            self._step()

    def fit_predict(self, X, y=None):
        self.fit()
        y = self._labels
        return y

    def get_params(self, deep=True):
       pass

    def set_params(self, **params):
        pass

    def _step(self):
        p, q = np.sort(np.unravel_index(self._step_info.current_distance.argmin(),
                                        self._step_info.current_distance.shape))

        self._step_info.cluster_list[p].merge(q)

        clusters_size = self._step_info.cluster_list.size
        for i in range(clusters_size):
            dist = self._step_info.cluster_list[p].distance(p, i)
            self._step_info.current_distance[p, i] = dist
            self._step_info.current_distance[i, p] = dist

        self._step_info.current_distance = np.delete(self._step_info.current_distance, q, axis=0)
        self._step_info.current_distance = np.delete(self._step_info.current_distance, q, axis=1)

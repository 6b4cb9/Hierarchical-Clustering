import numpy as np
from metric import MetricsFunctions
import cluster

class HierarchicalClustering:
    def __init__(self, affine, linkage, n_clusters):
        self.affine = affine
        self.linkage = linkage
        self.n_clusters = n_clusters
        self._points = None
        self._labels = None
        self._step_info = cluster.Cluster.step_info
        self._step_info.select_class(linkage)

    def fit(self, X):
        #sprawdzic linkage i w zaleznosci od tego odpowiednio uzupelnic
        #na podstawie affine_metric
        #self.cluster_list = np.array(len)
        #addifne -> MetricFunctions('eucl')
        #affine ma stringa, odpowiada funkcji
        X = np.array(X)
        self._points = X
        self._labels = np.arange(len(X))
        self._init_distance(X)
        self._step_info.current_distance = np.copy(self._step_info.initial_distance)
        self._init_cluster_list(X.size)
        
        #cluster list ma byc tyle elementow ile w n_clusters
        while len(self._step_info.cluster_list) > self.n_clusters:
            self._step()

    def _init_distance(self,X):
        metric = MetricsFunctions(self.affine)
        size = X.size
        self._step_info.initial_distance = np.zeros(shape=(size, size), dtype=np.float)
        for i in range(size):
            self._step_info.initial_distance[i, i] = 0
            for j in range(size - i - 1):
                distance = metric.compute(X[i], X[j + i + 1])
                self._step_info.initial_distance[i, j] = distance
                self._step_info.initial_distance[j, i] = distance

    def _init_cluster_list(self, n):
        self._step_info.cluster_list = np.array([self._step_info.cluster_class(i) for i in range(n)])


    def fit_predict(self, X, y=None):
        self.fit(X)
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
            dist = self._step_info.cluster_class.distance(p, i)
            self._step_info.current_distance[p, i] = dist
            self._step_info.current_distance[i, p] = dist

        self._step_info.current_distance = np.delete(self._step_info.current_distance, q, axis=0)
        self._step_info.current_distance = np.delete(self._step_info.current_distance, q, axis=1)
        self._step_info.cluster_list = np.delete(self._step_info.cluster_list, q)

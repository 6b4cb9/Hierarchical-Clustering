import numpy as np
from hierarchical_clustering.metric import MetricsFunctions
import hierarchical_clustering.cluster as cluster

class HierarchicalClustering:
    def __init__(self, n_clusters=2, affinity="euclidean", linkage="ward"):
        self.affinity = affinity
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
        size = np.shape(X)[0]
        self._points = X
        self._labels = np.arange(size)
        self._init_distance(X)
        self._step_info.current_distance = np.copy(self._step_info.initial_distance)
        self._init_cluster_list(size)
        
        #cluster list ma byc tyle elementow ile w n_clusters
        while len(self._step_info.cluster_list) > self.n_clusters:
            self._step()

        for i in range(self.n_clusters):
            cluster_size = self._step_info.cluster_list[i].points_id.size
            for j in range(cluster_size):
                self._labels[self._step_info.cluster_list[i].points_id[j]] = i


    def _init_distance(self,X):
        metric = MetricsFunctions(self.affinity)
        size = np.shape(X)[0]
        self._step_info.initial_distance = np.zeros(shape=(size, size), dtype=np.float)
        for i in range(size):
            self._step_info.initial_distance[i, i] = np.nan
            for j in range(i):
                distance = metric.compute(X[i], X[j])
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
        p, q = np.sort(np.unravel_index(np.nanargmin(self._step_info.current_distance),
                                        self._step_info.current_distance.shape))

        self._step_info.cluster_list[p].merge(q)

        clusters_size = self._step_info.cluster_list.size
        new_distances = np.zeros(shape=(clusters_size), dtype=np.float)
        for i in range(clusters_size):
            if i == p:
                new_distances[i] = np.nan
            else:
                new_distances[i] = self._step_info.cluster_class.distance(p, i)

        self._step_info.current_distance[p,:] = new_distances
        self._step_info.current_distance[:,p] = new_distances

        self._step_info.current_distance = np.delete(self._step_info.current_distance, q, axis=0)
        self._step_info.current_distance = np.delete(self._step_info.current_distance, q, axis=1)
        self._step_info.cluster_list = np.delete(self._step_info.cluster_list, q)

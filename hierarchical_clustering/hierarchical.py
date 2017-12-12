import numpy as np
from hierarchical_clustering.metric import MetricsFunctions
import hierarchical_clustering.cluster as cluster

class HierarchicalClustering:
    """
    Main class for hierarchical clustering.
    """
    def __init__(self, n_clusters=2, affinity="euclidean", linkage="ward"):
        """
        Constructor.
        :param n_clusters: number of expected final clusters.
        :param affinity: metric used to compute the linkage; can be 'eucl', 'l1', 'l2', 'euclidean', 'manhattan'; default is 'euclidean'
        :param linkage: which linkage criterion to use; can be 'ward', 'complete', 'max', 'average'; default is 'ward'
        """
        self.affinity = affinity
        self.linkage = linkage
        self.n_clusters = n_clusters
        self._points = None
        self._labels = None
        self._step_info = cluster.Cluster.step_info
        self._step_info.select_class(linkage)

    def fit(self, X):
        """
        Fit the hierarchical clustering on the data.
        :param X: data
        :return:
        """
        X = np.array(X)
        size = np.shape(X)[0]
        self._points = X
        self._labels = np.arange(size)
        self._init_distance(X)
        self._step_info.current_distance = np.copy(self._step_info.initial_distance)
        self._init_cluster_list(size)
        
        while len(self._step_info.cluster_list) > self.n_clusters:
            self._step()

        for i in range(self.n_clusters):
            cluster_size = self._step_info.cluster_list[i].points_id.size
            for j in range(cluster_size):
                self._labels[self._step_info.cluster_list[i].points_id[j]] = i


    def _init_distance(self,X):
        """
        Create initial distance matrix.
        :param X: data
        :return:
        """
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
        """
        Create initial cluster list.
        :param n: number of clusters
        :return:
        """
        self._step_info.cluster_list = np.array([self._step_info.cluster_class(i) for i in range(n)])


    def fit_predict(self, X):
        """
        Fit the hierarchical clustering on the data and return labels.
        :param X: data
        :return:
        """
        self.fit(X)
        y = np.copy(self._labels)
        return y

    def _step(self):
        """
        Merge two clusters into one.
        :return:
        """
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

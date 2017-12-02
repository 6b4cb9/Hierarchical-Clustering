import unittest
import numpy as np
import sklearn.cluster

from hierarchical_clustering import hierarchical_clustering


class TestWardClustering(unittest.TestCase):

    def setUp(self):
        pass

    def test_euclidean_metric(self):

        test_data = np.array([[0, 1, 3, 12, 12, 11, 13, 1055], [-1, -1, -1, 0, 0, 0, 0, 1]])
        test_data = test_data.transpose()

        reference = sklearn.cluster.AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward")
        reference_ans = reference.fit_predict(test_data)

        test = hierarchical_clustering.HierarchicalClustering(n_clusters=3, affinity="euclidean", linkage="ward")
        test_ans = test.fit_predict(test_data)

        self.assertEqual(test_ans, reference_ans)

    def test_l2_metric(self):
        pass


if __name__ == "__main__":
    unittest.main()


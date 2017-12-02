import numpy as np
import unittest

from hierarchical_clustering import cluster
from hierarchical_clustering import hierarchical_clustering


class TestClusterInit(unittest.TestCase):
    def setUp(self):
        pass

    def test_distance(self):
        step_info = hierarchical_clustering.StepInfo()
        step_info.initial_distance = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 2], [3, 2, 1, 0]])
        step_info.current_distance = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 2], [3, 2, 1, 0]])
        clusters = list()
        for i in range(4):
            clusters.append(cluster.ClusterWard(i, step_info))
        step_info.cluster_list = np.array(clusters)
        self.assertEqual(clusters[1].distance(1, 3), 2.0)



if __name__ == '__main__':
    unittest.main()
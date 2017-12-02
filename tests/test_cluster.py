import numpy as np
import unittest

from hierarchical_clustering import cluster


class TestWardCluster(unittest.TestCase):
    def setUp(self):
        pass

    def test__fail_init(self):
        step_info = cluster.Cluster.step_info
        self.assertRaises(Exception, step_info.select_class, "asdasd")

    def test_initial_conditions(self):
        step_info = cluster.Cluster.step_info
        step_info.select_class("ward")
        step_info.initial_distance = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 2], [3, 2, 1, 0]])
        step_info.current_distance = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 2], [3, 2, 1, 0]])
        clusters = list()
        for i in range(4):
            clusters.append(step_info.cluster_class(i))
        step_info.cluster_list = np.array(clusters)
        self.assertEqual(step_info.cluster_class.distance(1, 3), 2.0)

    def test_merge(self):
        step_info = cluster.Cluster.step_info
        step_info.select_class("ward")
        step_info.initial_distance = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 2], [3, 2, 1, 0]])
        step_info.current_distance = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 2], [3, 2, 1, 0]])
        clusters = list()
        for i in range(4):
            clusters.append(step_info.cluster_class(i))
        step_info.cluster_list = np.array(clusters)
        clusters[1].merge(2)
        self.assertEqual(str(clusters[1]), "[1 2]")


if __name__ == '__main__':
    unittest.main()
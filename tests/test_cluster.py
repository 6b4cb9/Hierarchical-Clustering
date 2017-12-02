import numpy as np
import unittest

from hierarchical_clustering import cluster


class TestWardCluster(unittest.TestCase):
    def setUp(self):
        pass

    def test__fail_init(self):
        """Tests if StepInfo throw exception, when initialize with wrong value."""
        step_info = cluster.Cluster.step_info
        self.assertRaises(Exception, step_info.select_class, "asdasd")

    def test_initial_conditions(self):
        """Test distance before first merge"""
        step_info = cluster.Cluster.step_info
        step_info.select_class("ward")
        step_info.initial_distance = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 2], [3, 2, 1, 0]])
        step_info.current_distance = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 2], [3, 2, 1, 0]])
        clusters = list()
        for i in range(4):
            clusters.append(step_info.cluster_class(i))
        step_info.cluster_list = np.array(clusters)
        self.assertEqual(step_info.cluster_class.distance(1, 3), 2.0)

    def test_further_state(self):
        """Tests if distance is calculated properly."""
        step_info = cluster.Cluster.step_info
        step_info.select_class("ward")
        step_info.initial_distance = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        step_info.current_distance = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        clusters = list()
        for i in range(3):
            clusters.append(step_info.cluster_class(i))
        step_info.cluster_list = np.array(clusters)
        clusters[0].merge(1)
        self.assertEqual(step_info.cluster_class.distance(0, 2), 5/3)

    def test_merge(self):
        """Tests if merge work correctly"""
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


class TestCompleteCluster(unittest.TestCase):
    def setUp(self):
        pass

    def test__fail_init(self):
        """Tests if StepInfo throw exception, when initialize with wrong value."""
        step_info = cluster.Cluster.step_info
        self.assertRaises(Exception, step_info.select_class, "asdasd")

    def test_initial_conditions(self):
        """Test distance before first merge"""
        step_info = cluster.Cluster.step_info
        step_info.select_class("complete")
        step_info.initial_distance = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 2], [3, 2, 1, 0]])
        step_info.current_distance = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 2], [3, 2, 1, 0]])
        clusters = list()
        for i in range(4):
            clusters.append(step_info.cluster_class(i))
        step_info.cluster_list = np.array(clusters)
        self.assertEqual(step_info.cluster_class.distance(1, 3), 2.0)

    def test_further_state(self):
        """Tests if distance is calculated properly."""
        step_info = cluster.Cluster.step_info
        step_info.select_class("complete")
        step_info.initial_distance = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        step_info.current_distance = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        clusters = list()
        for i in range(3):
            clusters.append(step_info.cluster_class(i))
        step_info.cluster_list = np.array(clusters)
        clusters[0].merge(1)
        self.assertEqual(step_info.cluster_class.distance(0, 2), 2)

    def test_merge(self):
        """Tests if merge work correctly"""
        step_info = cluster.Cluster.step_info
        step_info.select_class("complete")
        step_info.initial_distance = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 2], [3, 2, 1, 0]])
        step_info.current_distance = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 2], [3, 2, 1, 0]])
        clusters = list()
        for i in range(4):
            clusters.append(step_info.cluster_class(i))
        step_info.cluster_list = np.array(clusters)
        clusters[1].merge(2)
        self.assertEqual(str(clusters[1]), "[1 2]")


class TestAverageCluster(unittest.TestCase):
    def setUp(self):
        pass

    def test__fail_init(self):
        """Tests if StepInfo throw exception, when initialize with wrong value."""
        step_info = cluster.Cluster.step_info
        self.assertRaises(Exception, step_info.select_class, "asdasd")

    def test_initial_conditions(self):
        """Test distance before first merge"""
        step_info = cluster.Cluster.step_info
        step_info.select_class("average")
        step_info.initial_distance = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 2], [3, 2, 1, 0]])
        step_info.current_distance = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 2], [3, 2, 1, 0]])
        clusters = list()
        for i in range(4):
            clusters.append(step_info.cluster_class(i))
        step_info.cluster_list = np.array(clusters)
        self.assertEqual(step_info.cluster_class.distance(1, 3), 2.0)

    def test_further_state(self):
        """Tests if distance is calculated properly."""
        step_info = cluster.Cluster.step_info
        step_info.select_class("average")
        step_info.initial_distance = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        step_info.current_distance = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        clusters = list()
        for i in range(3):
            clusters.append(step_info.cluster_class(i))
        step_info.cluster_list = np.array(clusters)
        clusters[0].merge(1)
        self.assertEqual(step_info.cluster_class.distance(0, 2), 1.5)

    def test_merge(self):
        """Tests if merge work correctly"""
        step_info = cluster.Cluster.step_info
        step_info.select_class("average")
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
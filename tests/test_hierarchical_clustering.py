import unittest
import numpy as np
import sklearn.cluster

from hierarchical_clustering import hierarchical


class TestWardClustering(unittest.TestCase):
    """
    Compare results of clustering on surface between our ward function
    and agglomerative clustering from scikit-learn.
    """

    def setUp(self):
        """Empty setup function"""
        pass

    def test_euclidean_metric(self):
        """Tests of euclidean metrics."""
        test_data = np.array([[0, 1, 3, 12, 12, 11, 13, 1055], [-1, -1, -1, 0, 0, 0, 0, 1]])
        test_data = test_data.transpose()

        reference = sklearn.cluster.AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward")
        reference_ans = reference.fit_predict(test_data)

        test = hierarchical.HierarchicalClustering(n_clusters=3, affinity="euclidean", linkage="ward")
        test_ans = test.fit_predict(test_data)

        ans = np.array_equal(test_ans, reference_ans)
        msg = "get: " + str(test_ans) + "instead of: " + str(reference_ans)
        self.assertEqual(ans, True, msg)


class TestAverageClustering(unittest.TestCase):
    """
    Thise class compare results of clustering on surface between our average function
    and agglomerative clustering from scikit-learn.
    """
    def setUp(self):
        """Empty setup function"""
        pass

    def test_euclidean_in_simple_case(self):
        """Tests of euclidean metrics."""

        test_data = np.array([[0, 1, 3, 12, 12, 11, 13, 1055], [-1, -1, -1, 0, 0, 0, 0, 1]])
        test_data = test_data.transpose()

        reference = sklearn.cluster.AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="average")
        reference_ans = reference.fit_predict(test_data)

        test = hierarchical.HierarchicalClustering(n_clusters=3, affinity="euclidean", linkage="average")
        test_ans = test.fit_predict(test_data)

        ans = np.array_equal(test_ans, reference_ans)
        msg = "get: " + str(test_ans) + "instead of: " + str(reference_ans)
        self.assertEqual(ans, True, msg)

    def test_l2_in_simple_case(self):
        """Tests of L2 metrics."""
        test_data = np.array([[0, 1, 3, 12, 12, 11, 13, 1055], [-1, -1, -1, 0, 0, 0, 0, 1]])
        test_data = test_data.transpose()

        reference = sklearn.cluster.AgglomerativeClustering(n_clusters=3, affinity="l2", linkage="average")
        reference_ans = reference.fit_predict(test_data)

        test = hierarchical.HierarchicalClustering(n_clusters=3, affinity="l2", linkage="average")
        test_ans = test.fit_predict(test_data)

        ans = np.array_equal(test_ans, reference_ans)
        msg = "get: " + str(test_ans) + "instead of: " + str(reference_ans)
        self.assertEqual(ans, True, msg)


class TestCompleteClustering(unittest.TestCase):
    """
    Thise class compare results of clustering on surface between our complete function
    and agglomerative clustering from scikit-learn.
    """
    def setUp(self):
        """Empty setup function"""
        pass

    def test_euclidean_in_simple_case(self):
        """Tests of euclidean metrics."""
        test_data = np.array([[0, 1, 3, 12, 12, 11, 13, 1055], [-1, -1, -1, 0, 0, 0, 0, 1]])
        test_data = test_data.transpose()

        reference = sklearn.cluster.AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="complete")
        reference_ans = reference.fit_predict(test_data)

        test = hierarchical.HierarchicalClustering(n_clusters=3, affinity="euclidean", linkage="complete")
        test_ans = test.fit_predict(test_data)

        ans = np.array_equal(test_ans, reference_ans)
        msg = "get: " + str(test_ans) + "instead of: " + str(reference_ans)
        self.assertEqual(ans, True, msg)

    def test_l2_in_simple_case(self):
        """Tests of L2 metrics."""
        test_data = np.array([[0, 1, 3, 12, 12, 11, 13, 1055], [-1, -1, -1, 0, 0, 0, 0, 1]])
        test_data = test_data.transpose()

        reference = sklearn.cluster.AgglomerativeClustering(n_clusters=3, affinity="l2", linkage="complete")
        reference_ans = reference.fit_predict(test_data)

        test = hierarchical.HierarchicalClustering(n_clusters=3, affinity="l2", linkage="complete")
        test_ans = test.fit_predict(test_data)

        ans = np.array_equal(test_ans, reference_ans)
        msg = "get: " + str(test_ans) + "instead of: " + str(reference_ans)
        self.assertEqual(ans, True, msg)


class GeneralTest(unittest.TestCase):
    def setUp(self):
        """Empty setup function"""
        pass

    def test_argument_matching(self):
        """Tests if our interface is compatible with sklearn."""
        test_data = np.array([[0, 1, 3, 12, 12, 11, 13, 1055], [-1, -1, -1, 0, 0, 0, 0, 1]])
        test_data = test_data.transpose()

        reference = sklearn.cluster.AgglomerativeClustering(3, "euclidean")
        reference_ans = reference.fit_predict(test_data)

        test = hierarchical.HierarchicalClustering(3, "euclidean")
        test_ans = test.fit_predict(test_data)

        ans = np.array_equal(test_ans, reference_ans)
        msg = "get: " + str(test_ans) + "instead of: " + str(reference_ans)
        self.assertEqual(ans, True, msg)


if __name__ == "__main__":
    unittest.main()


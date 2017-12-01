import unittest
import numpy as np

from hierarchical_clustering import metric


class TestEucl(unittest.TestCase):

    def setUp(self):
        pass

    def test_list_input(self):
        point_a = [6, 3.1]
        point_b = [7.2, 1/3]
        self.assertEqual(metric.eucl(point_a, point_b), np.linalg.norm(point_a - point_b))

    def test_np_array_input(self):
        point_a = np.array([6, 3.1])
        point_b = np.array([7.2, 1/3])
        self.assertEqual(metric.eucl(point_a, point_b), np.linalg.norm(point_a - point_b))

    def test_str_input(self):
        self.assertRaises(TypeError, metric.eucl, "0.312", "11")

    def test_te_Same_point(self):
        point = np.array([6, 3.1])
        self.assertEqual(metric.eucl(point, point), 0)
        pass

    def test_none_arg(self):
        self.assertRaises(TypeError, metric.eucl, None, 3)

    def test_symmetry(self):
        point_a = [6.0000000001, 7]
        point_b = [13.5000000001, -2]
        self.assertEqual(metric.eucl(point_a, point_b), metric.eucl(point_b, point_a))

    def test_1D_case(self):
        self.assertEqual(metric.eucl(3, 6), np.linalg.norm(3-6))

    def test_2D_case(self):
        point_a = [6, 7]
        point_b = [13.5, -2]
        self.assertEqual(metric.eucl(point_a, point_b), np.linalg.norm(point_a - point_b))

    def test_3D_case(self):
        point_a = [6, 7, 0]
        point_b = [13.5, -2, 3.3]
        self.assertEqual(metric.eucl(point_a, point_b), np.linalg.norm(point_a - point_b))

    def test_nD_case(self):
        n = 123456
        point_a = list(range(n))
        point_b = list(range(n, 0))
        self.assertEqual(metric.eucl(point_a, point_b), np.linalg.norm(point_a - point_b))


if __name__ == "__main__":
    unittest.main()


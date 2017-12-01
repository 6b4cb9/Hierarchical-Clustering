import unittest
import numpy as np

from hierarchical_clustering import metric


class TestEucl(unittest.TestCase):

    def setUp(self):
        pass

    def test_list_input(self):
        """
        Tests if function accept list arg. Should throw TypeError.
        """
        point_a = [6, 3.1]
        point_b = [7.2, 1/3]
        self.assertRaises(TypeError, metric.eucl, point_a, point_b)

    def test_np_array_input(self):
        """
        Tests if function accept np.array as argument. It should pass.
        """
        point_a = np.array([6, 3.1])
        point_b = np.array([7.2, 1/3])
        self.assertEqual(metric.eucl(point_a, point_b), np.linalg.norm(point_a - point_b))

    def test_str_input(self):
        """
        Tests if function accept string arg. Should throw TypeError.
        """
        self.assertRaises(TypeError, metric.eucl, "0.312", "11")

    def test_te_Same_point(self):
        """
        Tests if distance from point to the same point is zero.
        """
        point = np.array([6, 3.1])
        self.assertEqual(metric.eucl(point, point), 0)
        pass

    def test_none_arg(self):
        """
        Tests if function accept None arg. Should throw TypeError.
        """
        self.assertRaises(TypeError, metric.eucl, None, 3)

    def test_symmetry(self):
        """
        Tests if distance from point A to point B is equal to distance from B to A.
        """
        point_a = np.array([6.0000000001, 7])
        point_b = np.array([13.5000000001, -2])
        self.assertEqual(metric.eucl(point_a, point_b), metric.eucl(point_b, point_a))

    def test_1D_case(self):
        """
        Tests metric in 1 dimensional space. Result is compered with np.metric.
        """
        self.assertEqual(metric.eucl(3, 6), np.linalg.norm(3-6))

    def test_2D_case(self):
        """
        Tests metric in 2 dimensional space. Result is compered with np.metric.
        """
        point_a = np.array([6, 7])
        point_b = np.array([13.5, -2])
        self.assertEqual(metric.eucl(point_a, point_b), np.linalg.norm(point_a - point_b))

    def test_3D_case(self):
        """
        Tests metric in 3 dimensional space. Result is compered with np.metric.
        """
        point_a = np.array([6, 7, 0])
        point_b = np.array([13.5, -2, 3.3])
        self.assertEqual(metric.eucl(point_a, point_b), np.linalg.norm(point_a - point_b))

    def test_nD_case(self):
        """
        Tests metric in multi-dimensional space. Result is compered with np.metric.
        """
        n = 123456
        point_a = np.array(list(range(n)))
        point_b = np.array(list(range(n, 0, -1)))
        self.assertEqual(metric.eucl(point_a, point_b), np.linalg.norm(point_a - point_b))


class TestL1(unittest.TestCase):

    def setUp(self):
        pass

    def test_list_input(self):
        """
        Tests if function accept list arg. Should throw TypeError.
        """
        point_a = [6, 3.1]
        point_b = [7.2, 1/3]
        self.assertRaises(TypeError, metric.l1, point_a, point_b)

    def test_np_array_input(self):
        """
        Tests if function accept np.array as argument. It should pass.
        """
        point_a = np.array([6, 3.1])
        point_b = np.array([7.2, 1/3])
        self.assertEqual(metric.l1(point_a, point_b), np.linalg.norm(point_a - point_b, ord=1))

    def test_str_input(self):
        """
        Tests if function accept string arg. Should throw TypeError.
        """
        self.assertRaises(TypeError, metric.l1, "0.312", "11")

    def test_te_Same_point(self):
        """
        Tests if distance from point to the same point is zero.
        """
        point = np.array([6, 3.1])
        self.assertEqual(metric.l1(point, point), 0)
        pass

    def test_none_arg(self):
        """
        Tests if function accept None arg. Should throw TypeError.
        """
        self.assertRaises(TypeError, metric.l1, None, 3)

    def test_symmetry(self):
        """
        Tests if distance from point A to point B is equal to distance from B to A.
        """
        point_a = np.array([6.0000000001, 7])
        point_b = np.array([13.5000000001, -2])
        self.assertEqual(metric.l1(point_a, point_b), metric.l1(point_b, point_a))

    def test_1D_case(self):
        """
        Tests metric in 1 dimensional space. Result is compered with np.metric.
        """
        self.assertEqual(metric.l1(3, 6), np.linalg.norm(3-6))

    def test_2D_case(self):
        """
        Tests metric in 2 dimensional space. Result is compered with np.metric.
        """
        point_a = np.array([6, 7])
        point_b = np.array([13.5, -2])
        self.assertEqual(metric.l1(point_a, point_b), np.linalg.norm(point_a - point_b, ord=1))

    def test_3D_case(self):
        """
        Tests metric in 3 dimensional space. Result is compered with np.metric.
        """
        point_a = np.array([6, 7, 0])
        point_b = np.array([13.5, -2, 3.3])
        self.assertEqual(metric.l1(point_a, point_b), np.linalg.norm(point_a - point_b, ord=1))

    def test_nD_case(self):
        """
        Tests metric in multi-dimensional space. Result is compered with np.metric.
        """
        n = 123456
        point_a = np.array(list(range(n)))
        point_b = np.array(list(range(n, 0, -1)))
        self.assertEqual(metric.l1(point_a, point_b), np.linalg.norm(point_a - point_b, ord=1))


class TestL2(unittest.TestCase):

    def setUp(self):
        pass

    def test_list_input(self):
        """
        Tests if function accept list arg. Should throw TypeError.
        """
        point_a = [6, 3.1]
        point_b = [7.2, 1/3]
        self.assertRaises(TypeError, metric.l2, point_a, point_b)

    def test_np_array_input(self):
        """
        Tests if function accept np.array as argument. It should pass.
        """
        point_a = np.array([6, 3.1])
        point_b = np.array([7.2, 1/3])
        self.assertEqual(metric.l2(point_a, point_b), np.linalg.norm(point_a - point_b, ord=2))

    def test_str_input(self):
        """
        Tests if function accept string arg. Should throw TypeError.
        """
        self.assertRaises(TypeError, metric.l2, "0.312", "11")

    def test_te_Same_point(self):
        """
        Tests if distance from point to the same point is zero.
        """
        point = np.array([6, 3.1])
        self.assertEqual(metric.l2(point, point), 0)
        pass

    def test_none_arg(self):
        """
        Tests if function accept None arg. Should throw TypeError.
        """
        self.assertRaises(TypeError, metric.l2, None, 3)

    def test_symmetry(self):
        """
        Tests if distance from point A to point B is equal to distance from B to A.
        """
        point_a = np.array([6.0000000001, 7])
        point_b = np.array([13.5000000001, -2])
        self.assertEqual(metric.l2(point_a, point_b), metric.l2(point_b, point_a))

    def test_1D_case(self):
        """
        Tests metric in 1 dimensional space. Result is compered with np.metric.
        """
        self.assertEqual(metric.l2(3, 6), np.linalg.norm(3-6))

    def test_2D_case(self):
        """
        Tests metric in 2 dimensional space. Result is compered with np.metric.
        """
        point_a = np.array([6, 7])
        point_b = np.array([13.5, -2])
        self.assertEqual(metric.l2(point_a, point_b), np.linalg.norm(point_a - point_b, ord=2))

    def test_3D_case(self):
        """
        Tests metric in 3 dimensional space. Result is compered with np.metric.
        """
        point_a = np.array([6, 7, 0])
        point_b = np.array([13.5, -2, 3.3])
        self.assertEqual(metric.l2(point_a, point_b), np.linalg.norm(point_a - point_b, ord=2))

    def test_nD_case(self):
        """
        Tests metric in multi-dimensional space. Result is compered with np.metric.
        """
        n = 123456
        point_a = np.array(list(range(n)))
        point_b = np.array(list(range(n, 0, -1)))
        self.assertEqual(metric.l2(point_a, point_b), np.linalg.norm(point_a - point_b, ord=2))


if __name__ == "__main__":
    unittest.main()


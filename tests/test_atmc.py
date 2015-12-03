import unittest
import numpy as np
import numpy.testing as nptest

import atmc


class TestFindDeviations(unittest.TestCase):

    def test_equal_arrays(self):
        A = np.column_stack((np.arange(100),
                             np.arange(100),
                             np.arange(100)))
        B = A.copy()

        devs = atmc.mcopt_wrapper.find_deviations(A, B)
        expect = np.zeros((100, 2), dtype='float64')
        nptest.assert_allclose(devs, expect, atol=1e-6)

    def test_plus_const(self):
        A = np.column_stack((np.arange(100),
                             np.arange(100),
                             np.arange(100)))
        B = np.column_stack((np.arange(100),
                             np.arange(100) + 100,
                             np.arange(100)))

        devs = atmc.mcopt_wrapper.find_deviations(A, B)
        expect = np.column_stack((np.zeros(100), -np.full(100, 100)))
        nptest.assert_allclose(devs, expect, atol=1e-8)

    def test_vs_zero(self):
        A = np.column_stack((np.zeros(100),
                             np.zeros(100),
                             np.arange(100)))
        B = np.column_stack((np.arange(0, 100),
                             np.arange(200, 300),
                             np.arange(100)))

        devs = atmc.mcopt_wrapper.find_deviations(A, B)
        expect = -B[:, :2]
        nptest.assert_allclose(devs, expect, atol=1e-8)

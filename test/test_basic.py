import os
import unittest

import h5py
import numpy as np
from numpy.testing import assert_array_equal

import zh5


class Basic(unittest.TestCase):
    def test_1d(self):
        NAME = "1d.h5"

        with h5py.File(NAME, "w") as f:
            f.create_dataset("1d", dtype="f8", shape=(10,))
            f.create_dataset("1dchunks", dtype="f4", shape=(10,), chunks=(2,))
            f.create_dataset("1dfilters", dtype="f8", shape=(10,), chunks=(2,), fletcher32=True, shuffle=True,
                             compression="gzip")
            f["1d"][...] = list(range(10))
            f["1dchunks"][...] = list(range(10))
            f["1dfilters"][...] = list(range(10))

        f = zh5.File(NAME)
        # assert_array_equal(f["1d"][:], np.arange(10))
        assert_array_equal(f["1dchunks"][:], np.arange(10))
        assert_array_equal(f["1dfilters"][:], np.arange(10))
        assert_array_equal(f["1dchunks"][5:], np.arange(5, 10))
        assert_array_equal(f["1dfilters"][5:], np.arange(5, 10))
        assert_array_equal(f["1dchunks"][0], np.arange(1))
        assert_array_equal(f["1dfilters"][0], np.arange(1))
        f.close()

        os.remove(NAME)

    def test2d(self):
        NAME = "2d.h5"

        with h5py.File(NAME, "w") as f:
            f.create_dataset("2d", shape=(10, 10), chunks=(3, 3), compression="gzip", compression_opts=9)
            f["2d"][...] = np.arange(100).reshape((10, 10))

        arr = np.arange(100).reshape((10, 10))
        f = zh5.File(NAME)
        self.assertEqual(f["2d"][0, 0], 0)
        assert_array_equal(f["2d"][3:, 6:9], arr[3:, 6:9])
        assert_array_equal(f["2d"][8:, 8:], arr[-2:, -2:])
        # assert_array_equal(f["2d"][-2:, -2:], arr[-2:, -2:])
        f.close()

        os.remove(NAME)

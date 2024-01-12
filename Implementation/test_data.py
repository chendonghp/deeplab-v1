import unittest
from data import rgb2onehot, colorDict
import numpy as np


class DataTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.label = np.array(list(colorDict.values()))  # NX3
        self.label_1d = np.asarray(range(len(colorDict.values()))) # N
        self.size = len(colorDict.values())
        # print(self.size) -> 6

    def tearDown(self) -> None:
        return super().tearDown()

    def test_rgb2onehot(self):
        label_map_3d = [
            np.roll(self.label, shift=i, axis=0) for i in range(len(self.label))
        ]
        label_map_3d = np.stack(label_map_3d)  # NxNx3
        label_map_2d = [np.roll(self.label_1d, shift=i, axis=0) for i in range(len(self.label_1d))]
        label_map_2d = np.stack(label_map_2d)
        # print(label_map_2d.shape) -> NxN
        self.assertEqual(label_map_3d.shape, (self.size, self.size, 3), "The shape doesn't match!") 
        self.assertTrue(np.all(label_map_2d == rgb2onehot(label_map_3d, colorDict)), "The label is not match!")

if __name__ == "__main__":
    unittest.main()

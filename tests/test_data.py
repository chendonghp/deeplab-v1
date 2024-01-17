# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from Implementation.data import colorDict, compute_mean_std, onehot2rgb, rgb2onehot


class DataTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.label = np.array(list(colorDict.values()))  # NX3
        self.label_1d = np.asarray(range(len(colorDict.values())))  # N
        self.size = len(colorDict.values())
        # print(self.size) -> 6

    def tearDown(self) -> None:
        return super().tearDown()

    def test_rgb2onehot(self):
        label_map_3d = [
            np.roll(self.label, shift=i, axis=0) for i in range(len(self.label))
        ]
        label_map_3d = np.stack(label_map_3d)  # NxNx3
        label_map_2d = [
            np.roll(self.label_1d, shift=i, axis=0) for i in range(len(self.label_1d))
        ]
        label_map_2d = np.stack(label_map_2d)
        # print(label_map_2d.shape) -> NxN
        self.assertEqual(
            label_map_3d.shape, (self.size, self.size, 3), "The shape doesn't match!"
        )
        self.assertTrue(
            np.all(label_map_2d == rgb2onehot(label_map_3d, colorDict)),
            "The label is not match!",
        )

    def test_onehot2rgb(self):
        mask_3d = np.asarray(
            [[[255, 255, 255], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]]
        )
        mask_2d = np.asarray([[0, 1], [2, 3]])
        self.assertTrue(np.all(onehot2rgb(mask_2d, colorDict) == mask_3d))

    def test_compute_mean_std(self):
        # Create a dummy DataLoader with mock data
        mock_data = [(torch.randn(3, 32, 32), torch.zeros([1])) for _ in range(10)]
        mock_dataloader = DataLoader(mock_data, batch_size=1, shuffle=False)

        # Call the function to compute mean and std
        mean, std = compute_mean_std(mock_dataloader)

        # Assert the expected mean and std values
        expected_mean = torch.zeros(3)  # Replace with your expected mean values
        expected_std = torch.ones(3)  # Replace with your expected std values
        print(mean, std)
        self.assertTrue(torch.allclose(mean, expected_mean, atol=5e-2))
        self.assertTrue(torch.allclose(std, expected_std, atol=5e-2))


if __name__ == "__main__":
    unittest.main()

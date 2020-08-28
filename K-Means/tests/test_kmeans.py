import unittest
import numpy as np

from kmeans import KMeans, random_initialize, plus_plus_initialize


class TestKMeans(unittest.TestCase):
    def test_random_initialize(self):
        data = np.array([[1, 2, 3], [4, 5, 6]])
        means = random_initialize(data, 2)
        self.assertEqual(2, len(means))
        self.assertIsInstance(means, list)
        for i in range(4):
            means = random_initialize(data, 2)
            self.assertFalse(all(means[0] == means[1]))

    def test_plus_plus_initialize(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        means = plus_plus_initialize(data, 2)
        self.assertEqual(2, len(means))
        self.assertIsInstance(means, list)

    def test_initialize(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        model = KMeans(2)
        model.initialize(data)
        self.assertNotEqual(0, len(model.means))

    def test_fit_predict(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        model = KMeans(2)
        model.fit(data)
        labels = model.predict(data)
        self.assertEqual(len(labels), len(data))
        self.assertGreaterEqual(2, len(np.unique(labels)))
        for label in labels:
            self.assertTrue(isinstance(label, np.int64) or
                            isinstance(label, np.int32) or
                            isinstance(label, int))

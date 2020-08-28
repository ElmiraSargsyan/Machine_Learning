import unittest
import numpy as np

from mixture import multi_normal, evaluate_loss, GaussianMixtureModel


class TestGMM(unittest.TestCase):
    def test_multi_normal(self):
        prob = multi_normal(np.array([2, 3]), mean=[0, 0],
                            covariance=np.array([[4, 1], [1, 4]]))
        self.assertIsInstance(prob, float)

    def test_evaluate_loss(self):
        data = np.array([[0.87, 1.46], [1.71, 1.11], [1.03, 1.90],
                         [-2.4, -0.3], [-2.6, 0.12], [-1.80, -0.3]])
        weights = [1, 3]
        centers = [np.array([1.5, 1.5]), np.array([-2.5, -2])]
        covariances = [np.array([[3.4, 1], [1, 3.4]]),
                       np.array([[2.9, 0.5], [0.5, 2.9]])]

        loss = evaluate_loss(data, num_mixtures=2, weights=weights,
                             centers=centers, covariances=covariances)
        self.assertIsInstance(loss, float)

    def test_fit(self):
        data = np.array([[0.87, 1.46], [1.71, 1.11], [1.03, 1.90],
                         [-2.4, -0.3], [-2.6, 0.12], [-1.80, -0.3]])
        model = GaussianMixtureModel(2)
        model.fit(data)
        self.assertNotEqual(0, len(model.covariances))
        self.assertTupleEqual((2, 2), model.covariances[0].shape)

        self.assertNotEqual(0, len(model.centers))
        self.assertTupleEqual(data[0].shape, model.centers[0].shape)

        self.assertNotEqual(0, len(model.weights))
        self.assertIsInstance(model.weights[0], float)

        self.assertNotEqual(0, len(model.r))
        self.assertTupleEqual(data.shape, model.r.shape)

    def test_predict_cluster(self):
        data = np.array([[0.87, 1.46], [1.71, 1.11], [1.03, 1.90],
                         [-2.4, -0.3], [-2.6, 0.12], [-1.80, -0.3]])
        model = GaussianMixtureModel(2)
        model.fit(data)
        labels = model.predict_cluster(data)
        self.assertEqual(len(labels), len(data))
        self.assertGreaterEqual(2, len(np.unique(labels)))
        for label in labels:
            self.assertTrue(isinstance(label, np.int64) or
                            isinstance(label, np.int32) or
                            isinstance(label, int))

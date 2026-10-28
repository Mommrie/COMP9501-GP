"""Mathematical contracts independent of the optimization implementation."""
import unittest
import numpy as np
from csa import CSA, PCA
from csa_tensor import TensorCSA, mode_product, unfold_batch


class TensorTests(unittest.TestCase):
    def test_mode_product_matches_explicit_index_sums(self):
        rng = np.random.default_rng(8)
        x, a = rng.normal(size=(2, 3, 4, 5)), rng.normal(size=(6, 4))
        got = mode_product(x, a, 1)
        expected = np.einsum('nijk,aj->niak', x, a)
        np.testing.assert_allclose(got, expected, atol=1e-12)
        flat = unfold_batch(x, 1)
        np.testing.assert_allclose(flat @ flat.T, np.einsum('nijk,nilk->jl', x, x))

    def test_2d_matches_frozen_implementation(self):
        x = np.random.default_rng(9).normal(size=(35, 8, 7))
        old, new = CSA(3).fit(x), TensorCSA((3, 3)).fit(x)
        np.testing.assert_allclose(old.reconstruct(x), new.reconstruct(x), atol=1e-9)
        np.testing.assert_allclose(old.history, new.history, atol=1e-9)

    def test_1d_is_pca(self):
        x = np.random.default_rng(10).normal(size=(30, 9))
        c, p = TensorCSA((4,)).fit(x), PCA(4).fit(x)
        np.testing.assert_allclose(c.reconstruct(x), p.reconstruct(x), atol=1e-11)

    def test_3d_exact_separable_reconstruction_and_kronecker(self):
        rng = np.random.default_rng(11)
        factors = [np.linalg.qr(rng.normal(size=(m, r)))[0]
                   for m, r in [(7, 2), (6, 3), (5, 2)]]
        train = rng.normal(size=(40, 2, 3, 2))
        test = rng.normal(size=(10, 2, 3, 2))
        mean = rng.normal(size=(7, 6, 5))
        def expand(core):
            return np.einsum('ia,jb,kc,nabc->nijk', *factors, core) + mean
        x, heldout = expand(train), expand(test)
        model = TensorCSA((2, 3, 2)).fit(x)
        np.testing.assert_allclose(model.reconstruct(heldout), heldout, atol=1e-11)
        kron = np.kron(np.kron(*model.factors[:2]), model.factors[2])
        np.testing.assert_allclose((x-model.mean).reshape(40, -1) @ kron,
                                   model.transform(x).reshape(40, -1), atol=1e-11)

    def test_random_3d_energy_and_pca_bound(self):
        x = np.random.default_rng(12).normal(size=(50, 5, 4, 3))
        c, p = TensorCSA((2, 2, 2), init='random', random_state=7).fit(x), PCA(8).fit(x)
        self.assertGreaterEqual(np.diff(c.history).min(), -1e-9)
        self.assertLessEqual(np.square(x-p.reconstruct(x)).sum(),
                             np.square(x-c.reconstruct(x)).sum()+1e-9)
        self.assertLess(c.diagnostics['energy_identity_error'], 1e-9)

    def test_full_rank_and_constant_data(self):
        x = np.random.default_rng(13).normal(size=(9, 4, 5, 3))
        c = TensorCSA((4, 5, 3)).fit(x)
        np.testing.assert_allclose(c.reconstruct(x), x, atol=1e-11)
        c = TensorCSA((2, 2)).fit(np.ones((5, 4, 4)))
        self.assertTrue(c.converged)
        self.assertEqual(c.history[-1], 0)

    def test_rejects_invalid_inputs_and_unfitted_use(self):
        for ranks in [(0, 2), (2.5, 2), (), (True, 2)]:
            with self.assertRaises(ValueError):
                TensorCSA(ranks)
        with self.assertRaises(ValueError):
            TensorCSA((2, 2), max_sweeps=0)
        c = TensorCSA((2, 2))
        with self.assertRaises(RuntimeError):
            c.transform(np.zeros((2, 4, 4)))
        for x in [np.zeros((0, 4, 4)), np.zeros((2, 1, 4)),
                  np.full((2, 4, 4), np.nan), np.ones((2, 4, 4))*1j]:
            with self.assertRaises(ValueError):
                c.fit(x)
        c.fit(np.zeros((2, 4, 4)))
        with self.assertRaises(ValueError):
            c.transform(np.zeros((2, 3, 4)))


if __name__ == '__main__':
    unittest.main()

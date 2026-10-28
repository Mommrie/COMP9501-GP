"""Behavioral checks for mathematical errors that would invalidate experiments."""
import unittest
import torch
from merge import merge_tensors
from probe import cosine_matrix


class MergeMath(unittest.TestCase):
    def test_merge_factors_is_not_merge_delta(self):
        a1, b1, a2, b2 = [torch.tensor([[x]]) for x in (1., 2., 3., 4.)]
        correct = merge_tensors([b1 @ a1, b2 @ a2])
        self.assertEqual(correct.item(), 7.)
        self.assertNotEqual(correct.item(), (((b1 + b2) / 2) @ ((a1 + a2) / 2)).item())

    def test_task_arithmetic_scaling(self):
        x, y = torch.tensor([2., -4.]), torch.tensor([6., 2.])
        torch.testing.assert_close(merge_tensors([x, y], "task_arithmetic", .5), (x + y) / 2)

    def test_ties_sign_election(self):
        x, y = torch.tensor([3., -2., 0.]), torch.tensor([-1., -4., 0.])
        torch.testing.assert_close(merge_tensors([x, y], "ties", density=1), torch.tensor([3., -3., 0.]))

    def test_dare_full_density_and_seed(self):
        x, y = torch.ones(200), torch.ones(200) * 2
        torch.testing.assert_close(merge_tensors([x, y], "dare", density=1), x + y)
        torch.testing.assert_close(merge_tensors([x, y], "dare", seed=4), merge_tensors([x, y], "dare", seed=4))

    def test_delta_reveals_opposite_signal(self):
        base = torch.full((2, 2), 100.)
        change = torch.tensor([[1., -1.], [1., -1.]])
        self.assertGreater(cosine_matrix([base + change, base - change])[0, 1], .99)
        self.assertAlmostEqual(cosine_matrix([change, -change])[0, 1].item(), -1., places=6)


if __name__ == "__main__":
    unittest.main()

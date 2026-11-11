import unittest
import numpy as np
import torch
from activation_extension import pooled,hidden_loss,derangement,best_expert


class ActivationChecks(unittest.TestCase):
    def test_mask_and_loss(self):
        h=torch.tensor([[[2.,4.],[4.,8.],[900.,900.]]],requires_grad=True)
        p=pooled(h,torch.tensor([[1,1,0]]))
        torch.testing.assert_close(p,torch.tensor([[3.,6.]]))
        target=torch.tensor([[2.,4.]])
        loss=hidden_loss(p,target,1e-4);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(h.grad.abs().sum()),0.)
        self.assertEqual(float(h.grad[0,2].abs().sum()),0.)
        self.assertEqual(float(hidden_loss(target,target,1e-4)),0.)

    def test_shuffle_and_oracle(self):
        d=derangement(128,42)
        self.assertEqual(sorted(d.tolist()),list(range(128)))
        self.assertTrue(np.all(d!=np.arange(128)))
        np.testing.assert_array_equal(d,derangement(128,42))
        logits=np.array([[[2.,1.],[1.,2.]],[[1.,2.],[2.,1.]]])
        np.testing.assert_array_equal(best_expert(logits,np.array([0,0])),[0,1])
        # Additive logit shifts must not change the minimum-NLL expert.
        shifted=logits+np.array([100.,-40.])[:,None,None]
        np.testing.assert_array_equal(best_expert(shifted,np.array([0,0])),[0,1])
        rng=np.random.default_rng(4)
        values=rng.normal(size=(4,32,2));labels=rng.integers(0,2,32)
        route=best_expert(values,labels)
        correct=values[route,np.arange(32)].argmax(-1)==labels
        self.assertTrue(np.all(correct | ~(values.argmax(-1)==labels).any(0)))


if __name__=='__main__':unittest.main()

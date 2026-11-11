import unittest
import torch
from core import Coefficients,MixedLinear


class ExactMix(unittest.TestCase):
    def test_materialized_equivalence_and_frozen_parameters(self):
        torch.manual_seed(4);base=torch.nn.Linear(5,3);factors=[(torch.randn(2,5),torch.randn(3,2)) for _ in range(4)]
        coeff=Coefficients();module=MixedLinear(base,factors,coeff,0,2.)
        x=torch.randn(7,5);weight=base.weight.detach()+sum(float(coeff.weights()[0,i]*coeff.gain())*(b@a)*2 for i,(a,b) in enumerate(factors))
        torch.testing.assert_close(module(x),torch.nn.functional.linear(x,weight,base.bias))
        module(x).square().mean().backward()
        self.assertIsNone(base.weight.grad);self.assertIsNone(base.bias.grad)
        self.assertGreater(float(coeff.logits.grad.norm()),0)
        self.assertGreater(float(coeff.gain_logit.grad.abs()),0)


if __name__=='__main__':unittest.main()

import unittest
import torch
from maths import solve,simplex


class Matching(unittest.TestCase):
    def test_common_inputs_degenerate_to_uniform(self):
        torch.manual_seed(1)
        responses=torch.randn(4,100,dtype=torch.float64)
        gram=responses@responses.T
        g=4*gram;b=gram.sum(1)
        a,_=solve(g,b)
        torch.testing.assert_close(a,torch.full((4,),.25,dtype=a.dtype),atol=1e-9,rtol=0)

    def test_quadratic_matches_responses(self):
        torch.manual_seed(2)
        h=torch.randn(4,4,30,dtype=torch.float64)
        gram=h@h.transpose(1,2)
        g=gram.sum(0);b=sum(gram[i,:,i] for i in range(4))
        a,info=solve(g,b)
        direct=sum(((a@h[i])-h[i,i]).square().sum() for i in range(4))
        constant=sum(h[i,i].square().sum() for i in range(4))
        torch.testing.assert_close(direct,a@g@a-2*b@a+constant)
        self.assertLessEqual(info['objective'],info['uniform_objective']+1e-9)

    def test_boundary(self):
        a,_=solve(torch.eye(4),torch.tensor([100.,0,0,0]))
        torch.testing.assert_close(a,torch.tensor([1.,0,0,0],dtype=a.dtype))
        self.assertAlmostEqual(float(simplex(torch.tensor([-1.,2.,0.])).sum()),1.)


if __name__=='__main__':unittest.main()

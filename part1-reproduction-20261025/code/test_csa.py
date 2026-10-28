import unittest
import numpy as np
from csa import CSA, PCA


class Tests(unittest.TestCase):
    def test_projection_and_pca_bound(self):
        x=np.random.default_rng(7).normal(size=(40,7,6))
        c=CSA(2).fit(x);p=PCA(4).fit(x)
        self.assertLessEqual(np.square(x-p.reconstruct(x)).sum(),np.square(x-c.reconstruct(x)).sum()+1e-8)
        self.assertTrue(np.allclose((x-c.mean).reshape(len(x),-1) @ np.kron(c.u,c.v), c.transform(x).reshape(len(x),-1)))

    def test_exact_separable_reconstruction(self):
        g=np.random.default_rng(3)
        u=np.linalg.qr(g.normal(size=(8,2)))[0];v=np.linalg.qr(g.normal(size=(7,2)))[0]
        x=np.einsum('ha,nab,wb->nhw',u,g.normal(size=(30,2,2)),v)+g.normal(size=(8,7))
        c=CSA(2).fit(x)
        self.assertLess(np.square(x-c.reconstruct(x)).mean(),1e-20)


if __name__=='__main__':unittest.main()

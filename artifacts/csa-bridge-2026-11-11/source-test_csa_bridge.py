import unittest
import numpy as np
from csa_bridge import TensorCSA,FlatPCA,storage,budget_pca_rank,relative_error

class BridgeTests(unittest.TestCase):
    def test_projection_and_kronecker(self):
        x=np.random.default_rng(5).normal(size=(20,3,5));m=TensorCSA((2,3)).fit(x)
        a=m.transform(x).reshape(20,-1);b=(x-m.mean).reshape(20,-1)@np.kron(*m.factors)
        np.testing.assert_allclose(a,b,atol=1e-10)
        self.assertTrue(np.all(np.diff(m.history)>=-1e-9))
    def test_full_rank_and_energy_identity(self):
        x=np.random.default_rng(6).normal(size=(20,3,5));m=TensorCSA((3,5)).fit(x)
        np.testing.assert_allclose(x,m.reconstruct(x),atol=1e-10)
        m=TensorCSA((2,3)).fit(x);z=x-m.mean
        self.assertAlmostEqual(np.square(z).sum(),np.square(m.transform(x)).sum()+np.square(x-m.reconstruct(x)).sum(),places=8)
    def test_equal_latent_training_optimum(self):
        x=np.random.default_rng(7).normal(size=(20,3,5));a=TensorCSA((2,2)).fit(x);b=FlatPCA(4).fit(x)
        self.assertLessEqual(relative_error(x,b.reconstruct(x),b.mean),relative_error(x,a.reconstruct(x),a.mean)+1e-10)
    def test_full_storage_budget(self):
        shape=(4,896);r=(2,8);n=512;s=storage(shape,r,n,'csa');d=budget_pca_rank(shape,r,n)
        self.assertEqual(s['total'],4*896+4*2+896*8+512*16)
        self.assertLessEqual(storage(shape,[1,d],n,'pca')['total'],s['total'])
        self.assertGreater(storage(shape,[1,d+1],n,'pca')['total'],s['total'])

if __name__=='__main__':unittest.main()

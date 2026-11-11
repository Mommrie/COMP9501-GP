import unittest
import numpy as np
from csa_discriminant import fit_lda,probabilities,prior_routes,gain_routes,choose

class DiscriminantTests(unittest.TestCase):
    def test_lda_matches_reference_low_and_high_dimensional(self):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        rng=np.random.default_rng(19)
        for d in [5,45]:
            x=rng.normal(size=(30,d));y=np.arange(30)%3;x[:,0]+=y
            for alpha in [.1,.5,.9]:
                ours=fit_lda(x,y,alpha);ref=LinearDiscriminantAnalysis(solver='lsqr',shrinkage=alpha).fit(x,y)
                np.testing.assert_allclose(probabilities(ours,x),ref.predict_proba(x),atol=1e-8,rtol=1e-7)
    def test_single_class_and_missing_benefit_classes(self):
        x=np.arange(12).reshape(6,2);m=fit_lda(x,np.zeros(6,int),.5)
        np.testing.assert_array_equal(probabilities(m,x),np.ones((6,1)))
    def test_own_priority_and_strict_threshold(self):
        own=np.array([0,1]);prob=np.array([[.5,.5,0,0],[.7,.3,0,0]])
        np.testing.assert_array_equal(prior_routes(prob,own,0),[0,0])
        np.testing.assert_array_equal(prior_routes(prob,own,.4),own)
        np.testing.assert_array_equal(gain_routes(np.array([[9,0,0,0],[0,9,0,0.]]),own,0),own)
    def test_always_own_wins_safe_ties(self):
        value={'native':.7,'accuracy':.8,'worst_accuracy':.6,'switch_fraction':0.}
        options=[{'alpha':.9,'threshold':.1,'metrics':value.copy()},{'alpha':1.,'threshold':None,'always_own':True,'metrics':value.copy()}]
        self.assertEqual(choose(options,value,.01,True),1)
    def test_reject_selection_harm(self):
        base={'native':.7,'accuracy':.8,'worst_accuracy':.6,'switch_fraction':0.}
        bad={**base,'accuracy':.79,'native':.71,'switch_fraction':.1}
        options=[{'alpha':.1,'threshold':0,'metrics':bad},{'alpha':1.,'threshold':None,'always_own':True,'metrics':base}]
        self.assertEqual(choose(options,base,.01,True),1)

if __name__=='__main__':unittest.main()

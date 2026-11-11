import unittest
import numpy as np
from robustness_controls import benefit_routes,confidence_routes,choose_safe,clean_indices,stored_probabilities,select_lambdas,predict_locked_gate
from independent_protocol import content_key


class RobustnessChecks(unittest.TestCase):
    def test_saved_fallback_bypasses_feature_extraction(self):
        lock={'confidence_selected':{'threshold':None},'benefit_selected':{'threshold':None}}
        for kind in ['confidence_gate','benefit_gate']:
            np.testing.assert_array_equal(predict_locked_gate(lock,None,[0,2,3],kind),[0,2,3])

    def test_lambda_selection_uses_selection_scores_and_prefers_zero_on_tie(self):
        chosen={'score':.7,'weights':[[.25]*4],'gain':2.,'step':0}
        old={'fits':{n:{'selected':chosen.copy()} for n in ['layer_ce_fp32','layer_ce_hidden','layer_ce_shuffled']},'final_static':{'deliberately_irrelevant':100.}}
        new={'fits':{n:{'selected':chosen.copy()} for n in ['ce_h001','ce_s001','ce_h010','ce_s010']},'final_static':{'deliberately_irrelevant':-100.}}
        selected,_=select_lambdas(old,new)
        self.assertEqual(selected['hidden']['lambda'],0.)
        self.assertEqual(selected['shuffled']['lambda'],0.)

    def test_fallback_never_uses_own_predicted_gain(self):
        gain=np.array([[100.,.2,.1,-1.],[.1,100.,.3,-1.]])
        own=np.array([0,1])
        np.testing.assert_array_equal(benefit_routes(gain,own,.25),[0,2])
        np.testing.assert_array_equal(benefit_routes(gain,own,None),own)
        np.testing.assert_array_equal(confidence_routes(np.array([2,3]),np.array([.4,.9]),own,.8),[0,3])

    def test_selection_rejects_harm_and_prefers_fallback_on_tie(self):
        ref={'score':.8,'mean_accuracy':.9,'worst_accuracy':.8,'switch_fraction':0.}
        bad={**ref,'score':.95,'mean_accuracy':.89}
        tie={**ref,'switch_fraction':.3}
        c=[{'evaluation':ref},{'evaluation':bad},{'evaluation':tie}]
        self.assertEqual(choose_safe(c,ref,.01),0)
        self.assertFalse(c[1]['selection_eligible'])

    def test_content_filtering_does_not_read_labels(self):
        rows=[{'sentence':' A  B '},{'sentence':'a b'},{'sentence':'kept'},{'sentence':'KEPT'}]
        keep,keys,overlap,dup=clean_indices(rows,['sentence'],{content_key(['a b'])})
        self.assertEqual(keep,[2]);self.assertEqual((overlap,dup),(2,1))

    def test_stored_binary_and_multiclass_probabilities(self):
        r={'classes':[1,3],'coef':[[1.,0.]],'intercept':[0.]}
        p,c=stored_probabilities(r,np.array([[0.,0.],[100.,2.]]))
        np.testing.assert_allclose(p[0],[.5,.5]);self.assertEqual(c[p[1].argmax()],3)
        r={'classes':[0,1,2],'coef':np.eye(3).tolist(),'intercept':[0.,0.,0.]}
        p,c=stored_probabilities(r,np.array([[1.,2.,3.]]));self.assertEqual(c[p.argmax()],2)
        np.testing.assert_allclose(p.sum(1),1.)


if __name__=='__main__':unittest.main()

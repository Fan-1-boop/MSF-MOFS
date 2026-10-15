"""Research implementation contract tests (not full statistical replication)."""
import unittest
import numpy as np
from irmofs.stable_guidance import fisher_scores, make_cisr_multiscale_population
from irmofs.highdim_formal import make_base_population
from irmofs.nsga2 import hypervolume_2d
from irmofs.environments import make_nested_ir_environments
from irmofs.evaluator import FeatureSubsetEvaluator


class FrozenMSFTests(unittest.TestCase):
    def test_exactly_three_guides_in_population_12_and_50(self):
        for n in [12,50]:
            for d,expect in [(100,[5,10,20]), (166,[8,10,17]),
                             (500,[10,25,50]),(10000,[10,500,1000])]:
                with self.subTest(n=n,d=d):
                    base=make_base_population(n,d,92,0.20)
                    scores=np.arange(d,0,-1,dtype=float)
                    actual,sizes=make_cisr_multiscale_population(base,scores,3/n)
                    self.assertEqual(sizes,expect)
                    np.testing.assert_array_equal(actual[3:],base[3:])
                    self.assertEqual([int(z.sum()) for z in actual[:3]],expect)
                    for k in range(2):
                        self.assertTrue(set(np.flatnonzero(actual[k])).issubset(set(np.flatnonzero(actual[k+1]))))

    def test_fisher_training_only_and_label_contract(self):
        x=np.array([[0.,1.],[1.,1.],[2.,2.],[3.,2.]])
        y=np.array([0,0,1,1])
        got=fisher_scores(x,y)
        self.assertTrue(np.all(got>=0))
        self.assertAlmostEqual(got[0],8.,places=8)
        self.assertGreater(got[1],1e11)
        with self.assertRaises(ValueError): fisher_scores(x,np.array([0,1,2,3]))

    def test_hv_exact_minimization_and_dominance(self):
        F=np.array([[.2,.4],[.5,.1]])
        self.assertAlmostEqual(hypervolume_2d(F),.63,places=13)
        self.assertAlmostEqual(hypervolume_2d(np.vstack([F,[.8,.8]])),.63,places=13)
        self.assertEqual(hypervolume_2d(np.array([[1.1,1.1]])),0.)

    def test_fold_partition_and_evaluator_cache(self):
        rng=np.random.default_rng(4)
        X=rng.standard_normal((60,8))
        y=np.array([0]*30+[1]*30)
        envs=make_nested_ir_environments(X,y,irs=(1,3,5),n_splits=3,random_state=4)
        for env in envs.values():
            for tr,va in env.folds:
                self.assertEqual(len(set(tr).intersection(va)),0)
                self.assertEqual(len(tr)+len(va),len(env.y))
        evaluator=FeatureSubsetEvaluator(envs,mode='mean',n_neighbors=5,backend='sklearn')
        bits=np.ones(8,dtype=np.uint8)
        result=evaluator.evaluate(bits)
        np.testing.assert_array_equal(evaluator.evaluate(bits),result)
        self.assertEqual(len(evaluator.cache),1)
        self.assertTrue((result>=0).all() and (result<=1).all())


if __name__=='__main__': unittest.main()

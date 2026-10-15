"""Validation tests use synthetic data and never claim scientific results."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from msf_repro.validation_v3 import validate_rows, holm, rb, pair_stats

class V3ValidationTests(unittest.TestCase):
    def test_complete_reference_match(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td); p=d/'reproduced.csv';ref=d/'archived.csv'
            df=pd.DataFrame([{'dataset':'X','seed':92,'method':'MSF','test_pf_hv':0.9},
                             {'dataset':'X','seed':93,'method':'MSF','test_pf_hv':0.8}])
            df.to_csv(p,index=False);df.to_csv(ref,index=False)
            out,_=validate_rows(p,['dataset','seed','method'],{('X',92,'MSF'),('X',93,'MSF')},ref,['test_pf_hv'])
            self.assertTrue(out['complete']);self.assertEqual(out['reference_status'],'MATCH')
    def test_incomplete_disallowed(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.csv';pd.DataFrame([{'seed':92,'score':1}]).to_csv(p,index=False)
            out,_=validate_rows(p,['seed'],{(92,),(93,)})
            self.assertFalse(out['complete']);self.assertEqual(out['missing'],1)
    def test_duplicate_rows_disallowed(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.csv';pd.DataFrame([{'seed':92},{'seed':92}]).to_csv(p,index=False)
            out,_=validate_rows(p,['seed'],{(92,)})
            self.assertFalse(out['complete']);self.assertEqual(out['duplicates'],1)
    def test_numerical_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.csv';ref=Path(td)/'r.csv'
            pd.DataFrame([{'seed':92,'hv':0.2}]).to_csv(p,index=False)
            pd.DataFrame([{'seed':92,'hv':0.3}]).to_csv(ref,index=False)
            out,_=validate_rows(p,['seed'],{(92,)},ref,['hv'])
            self.assertEqual(out['reference_status'],'MISMATCH')
    def test_four_member_holm(self):
        self.assertTrue(np.allclose(holm([.001,.02,.04,.07]),[.004,.06,.08,.08]))
    def test_rb_range(self):
        self.assertEqual(rb([1,2,3]),1.0)
        self.assertEqual(rb([-1,-2,-3]),-1.0)
        self.assertEqual(rb([0,0]),0.)
    def test_pair_stats_requires_all_ten_seeds(self):
        a=pd.DataFrame({'seed':[92,93],'test_pf_hv':[0.6,0.7]})
        b=pd.DataFrame({'seed':[92,93],'test_pf_hv':[0.4,0.5]})
        with self.assertRaises(ValueError):pair_stats(a,b,'dummy','A-B')

if __name__=='__main__':unittest.main()

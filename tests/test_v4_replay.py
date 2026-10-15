"""Unit/regression tests for the independent v4 reliability evidence chain."""
from pathlib import Path
import tempfile
import unittest
import pandas as pd
import numpy as np

from msf_repro.fisher_recheck import build_pairwise
from msf_repro.validation_v4 import reliability_from_reruns

ROOT=Path(__file__).resolve().parents[1]

class V4EvidenceTests(unittest.TestCase):
    def test_fisher_recompute_has_complete_pairwise_family(self):
        df=build_pairwise(ROOT/'data/public/core')
        self.assertEqual(len(df),450)
        for ds in ('madelon','arcene'):
            q=df[df.dataset==ds]
            self.assertEqual(len(q),225)
            self.assertEqual((q.metric=='spearman_rank').sum(),45)
            self.assertEqual((q.metric=='topk_jaccard').sum(),180)
            self.assertTrue(q.value.between(-1,1).all())
            self.assertEqual(q[['seed_a','seed_b']].drop_duplicates().shape[0],45)

    def test_empty_reliability_evidence_must_not_be_complete(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); (p/'results/reference').mkdir(parents=True)
            report=reliability_from_reruns(p,p/'results/reference')
            self.assertEqual(report['reference_status'],'PENDING')
            self.assertFalse(report['complete'])
            self.assertEqual(report['new_search_rows'],0)

    def test_partial_reliability_rows_must_not_produce_paper_association(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            folder=p/'results/reference/reliability';folder.mkdir(parents=True)
            (p/'results/reference/fisher_stability').mkdir(parents=True)
            pd.DataFrame([{'dataset':'madelon','seed':72,'method':'MSF-MOFS','train_hv_replayed':.3,'test_pf_hv_replayed':.2}]).to_csv(folder/'replayed_runs.csv',index=False)
            pd.DataFrame([{'dataset':'madelon','seed_a':72,'seed_b':73,'metric':'topk_jaccard','k':10,'value':.5}]).to_csv(p/'results/reference/fisher_stability/pairwise.csv',index=False)
            report=reliability_from_reruns(p,p/'results/reference')
            self.assertEqual(report['new_search_rows'],1)
            self.assertFalse(report['complete'])
            self.assertFalse((p/'results/reference/RELIABILITY_PAPER_COMPARISON.csv').exists())

if __name__=='__main__':unittest.main()

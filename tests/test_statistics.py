import json
"""Independent statistics contracts: coverage, direction, Holm family, ref-point expansion."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from msf_repro.statistics import (CORE, BREADTH, verify_input, analyze_files,
                                  holm, rank_biserial_paired, build_tables)
from msf_repro.reliability import reaggregate_reliability

ROOT = Path(__file__).resolve().parents[1]


class StatisticsTests(unittest.TestCase):
    def test_holm_and_signed_rank_rbc(self):
        np.testing.assert_allclose(holm([.01,.04,.02]),[.03,.04,.04])
        self.assertAlmostEqual(rank_biserial_paired([1,2,-3]), 0.0)
        self.assertAlmostEqual(rank_biserial_paired([1,2,3]), 1.0)
        self.assertAlmostEqual(rank_biserial_paired([-1,-2,-3]), -1.0)

    def test_strict_coverage_and_deduplication(self):
        x=pd.read_csv(ROOT/'evidence/public/untouched_breadth/raw.csv')
        self.assertEqual(len(verify_input(x,expected_datasets=BREADTH)),200)
        with self.assertRaisesRegex(ValueError,'Missing experiment cells'):
            verify_input(x.iloc[:-1],expected_datasets=BREADTH)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            verify_input(pd.concat([x,x.iloc[:1]]),expected_datasets=BREADTH)
        with self.assertRaisesRegex(ValueError,'Dataset coverage mismatch'):
            verify_input(x,expected_datasets=CORE+BREADTH)

    def test_reaggregate_breadth_archive_and_hv_references(self):
        raw=ROOT/'evidence/public/untouched_breadth/raw.csv'
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            result=analyze_files([raw],out,scope='breadth',evidence_type='archived_per_seed')
            self.assertEqual(result['raw_rows'],200)
            self.assertEqual(result['groups'],10)
            ci=pd.read_csv(out/'paired_hv_ci_effect.csv')
            sensitivity=pd.read_csv(out/'hv_reference_sensitivity.csv')
            archived=pd.read_csv(ROOT/'evidence/revision/statistics_existing_results/paired_hv_ci_effect_9dataset.csv')
            merged=ci.merge(archived,on=['dataset','carrier'],suffixes=('_new','_old'))
            self.assertEqual(len(merged),10)
            np.testing.assert_allclose(merged.mean_delta_new,merged.mean_delta_old,atol=1e-12)
            np.testing.assert_allclose(merged.p_raw_new,merged.p_raw_old,atol=1e-12)
            self.assertEqual(len(sensitivity),50)
            base=sensitivity[sensitivity.ref=='(1.00,1.00)'].set_index(['dataset','carrier'])
            ciindex=ci.set_index(['dataset','carrier'])
            np.testing.assert_allclose(base.loc[ciindex.index,'mean_delta'],ciindex.mean_delta,atol=1e-12)
            # Nine-dataset adjusted p-values MUST NOT be copied to the five-dataset family.
            self.assertAlmostEqual(float(ci.p_holm_within_carrier.iloc[0]), 0.009765625)
            self.assertEqual(result['evidence_type'],'archived_per_seed')

    def test_reliability_source_to_archived_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=reaggregate_reliability(ROOT,Path(tmp))
            self.assertEqual(result['status'],'MATCH_ARCHIVED_REAGGREGATION')
            self.assertEqual(result['seed_rows'],20)
            self.assertAlmostEqual(result['dataset_centered_spearman']['delta_heldout']['rho'],0.021052631578947364)


if __name__ == '__main__':
    unittest.main()

class MuskPortabilityTests(unittest.TestCase):
    def test_derived_musk_exact_hashes_and_loader(self):
        from msf_repro.audit import _musk_compat
        from irmofs.highdim_datasets import load_highdim_dataset
        self.assertEqual(_musk_compat(ROOT)['status'],'PASS')
        data=load_highdim_dataset('musk1',ROOT/'data/public/core')
        self.assertEqual(data.X.shape,(476,166))
        self.assertEqual(int(data.y.sum()),207)

class OutputMergeTests(unittest.TestCase):
    def test_shard_merge_preserves_equality_and_rejects_conflicts(self):
        from msf_repro.merge import merge_core_inputs
        raw=pd.read_csv(ROOT/'results/reference/table3/core/raw.csv')
        item=raw.iloc[:2].copy()
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            a=tmp/'a.csv'; b=tmp/'b.csv';target=tmp/'merged.csv'
            item.iloc[:1].to_csv(a,index=False)
            item.to_csv(b,index=False)
            result=merge_core_inputs([a,b],target)
            self.assertEqual(result['n_final_rows'],2)
            self.assertEqual(result['deduplicated_rows'],1)
            altered=item.copy()
            altered.iloc[0,altered.columns.get_loc('test_pf_hv')]+=0.01
            altered.to_csv(b,index=False)
            with self.assertRaisesRegex(ValueError,'Conflicting'):
                merge_core_inputs([a,b],target)
            self.assertEqual(len(pd.read_csv(target)),2)
            manifest=json.loads((target.parent/'merge_manifest.json').read_text(encoding='utf-8'))
            self.assertNotIn('/mnt/data/', json.dumps(manifest))
            self.assertIsNone(manifest['target']['path'])  # 临时目录在项目外
            self.assertEqual(manifest['target']['path_kind'], 'external_file_not_packaged')

class PortableResultProvenanceTests(unittest.TestCase):

    def test_paper_reference_must_not_claim_new_run(self):
        from msf_repro.statistics import analyze_files
        from pathlib import Path
        with tempfile.TemporaryDirectory() as target:
            with self.assertRaisesRegex(ValueError, 'Archived CSV'):
                analyze_files([ROOT/'results/reference/table3/core/raw.csv',
                               ROOT/'results/reference/table3/breadth/raw.csv'],
                              Path(target), scope='nine', evidence_type='reproduced', bootstrap_draws=10)

class RerunRegressionTests(unittest.TestCase):
    def test_core_40_available_per_seed_references(self):
        from msf_repro.replay import compare_core_msf_nsga_with_archive
        report=compare_core_msf_nsga_with_archive(ROOT,ROOT/'results/reference/table3/core/raw.csv')
        self.assertEqual(report['status'],'MATCH')
        self.assertEqual(report['compared_rows'],40)

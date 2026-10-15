"""卫星论文结论统一入口的回归测试。

只验证统计与入口的可靠性，不把测试直接当成新一轮四子系统搜索。
"""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/satellite/paper_claims_check.py'
spec = importlib.util.spec_from_file_location('satellite_paper_claims_check', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SatellitePaperClaimsTests(unittest.TestCase):
    def test_archived_table4_recalculation(self):
        """真实逐种子数据的四项 ΔHV、Holm p 应可稳定重算。"""
        x = module.check_archived_table4()
        self.assertTrue(x['allPass'], x['errors'])
        self.assertEqual(x['sourceRecords'], 40)
        self.assertTrue(x['all40Wins'])
        self.assertEqual(len(x['rows']), 4)
        self.assertTrue(all(abs(r['holmP'] - .0078125) < 1e-12 for r in x['rows']))

    def test_rejects_modified_or_duplicate_seeds(self):
        """历史 CSV 缺失或重复一个种子时，不能继续宣称 40/40 覆盖。"""
        with tempfile.TemporaryDirectory() as t:
            temp = Path(t)
            for subsystem in module.PUBLISHED:
                source = module.DATA/subsystem/'per_seed_results.csv'
                target = temp/subsystem/'per_seed_results.csv'
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
            power = temp/'power/per_seed_results.csv'
            s = power.read_text(encoding='utf8').splitlines()
            s[-1] = s[-2]  # 故意制造重复种子
            power.write_text('\n'.join(s)+'\n', encoding='utf8')
            with patch.object(module, 'DATA', temp):
                x = module.check_archived_table4()
            self.assertFalse(x['allPass'])
            self.assertTrue(x['errors'])

    def test_frozen_model_bytes(self):
        """从真实模型文件计算 SHA-256，不使用报告里的固定字符串假验证。"""
        self.assertTrue(module.check_frozen_model_hash()['pass'])

    def test_v8_manifest_is_only_metadata(self):
        """411场景、9042窗口只能视为清单信息核查。"""
        x = module.check_v8_dataset_manifest()
        self.assertTrue(x['pass'])
        self.assertIn('非独立重算', x['evidenceClass'])


if __name__ == '__main__':
    unittest.main()

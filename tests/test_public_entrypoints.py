"""保证论文公开实验入口在目录重组后仍能定位真实运行脚本。

只检查路径与默认输出隔离，不执行耗时的完整优化搜索。
"""
from pathlib import Path
import unittest

from msf_repro import cli


class TestPublicEntrypoints(unittest.TestCase):
    def test_public_search_scripts_exist(self):
        """各论文实验的正式 Python 运行器必须实际存在。"""
        names = (
            'run_v2_spea2_plugin.py', 'run_v2_untouched_breadth.py',
            'run_frozen_confirmation.py', 'sparse_control_portable.py',
            'run_breadth5_sparse_control_nsga2.py',
            'population50_madelon_portable.py', 'population50_extension.py',
            'replay_reliability_portable.py',
        )
        for name in names:
            with self.subTest(name=name):
                self.assertTrue((cli.ROOT / 'scripts/public' / name).is_file())

    def test_new_outputs_are_separated_from_archive(self):
        """新搜索与存档目录必须不同，以免意外覆盖历史证据。"""
        self.assertEqual(cli.OUT, cli.ROOT / 'results/reference')
        self.assertEqual(cli.LOCAL, cli.ROOT / 'results/reproduced/paper')
        self.assertNotEqual(cli.OUT, cli.LOCAL)

    def test_public_data_present(self):
        """公开数据根目录至少应包含核心及扩展数据集。"""
        self.assertTrue(cli.CORE.is_dir())
        self.assertTrue(cli.BREADTH.is_dir())

    def test_satellite_v4_compatibility_files_present(self):
        """保留旧 CLI 所引用的 V4 原始脚本，避免兼容命令失效。"""
        base = cli.ROOT / 'scripts/satellite/v4'
        for name in ('msf_core.py', 'run_msf_vs_random.py', 'analyze_msf_vs_random.py'):
            with self.subTest(name=name):
                self.assertTrue((base / name).is_file())


if __name__ == '__main__':
    unittest.main()

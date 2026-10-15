"""外部基线的跨平台互信息回归测试。

Windows 不存在 fork；测试通过模拟 spawn-only 环境强制运行线程回退分支。
这些测试验证互信息数值不因并行后端变化而改变。
"""
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from irmofs import external_baselines as baseline


class TestWindowsCompatibleMutualInformation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1234)
        self.X = rng.normal(size=(48, 9))
        self.y = np.repeat([0, 1], 24)
        # 模拟不同尺度、不同信息量的连续特征。
        self.X[self.y == 1, 0] += 2.0
        self.X[self.y == 1, 4] -= 1.5

    def test_spawn_only_windows_branch_equals_sklearn(self):
        # Windows 下禁止触发 ProcessPoolExecutor，也不能查询 fork 上下文。
        with (patch.object(baseline.mp, 'get_all_start_methods', return_value=['spawn']),
              patch.object(baseline, 'ProcessPoolExecutor', side_effect=AssertionError('不可使用 fork 进程池'))):
            actual = baseline._mutual_info_classif_parallel_dense(self.X, self.y, 92)
        expected = mutual_info_classif(self.X, self.y, random_state=92, n_neighbors=3)
        np.testing.assert_array_equal(actual, expected)
        self.assertIsNone(baseline._MI_X)
        self.assertIsNone(baseline._MI_Y)

    def test_thread_backend_reproducible_and_non_destructive(self):
        original = self.X.copy()
        with patch.object(baseline.mp, 'get_all_start_methods', return_value=['spawn']):
            a = baseline._mutual_info_classif_parallel_dense(self.X, self.y, 101)
            b = baseline._mutual_info_classif_parallel_dense(self.X, self.y, 101)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(self.X, original)

    def test_fork_backend_matches_thread_when_available(self):
        if 'fork' not in baseline.mp.get_all_start_methods():
            self.skipTest('本平台没有 fork，已执行 Windows 回退分支测试')
        fork_scores = baseline._mutual_info_classif_parallel_dense(self.X, self.y, 92)
        with patch.object(baseline.mp, 'get_all_start_methods', return_value=['spawn']):
            thread_scores = baseline._mutual_info_classif_parallel_dense(self.X, self.y, 92)
        np.testing.assert_array_equal(fork_scores, thread_scores)


if __name__ == '__main__':
    unittest.main()

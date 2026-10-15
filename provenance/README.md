# 源码与证据可追溯性

- `core_python_original_sha256.json`：首次打包时历史 v4 核心 Python 的 SHA-256。
- `core_algorithm_ast_equivalence.json`：v1.2 的历史中文注释版 AST 对照；v1.2.1 对 `external_baselines.py` 做了 Windows 兼容性修改，因此当前仅 10 个核心文件仍与 v4 AST 相同，修改细节见 `docs/13_Windows外部基线兼容性修复.md`。
- `V8_FROZEN-SOURCE-SHA256.txt`：历史V8冻结来源摘要，注意2个确认脚本来源差异。
- `v4_source_integrity_original.json`：旧项目声明的哈希，其中2项不匹配，不能视作全部通过。
- `FILE_MANIFEST.csv`、`SHA256SUMS.txt`：最终交付包内逐文件大小、内容哈希，排除自身以避免哈希循环。
- `v8_reference_scripts/`：保留原始两个V8确认程序用于核对；可运行路径适配版位于 `scripts/satellite/`。

`SHA256SUMS.txt` 使用标准 `sha256sum -c provenance/SHA256SUMS.txt` 校验（从根目录执行）。`results/reproduced/` 不属于冻结归档，不进入来源校验。

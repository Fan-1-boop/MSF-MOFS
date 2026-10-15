# V6 卫星部署源码导读

这一层源码来源于历史 V6 原始归档。为了保留冻结模型的 SHA-256 和 `require()` 的相对导入关系，**原始 JavaScript 源文件不直接翻译注释、不更改函数及变量**；对应说明在此文档与 `docs/06_冻结140特征与部署说明.md` 中用中文解释。

- `health-v2-service/core/health-features.js`：582维候选空间、短/中/长因果窗口、全量与按需抽取。
- `health-v2-service/core/health-algorithm.js`：读取冻结模型，将子系统证据汇合为状态化健康评分。
- `health-v2-service/core/health-algorithm-v5.js`：被 V6 兼容层直接引用的历史 V5 基础健康逻辑，并非无用旧文件。
- `health-v2-service/core/models/*.json`：历史冻结模型与其基础依赖，原始文件未经修改。
- `satellite-datastream-service/telemetry-simulator.js`：带固定随机种子的因果遥测流。
- `satellite-datastream-service/generate-training-data-v6.js`：354场景/7080窗口开发数据生成器。
- `msf-training/`：历史开发搜索、最终模型拟合和基准性能脚本；`experiments/v6_final/*/experiment_results.json` 是训练模型选择所需的原始多种子证据，而不是重复版本。

运行统一入口在项目根目录 `run.py`。

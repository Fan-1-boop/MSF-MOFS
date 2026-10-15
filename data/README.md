# 复现输入数据

- `public/core`：论文核心公开数据集；`public/breadth`：其余公开数据集。原始下载文件和 `public/DATASET_MANIFEST.csv` 一并保存。
- `satellite/v6_development/v6_development_regenerated.csv`：使用历史 V6 数据生成器重新生成的 354 场景/7080 窗口，**不是原始历史CSV的字节恢复**；用于重新计算开发/模型训练，不准使用V8反选特征。
- `satellite/v8_confirmation/`：历史冻结的独立 V8 确认材料（411 场景/9042 窗口）以及 V4四子系统对应文件；主压缩 CSV 有独立 manifest。

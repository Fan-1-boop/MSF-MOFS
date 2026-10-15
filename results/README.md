# 实验结果目录（从这里开始）

本目录只区分两类：**论文已存档参考结果**和**你本机刚运行的结果**。所有路径均相对项目根目录。

- `reference/`：论文历史记录（只读），由历史逐次结果迁入，不能将其当作本机重新运行结果。
- `reproduced/`：运行 `python run.py paper ...` 或卫星验证时生成。可删除后重跑，不影响原始参考记录。

## 参考结果怎样找

| 论文实验 | 位置 | 内容 |
|---|---|---|
| Table 2 | `reference/table2_external/raw.csv` | 120 条外部对照原始结果 |
| Table 3 | `reference/table3/core/raw.csv` 与 `reference/table3/breadth/raw.csv` | 160 + 200 条九数据集结果 |
| Sparse-control | `reference/sparse_control/raw.csv` | 50 条消融记录 |
| Population=50 | `reference/population50/{madelon,extension}/raw.csv` | 60 条记录 |
| Fisher 稳定性 | `reference/fisher_stability/pairwise.csv` | 配对排名稳定性 |
| 可靠性 | `reference/reliability/` | 重放记录和关联统计 |
| Table 4 卫星 | `reference/table4_satellite/{power,thermal,attitude,data}/per_seed_results.csv` | 40 组配对记录 |
| V6/部署 | `reference/satellite_v6/`、`reference/deployment/` | 开发阶段和原版冻结计时证据 |

`evidence/` 存放另一些**不同来源的原始证据及冻结哈希**，用于交叉核验，不是这里参考 CSV 的替代品。来源清单见 `provenance/`。

## 本机重新运行

```powershell
python run.py paper external
python run.py paper core
python run.py paper breadth
python run.py paper stats --scope nine --input results/reproduced/public/public_core/raw.csv results/reproduced/public/public_breadth/raw.csv --out results/reproduced/public/table3_new --evidence-type reproduced
python run.py satellite verify
```

注意：**不要把 `reference/` 下的数据当作新跑出的实验结果**。

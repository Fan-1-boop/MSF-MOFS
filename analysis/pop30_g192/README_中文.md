# MSF-MOFS Pop=30，192 代：完整续跑项目

本实验是新增的**预算与种群敏感性分析**。原论文 Table 2 的 `population=12, generations=6` 没有修改。本项目运行三种方法：MSF-MOFS、HFMOEA-CP、MBLPE-CP；四个核心数据集：Hill-Valley、MUSK1、Arcene、MADELON；每个数据集固定种子 `92–101`；`population=30`；真正的演化轮次 `0–192`（初始化记为 0）。

## 运行方法（Windows）

1. 解压主项目 `MSF_MOFS_Reproducibility_Paper_Clean_CN_v1_2_2.zip`，正确安装依赖。
2. 将本压缩包内容放入主项目 `analysis/pop30_g192/` 目录（保留 `by_seed/` 内已完成记录）。
3. 在**主项目根目录**的 PowerShell 执行：

```powershell
.\analysis\pop30_g192\run_all_windows.ps1
```

可中断后重新执行，已经生成并通过检查的三算法配对组会跳过。如果某组只完成其中一两个算法，会继续剩余算法，不会覆盖已完成的结果。

也可在项目根目录直接运行 Python，Linux/macOS 可设置 `--workers 2`；推荐 Windows 设置 `--workers 1`，避免高维 MI 计算消耗过多内存：

```powershell
python analysis/pop30_g192/scripts/run_missing.py --project-root . --out analysis/pop30_g192 --workers 1
python analysis/pop30_g192/scripts/verify_and_plot.py --out analysis/pop30_g192
```

## 输出与核验

- `progress.json`：已完成/未完成配对组列表，含每份原始 CSV 的 SHA-256。
- `by_seed/<dataset>/<seed>/per_generation_hv.csv`：193 个演化状态 × 三方法的原始结果。
- `by_seed/<dataset>/<seed>/console.log`：原始搜索控制台输出、异常详情。
- `analysis/verification.json`：独立合并核验，汇总平台判据与未完成数据。
- `analysis/per_generation_hv.csv`：全数据集合并的原始逐代数据。
- `analysis/paired_comparison_g192.csv`：同一种子下各算法末代 HV 比较。
- `figure/Figure_pop30_g192_pilot.pdf`：三算法逐代测试 HV 图，**只有 40/40 完成后才可视为四数据集十种子的完整图**。

如果不到 40 组，默认 `verify_and_plot.py` 会拒绝生成正式图。可以指定 `--allow-partial` 绘制已完成种子，此时图仅用于探索，不能用于完整结论。

## 关于“收敛”和统计结论

程序记录每一代当前训练种群的 Pareto 前沿测试 HV，不将测试集信息反馈到搜索中。末端训练 HV 的历史最好值，在最后 24 轮内提升不超过 0.005 时，被标记为“经验平台”；它不等同于严格数学收敛，且测试 HV 可能波动。不同算法每代的不同子集评价次数不同，不能把代数相等误写成计算成本相等。MBLPE 的 Fisher 预筛选维数会随种群规模改变（`T=90`）。

只有在全部 40 个数据集-种子配对组、120 个算法搜索完成后，才能进行论文四数据集十种子的正式比较。无论哪个基线获胜，都完整保留原始结果。

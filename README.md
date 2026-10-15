# MSF-MOFS 会议论文复现项目

本项目以 **MSF-MOFS 多目标特征选择算法**为中心，包括论文公开数据实验与卫星健康度应用验证。所有正式命令从项目根目录（`run.py` 所在位置）执行。项目无需依赖旧 Linux 路径 `/mnt/data/...`。

## 1. 安装与快速核验

Windows PowerShell 示例（已进入项目目录）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py integrity
python run.py verify
```

Node.js 仅用于卫星特征提取和部署，纯公开数据实验只需 Python。建议先运行一条真正的单种子搜索：

```powershell
python run.py paper demo --dataset hill_valley --seed 92
```

这会执行单种子搜索，**不是**读取历史结果。新结果位于 `results/reproduced/paper/demo/`。

## 2. 完整论文公开数据实验

| 论文实验 | 正式运行命令 | 预设种子 | 预期逐次记录 |
|---|---|---|---:|
| Table 2：外部基线 | `python run.py paper external` | 92–101 | 120 |
| Table 3：4 个核心数据集 | `python run.py paper core` | 92–101 | 160 |
| Table 3：5 个扩展数据集 | `python run.py paper breadth` | 92–101 | 200 |
| Sparse-control | `python run.py paper sparse` | 92–101 | 50 |
| Population=50 | `python run.py paper population50` | 72–81 | 60 |
| Fisher 排序稳定性 | `python run.py paper fisher-check` | 由历史方案确定 | 统计结果 |
| 可靠性重放搜索 | `python run.py paper replay-reliability` | 72–81 | 40 |

新增的analysis目录做了population=30，generation=192的实验，证明当所有算法收敛的时候，MSF仍然保持着平均优势，但随着generation增加，优势慢慢缩小

默认种子由冻结实验协议确定，**无需手工逐个输入**。82–91 是开发阶段的种子组，不要和正式确认组混淆。详细协议见 `configs/frozen_protocol.json` 和 `docs/03_公开数据复现.md`。

完成 Table 3 后，**只用本机刚生成的两个 CSV** 重新生成论文统计：

```powershell
python run.py paper stats --scope nine --input results/reproduced/paper/table3/core/raw.csv results/reproduced/paper/table3/breadth/raw.csv --out results/reproduced/paper/table3/statistics --evidence-type reproduced
```

## 3. 卫星应用验证（论文 Table 4 和冻结部署）

```powershell
# 使用历史 40 条逐种子数据复算 Table 4；再验证 140/582 特征、在线服务和计时
python run.py satellite verify

# 真正重新执行四子系统 10 种子 MSF/Random 配对搜索（较慢）
python run.py satellite rerun-table4

# 可选：附加 V8 独立确认仿真（耗时较长，不视为原版脚本哈希一致）
python run.py satellite verify --with-v8
```

历史耗时下降 59.8% 是原实验环境的结果；当前 CPU、Node.js、系统负载会造成数值差异。正确性重点是原始冻结模型、140 个特征映射和按需提取与全量提取的逐值一致性，详见 `docs/06_计时与环境说明.md`。

## 4. 从哪里看结果

```text
results/
├── README.md               # 结果查找指南
├── reference/              # 随包交付的论文参考结果（只读）
│   ├── table2_external/    # 120 条
│   ├── table3/            # core 160 + breadth 200
│   ├── sparse_control/    # 50 条
│   ├── population50/      # 60 条
│   ├── fisher_stability/  # 排名稳定性
│   ├── reliability/       # 可靠性
│   ├── table4_satellite/  # 卫星 40 组配对
│   ├── satellite_v6/      # V6 开发实验
│   └── deployment/        # 原版140特征计时记录
└── reproduced/             # 本机新跑结果（可清理、可重跑）
```

**参考结果 ≠ 本机重跑结果。** 两个目录不互相覆盖；论文统计以 `reproduced/` 中新生成的 CSV 为准。`evidence/` 存放历史来源及外部原始证据，`provenance/` 存放文件 SHA-256 和原模型追溯证明。来源 JSON 中的旧绝对路径不是运行依赖；详细说明见 `docs/05_结果证据与校验.md`。

## 5. 代码从哪里读

| 目录 | 用途 |
|---|---|
| `src/irmofs/` | MSF-MOFS Fisher 多尺度引导、NSGA-II、SPEA2 及外部对照方法 |
| `src/msf_repro/` | 论文统计、原始记录审计、公开数据命令行 |
| `src/satellite/` | V6 原版 JavaScript 冻结模型与特征提取器（保持字节不变） |
| `scripts/public/` | 各公开数据搜索脚本 |
| `scripts/satellite/` | 卫星验证、计时与联调脚本 |
| `configs/` | 原版 140 特征和冻结实验协议 |
| `data/` | 公开数据与卫星开发/确认数据 |
| `tests/` | 自动化回归测试 |
| `docs/` | 简明中文复现教程 |

**执行范围声明：** 随包参考结果已通过历史记录审计，但这不等于本机所有种子已从零重跑。完整搜索需要你按上表逐项执行。本项目保留历史模型来源及缺失证据边界，不人为调整计时数据以匹配论文百分比。

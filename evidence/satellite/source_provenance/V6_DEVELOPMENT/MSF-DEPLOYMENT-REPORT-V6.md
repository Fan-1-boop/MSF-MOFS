# MSF 卫星健康算法 V6：子系统局部 MSF + 耦合故障最终报告

## 1. 本阶段结论

V6 不再采用“整星 582 维统一 Fisher/MSF”作为最终部署方式，而采用**子系统局部特征空间**。原因是不同卫星子系统的异常机制差异较大，全局单一 Fisher 排名会稀释局部相关性；局部 MSF 保留原 MSF-MOFS 搜索机制，只改变其应用边界。

最终策略：

- 电源、热控、姿轨控、数据处理：使用原版 MSF-MOFS 做局部特征选择；
- 通信：69 维场景下 MSF 相对 Random 的 HV 提升未达到显著性，因此最终部署保留全部局部特征，不强行使用 MSF；
- 推进：仅 24 维，直接保留全部局部特征。

## 2. V6 数据集与特征空间

- 全局候选特征：**582**（单通道 288 + 跨通道 294）
- 场景数：**354**；窗口样本：**7080**
- 正常/异常/Transition：2277/3891/912
- 跨通道关系对：14；耦合故障模式：invert, lag, decouple
- 所有 train/val/test 按 scenario 切分，Transition 不进入 MSF 的 Fisher/5-NN 二分类训练。

局部数据集：

| 子系统 | 样本 | D | 正类 | 负类 |
|---|---:|---:|---:|---:|
| power | 3647 | 267 | 1370 | 2277 |
| thermal | 3184 | 135 | 907 | 2277 |
| attitude | 3192 | 135 | 915 | 2277 |
| communication | 2741 | 69 | 464 | 2277 |
| data | 2505 | 204 | 228 | 2277 |
| propulsion | 2503 | 24 | 226 | 2277 |

## 3. MSF vs Random / Sparse-control / Fisher Top-K

主指标保持论文式二维 held-out Pareto HV：`(1 - test G-mean, feature ratio)`，reference point=(1,1)，seeds 92–101 配对运行。

| 子系统 | D | MSF HV | Random HV | W/T/L | p | Sparse HV | MSF vs Sparse p | MSF平均K | MSF test G | Fisher same-K G | MSF vs Fisher p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| power | 267 | 0.8137 | 0.7222 | 10/0/0 | 0.001953 | 0.8012 | 0.027344 | 19.7 | 0.8435 | 0.8117 | 0.001953 |
| thermal | 135 | 0.8748 | 0.8240 | 10/0/0 | 0.001953 | 0.8732 | 0.769531 | 16.4 | 0.9185 | 0.8990 | 0.001953 |
| attitude | 135 | 0.9414 | 0.8722 | 10/0/0 | 0.001953 | 0.9385 | 0.375000 | 16.0 | 0.9893 | 0.8968 | 0.001953 |
| data | 204 | 0.9515 | 0.8323 | 10/0/0 | 0.001953 | 0.8932 | 0.001953 | 12.5 | 0.9987 | 0.9405 | 0.001953 |
| communication | 69 | 0.7007 | 0.6847 | 6/0/4 | 0.160156 | 0.6996 | 0.695312 | 6.0 | 0.7203 | 0.5959 | 0.001953 |

关键解释：

- 电源、热控、姿轨控、数据处理的 MSF Pareto HV 均为 **10/10 胜 Random，p=0.001953**。
- 与相同特征预算 Fisher Top-K 相比，上述四个子系统也均为 **10/10 胜，p=0.001953**，说明 wrapper 搜索得到的互补特征组合优于简单 Fisher 截断。
- Sparse-control 显示：电源和数据处理存在显著 Fisher identity 增益；热控和姿轨控主要收益来自多尺度稀疏位置，MSF 与 Sparse-control 差异未显著。
- 通信 D=69 时，MSF vs Random HV 为 6/10 胜、p=0.160156，不足以支持强制使用 MSF，因此最终部署不使用该搜索结果。

## 4. 最终六子系统部署模型

| 子系统 | 部署方式 | D | K | Test 5-NN G-mean | Test Logistic G-mean | Specificity | Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| power | MSF-MOFS | 267 | 18 | 0.8586 | 0.8200 | 0.9934 | 0.6769 |
| thermal | MSF-MOFS | 135 | 11 | 0.9307 | 0.8947 | 0.9974 | 0.8027 |
| attitude | MSF-MOFS | 135 | 15 | 0.9984 | 0.9384 | 0.9908 | 0.8889 |
| data | MSF-MOFS | 204 | 10 | 1.0000 | 0.9936 | 1.0000 | 0.9872 |
| communication | all-local-features | 69 | 69 | 0.7057 | 0.7333 | 0.9987 | 0.5385 |
| propulsion | all-local-features | 24 | 24 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

- 六个子系统最终联合需要计算 **140/582** 个唯一特征，特征计算量按数量压缩 **75.9%**。
- 健康证据阈值只用 validation 正常分布校准；在线证据再做时序平滑后进入三状态 DBN。
- 整星健康度使用关键度加权子系统健康 + mission-success 风险项，避免单个短时假阳性把整星健康度瞬间拉低。

## 5. 闭环故障验证

最终独立仿真检查使用新的随机种子，不把 ground truth 输入健康算法。检测判据：目标子系统证据≥0.5，或目标子系统健康度<85，或整星健康度<85。

| 项目 | 结果 |
|---|---:|
| 正常场景 | 20 |
| 正常场景假报警 | **0** |
| 正常平均/最低健康度 | **99.99 / 97** |
| 单通道故障 | **71/72 = 98.61%** |
| 单通道平均检测延迟 | 8.99 tick |
| 耦合故障 | **26/30 = 86.67%** |
| 耦合故障平均检测延迟 | 44.19 tick |

当前未检出的边界场景为：`param_8 noise`、`p03_p04 lag (weak)`、`p01_p08 lag` 两个强度以及 `p01_p08 decouple (weak)`。这些结果应保留为方法边界，而不是用测试集继续调参消除。

## 6. Normal → Transition → Degraded → Fault

12 路参数分别构造渐进 drift，ground-truth 阶段由故障注入量与工程限独立定义。

| 阶段 | 整星平均健康度 | 目标子系统平均健康度 | 平均证据 |
|---|---:|---:|---:|
| normal | 100.00 | 100.00 | 0.0011 |
| transition | 99.83 | 99.73 | 0.0109 |
| degraded | 72.18 | 27.04 | 0.6634 |
| fault | 60.58 | 1.05 | 1.0000 |

- 12/12 产生预警；12/12 都在工程限越界前预警。
- 平均故障开始后预警延迟：**53.08 tick**。
- 相对工程限越界平均提前：**80.25 tick**。

## 7. 在线计算开销

- 全 582 特征提取中位耗时：556.11 μs/frame。
- 140 个部署特征提取中位耗时：223.58 μs/frame。
- 特征提取中位耗时下降：**59.8%**，约 **2.49×**。
- 当前 Node.js 端到端健康计算中位耗时：597.10 μs/frame。

## 8. 重要边界

- 所有结果仍是合成卫星遥测环境，不能表述为真实在轨准确率。
- 跨通道关系组是工程启发的受控抽象，不是经过真实星上系统辨识得到的物理方程。
- 重复 seed 的统计证明的是固定合成数据集上的搜索稳定性，不等价于跨卫星/跨任务泛化。
- 通信子系统未证明 MSF 优于 Random，因此最终采用全局部特征证据；推进维度过低，也不使用 MSF。
- `p01_p08` 等部分耦合 lag/decouple 场景仍是当前方法明确的薄弱点。

## 9. 最终算法路径

```text
single telemetry frame
  -> persistent 64-point history
  -> 582-D candidate feature space definition
  -> subsystem-local feature plan
       power: MSF 18 / 267
       thermal: MSF 11 / 135
       attitude: MSF 15 / 135
       data: MSF 10 / 204
       communication: all 69
       propulsion: all 24
  -> 140 unique online features
  -> subsystem logistic evidence
  -> evidence smoothing
  -> stateful 3-state DBN
  -> subsystem health
  -> criticality-weighted overall health
```

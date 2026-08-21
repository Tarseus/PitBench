# CVRP QoI v2-candidate.0：experiment-conditioned matching pilot

日期：2026-08-21

状态：solver-free candidate validation；不是 v2.0 freeze，也不是 solver-response
experiment。旧的 v1.1 conditioned-matching pilot、测试和 artifact 已删除，本报告只对应
35 轴 `CVRP Static QoI Basis 2.0-candidate.0`。

## 实现范围

实现了审计表中的全部 35 轴，分为 scale、demand/capacity、global/depot geometry、
local geometry、MST、shape、DBSCAN clustering 和 demand-spatial coupling。

matching cost 的规则为：

- raw 轴只报告，不进入 cost；
- experimental 轴只进入 treatment difference profile，不进入默认 cost；
- scale、struct_core 和 scale_conditioned 轴是本轮 candidate cost；
- 对每个 treatment，排除其主动改变及可能 downstream 的 blocks；
- 在 pooled source/target population 上使用 median/MAD，MAD 为零时退回 IQR；
- blocks 等权，block 内 axes 等权；
- 固定同一 ground cost 后比较 greedy 与 exact assignment/OT。

DBSCAN 使用 customer-only、`min_samples=4`，epsilon 为 minimum-area oriented
bounding rectangle 的边长尺度除以 `sqrt(n)-1`。MST tie 使用 isometry-invariant
distance signatures；依赖非唯一 MST 的 topology 轴标记为 undefined。

## Hidden-CRN panel

每个 treatment 生成 48 对 source/target instances。两端共享 latent generator seeds，
只改变一个 Uchoa-style treatment；target 顺序随后被隐藏。共计算 576 个 instance
observations，solver runs 为 0。

| treatment | unconditioned OT recovery | conditioned OT recovery | confounder distance before | after |
|---|---:|---:|---:|---:|
| customer count | 72.92% | 72.92% | 1.0205 | 0.9901 |
| depot positioning | 91.67% | 100.00% | 0.0561 | 0.0000 |
| customer positioning | 50.00% | 91.67% | 0.5064 | 0.0353 |
| demand dispersion | 100.00% | 100.00% | 0.0000 | 0.0000 |
| route size | 100.00% | 100.00% | 0.0000 | 0.0000 |
| quadrant coupling | 100.00% | 100.00% | approximately 0 | approximately 0 |

Aggregate：

- unconditioned OT recovery：85.76%；
- conditioned OT recovery：94.10%；
- recovery gain：8.33 percentage points；
- mean confounder distance：0.2638 降至 0.1709，下降 35.23%；
- 6/6 treatments 没有 recovery loss；
- 6/6 treatments 没有更差的 confounder distance；
- 2/6 treatments 有严格 recovery gain。

## 结论边界

结果支持“同一静态 QoI basis 上按 treatment 改变 matching cost”在 synthetic hidden-CRN
panel 中具有增量价值，严格收益集中在 depot/customer positioning。三个 treatment 已经
在普通 cost 上达到 100%，无法展示增量。

customer-count recovery 仍只有 72.92%。这不是 matching algorithm 能修复的问题：当前
Clark-Evans、sqrt(n) MST correction、hull 和部分 topology quantities 仍存在 finite-n
leakage。正式 v2.0 必须先完成 Heins normalization 对照与 scale gate。

本实验不证明 canonical CVRP distance，不是 external real-population validation，也没有
测量任何 solver treatment effect。

## 复现

```powershell
uv run python -m pitbench.cli.main qoi validate-cvrp-v2-matching `
  --pair-count 48 `
  --generator-seed 20260821 `
  --output results/cvrp-qoi-v2-conditioned-matching-candidate.0.json
```

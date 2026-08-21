# CVRP v1.0 exact single-QoI solver pilot

日期：2026-08-21

## 研究问题

在冻结的 16 个 QoI 中，如果 generator pair 经 v1.0 extractor 反向验证、确实只有一个
QoI 改变，PyVRP 的 fixed-budget response 是否随之改变？

这不是 observational correlation scan，也不涉及统一 geometry。估计量是同一 generator
seed、同一 solver seed 下的 `target - source` paired mean；bootstrap 以 generator seed 为
cluster。

## 设计

- Solver：PyVRP 0.13.4，单线程；
- exact axes：`demand_cv`、`pairwise_distance_median`；
- generator seeds：6；matched solver seeds：3；
- 每个 axis 有 source/target 两级，共 72 个 evaluation runs；
- 固定预算：每 run 0.5 秒；
- reference phase：每个 pair level 用固定 seed 另跑 2 秒，共 24 runs；
- 72/72 evaluation 与 24/24 reference 最终解均通过 independent CVRP verifier；
- gap/reference：每个 pair level 的独立长预算结果与 evaluation results 中的最小 objective；
- 报告 feasibility、best-observed gap、primal integral、time-to-1%-target、归一化最终
  objective、wall/CPU time 与 iterations；
- Windows in-process pilot 未获得可信的 per-run peak RSS，因此 artifact 明确记为缺失。

完整结果位于 `results/cvrp-qoi-single-axis-pilot-v1.json`。

## 结果

### 只提高 `demand_cv`：0 → 0.5

构造保持客户坐标、客户数、总需求、容量、所有空间 QoI 和两个 radial coupling 不变。
target 相对 source：

| Outcome | Paired mean | Generator-cluster bootstrap 95% interval |
|---|---:|---:|
| feasible rate | 0 | [0, 0] |
| best-observed final gap | 0 | [0, 0] |
| primal integral | +0.000452 | [+0.000146, +0.000816] |
| time-to-target or budget | +0.00912 s | [+0.00521, +0.01334] |
| objective / pairwise median | +0.3191 | [+0.2031, +0.4533] |
| CPU time | -0.03559 s | [-0.07726, +0.00694] |
| iterations | +234.7 | [+153.2, +319.7] |

在这个 identification fixture 上，需求离散度没有改变最终可行率，三个 solver seeds 也都
达到各自 level 的 best-observed final objective；但 target 的 anytime area 更差、达到 1%
target 更慢，并触发更多 iterations。归一化最终 objective 的差异表示 demand allocation
改变了 routing optimum/难度，而不是距离单位变化。

### 只提高 `pairwise_distance_median`：坐标统一放大 2 倍

v1.0 extractor 验证其余 15 个 QoI 不变。target 相对 source：

| Outcome | Paired mean | Generator-cluster bootstrap 95% interval |
|---|---:|---:|
| feasible rate | 0 | [0, 0] |
| best-observed final gap | +0.000268 | [-0.00176, +0.00252] |
| primal integral | -0.000159 | [-0.00342, +0.00310] |
| time-to-target or budget | +0.0409 s | [-0.0182, +0.1177] |
| objective / pairwise median | +0.0125 | [-0.0865, +0.1275] |
| CPU time | -0.00260 s | [-0.03299, +0.02865] |
| iterations | -31.5 | [-44.3, -18.6] |

质量、anytime 和 CPU intervals 均跨 0，未检测到 normalized quality/resource response；
但 iterations 减少。这更像 EUC_2D rounding、整数 cost 尺度或单位时间内搜索机制的
representation response，而不是实例结构难度改变。它应进入 RQ3 representation
brittleness，而不应成为 structural treatment。

## 可支持与不可支持的结论

本 pilot 支持：

1. “QoI 能定义可干预 treatment axis”需要逐定义证明，不能从 feature correlation 推出；
2. `demand_cv` 在严格配对的 synthetic fixture 上产生可检测的 anytime/mechanism response；
3. coordinate scale 对 normalized quality 和 CPU 没有检测到影响，但对 PyVRP 的
   iteration 机制有影响，值得作为 semantic-equivalent representation transform 继续测；
4. final gap 会遗漏 `demand_cv` 的早期 response，支持同时报告 primal integral 与
   time-to-target。

本 pilot 不支持：

- 其他 14 个 QoI 的独立因果效应；它们当前不是 exact single-QoI treatments；
- 跨实例族、跨 solver 或 Base/Human/Agent 的总体结论；
- 把两个 exact axes 或全部 QoI 合成统一 geometry；
- 以 2 秒 best-observed reference 代替正式 oracle 的 confirmatory gap 结论。

复现实验（PyVRP 保持 optional，不写入 PitBench runtime dependencies）：

```powershell
uv run --with pyvrp==0.13.4 python -m pitbench.cli.main qoi `
  run-cvrp-single-qoi-pilot `
  --output results/cvrp-qoi-single-axis-pilot-v1.json `
  --budget-sec 0.5 --reference-budget-sec 2.0
```

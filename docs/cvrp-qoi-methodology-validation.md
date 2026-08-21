# CVRP QoI 与 PitBench OT-6D 方法需求验证报告

更新时间：2026-08-21

当前结论：**Falsified（v1.0 unified ground geometry 强主张） / Promising（axis-conditioned intervention 主线）**

## 0. QoI schema v1.1 定义与角色验证

QoI schema v1.1 已作为独立版本实现并冻结，没有修改或覆盖 v1.0 artifact。
v1.1 为每个实例轴增加 `raw`、`scale`、`struct_core`、
`scale_conditioned` 或 `experimental` role，并完成以下定义变更：

- Clark–Evans nearest-neighbor normalization；
- 带 `sqrt(n)` correction 的 MST edge mean；
- convex-hull area ratio；
- 明确命名的 capacity-volume route lower bound 和 radial coupling；
- `max_demand_fraction`、volume-LB route size 和非径向 quadrupole coupling；
- Uchoa-X-style C/E/R、R/C/RC、七种 demand family 和 route-size CRN generator。

这里不定义新的 “geometry v1.1”，也不把所有 QoI 合并成统一距离。v1.0 对 unified
ground geometry 的 **Falsified** 结论原样保留。v1.1 只验证定义和 role contract：
`struct_core` 轴接受跨规模稳定性检查；以下有限样本或规模条件轴明确归入
`scale_conditioned`，因此不再被误当成统一的 scale-free geometry：

- `convex_hull_area_ratio`
- `demand_cv`
- `mst_edge_mean_n_corrected`
- `nearest_neighbor_clark_evans_ratio`
- `nearest_neighbor_iqr_clark_evans_ratio`

`demand_mean_fraction` 同样属于 `scale_conditioned`，非径向 quadrupole coupling 属于
`experimental`。32 个 untouched confirmation seeds 上，七个受控方向（包括
non-radial coupling 与 route size）分别验证；Axiom 4 仅表示 `struct_core` role contract，
不构成 unified geometry 的验证或 solver 全局 gate。

paired solver gate 已改为逐 treatment 准入：`scale`、`depot`、`customer_structure`、
`demand_dispersion`、`non_radial_coupling`、`route_size` 各自依赖对应的受控轴证据。
某个轴失败只阻止该 treatment，不阻止其他已验证 treatment。代码中同时保留了 CRN
treatment panel、trajectory outcomes、RQ3 Wasserstein orbit dispersion 和 RQ4
greedy/exact-OT recovery primitives。

冻结结果位于 `results/cvrp-qoi-axioms-v1.1.json`。实际 Uchoa-X 外部验证入口为：

```bash
uv run pitbench qoi validate-uchoa-construct \
  --manifest path/to/uchoa-x-metadata.yaml \
  --output results/cvrp-qoi-uchoa-construct-v1.1.json
```

仓库目前没有附带实际 Uchoa-X 数据，因此尚未伪造 external-validation artifact，也未越过
“先 external construct validation、后 solver experiment”的顺序启动真实 solver runs。
这不是 geometry gate 的失败。外部数据未随仓库提供时，该命令不会创建占位或合成的
external-validation artifact。
manifest 中每个条目必须包含实际 instance path 和冻结的原始 benchmark metadata：

```yaml
instances:
  - id: X-n101-k25
    path: instances/X-n101-k25.json
    depot_positioning: C
    customer_positioning: R
    demand_family: quadrant
    route_size: 5.0
```

## 1. 报告目的

本报告对照 `pitbench_ot_method_v2.pdf` 中的设计要求，说明当前项目围绕 CVRP
Instance QoI、ground geometry、OT coupling 和 solver behavior 完成了什么工作，实验
效果如何，以及文档中的最小研究目标目前是否可行。

需要首先区分两个主张：

1. 当前 QoI 是否能作为 solver-independent、具有明确语义的实例变化坐标；
2. 当前 QoI 是否已经能组成统一的 solver-relevant ground geometry，并使 OT matching
   优于简单 matching。

实验支持第一个主张的大部分内容，但不支持第二个主张。QoI 能可靠描述若干受控实例
变化方向；当前完整 QoI 集合尚不能直接作为与规模分离的统一结构距离。

## 2. 文档要求的核心方法

PDF 将 solver 表示为从实例条件到随机行为分布的 stochastic kernel：

\[
K_A(dy \mid x).
\]

输入变化被拆成三类方向：

- `equiv`：语义保持的 representation / nuisance perturbation；
- `scale`：customer count 等 computational mass 变化；
- `struct`：空间结构、需求分布、capacity pressure、population shift 等变化。

文档的目标不是把所有差异压缩成一个 Wasserstein 数，而是构造 Transport Response
Matrix：

\[
R_{A,a,k}
=
\mathbb E_{(x,x')\sim\pi_a}
\left[D_{A,k}(x,x')\right],
\]

其中行 (a\) 是实例变化方向，列 (k\) 是 performance、reliability、resource 等
solver outcome 坐标。

文档还要求遵守 treatment–confounder separation：主动测试的轴是 treatment，不能在
matching 中被最小化；coupling 只控制其他 nuisance mismatch。有自然 generator pairing
时，应优先采用 common-random-number pairing，而不是为了使用 OT 强行重新匹配。

## 3. 已实现的干净 QoI 主线

清理后的实现只保留以下组件：

- 版本化、带 fingerprint 的 QoI schema；
- CVRP Instance QoI 和 Solver QoI 定义；
- 从 normalized CVRP instance 提取 16 个 Instance QoI 的确定性实现；
- 支持固定随机源的 CVRP 受控 generator；
- 平移、旋转、反射和 customer relabeling 等认证等价变换；
- 对 PDF 八条 ground-geometry 公理的 solver-free、逐轴验证；
- 核心 QoI 测试、公理测试和两个结果 artifact。

当前 Instance QoI 包含：

- 规模和有单位原始量：`customer_count`、`capacity`、`total_demand`、
  `pairwise_distance_median`；
- capacity / demand 结构：`vehicle_lower_bound`、`fleet_fill_ratio`、
  `demand_mean_fraction`、`demand_cv`；
- 空间结构：depot distance、nearest-neighbor distance、MST edge、convex hull；
- demand-location coupling：`demand_depot_correlation`、
  `demand_weighted_depot_ratio`。

有单位的 `capacity`、`total_demand` 和 `pairwise_distance_median` 被保留用于 raw
reporting，但不被误称为单位不变的 ground coordinates。

## 4. 公理验证实验

### 4.1 实验设计

该实验完全 solver-free，不调用 PyVRP，也不使用 target solver outcome 选择轴、阈值
或 normalization。

确定性 panel 包含：

- 16 个独立 generator seeds；
- customer count `{50, 100, 200, 500}`；
- 128 个仅实例侧 calibration designs，用于计算逐轴 IQR 尺度；
- 四种等价变换：translation、rotation、reflection、relabeling；
- 坐标单位和 demand/capacity 单位同时乘以 10 的单位负对照；
- unrelated instance negative controls；
- 五个固定 customer count 的受控结构变化方向。

预注册门槛包括：

- 数值不变性误差不超过 `1e-9 IQR`；
- equivalence / unrelated distance ratio 不超过 `0.05`；
- scale monotonicity Spearman 不低于 `0.99`；
- 单个结构轴在纯规模 sweep 中的 median leakage 不超过 `0.25 IQR`；
- 至少 75% 的结构轴满足 scale stability；
- 受控结构变化 median effect 至少为 `0.5 IQR`；
- 至少 80% 的受控结构方向通过。

### 4.2 八条公理结果

以下七条通过：

- **Axiom 1 — Solver independence**：全部 Instance QoI 声明为 solver-independent；
  添加 gap、wall time、iterations 等 synthetic solver annotations 后，QoI 最大变化为 0。
- **Axiom 2 — Semantic invariance**：四类语义等价变换下的 median standardized
  distance 为 `1.84e-16 IQR`；equivalence / unrelated ratio 为 `2.03e-16`。
- **Axiom 3 — Scale sensitivity**：`customer_count` 与请求规模完全一致，规模 log-ratio
  的 Spearman 为 `1.0`。
- **Axiom 5 — Unit and representation robustness**：全部 unit-robust axes 的最大误差为
  `5.17e-15 IQR`；三个 raw unit axes 被显式排除。
- **Axiom 6 — No circular learned semantics**：没有使用 target solver outcome 或 learned
  embedding；solver runs used 为 0。
- **Axiom 7 — Version freezing**：当前 QoI spec 与版本 `1.0` 独立固定的 fingerprint
  基准一致；所有 observation 也必须携带该基准 fingerprint。calibration fingerprint
  和确定性 repeat 同时冻结并记录；同版本下修改轴或元数据会使该公理失败。
- **Axiom 8 — Falsifiability**：equivalence、unit、unrelated negative controls、受控结构
  shifts 和固定阈值全部进入 artifact；不存在事后调权。

失败的是：

- **Axiom 4 — Structure–scale separability**：12 个结构轴只有 7 个满足 scale
  stability，通过比例为 `58.3%`，低于预注册的 `75%`。

### 4.3 受控结构方向的检测效果

五个受控方向全部通过，且每个方向都严格保持 `customer_count` 不变：

- capacity ratio `0.30 → 0.08`：
  - `vehicle_lower_bound` effect：`1.00 IQR`；
  - `demand_mean_fraction` effect：`1.00 IQR`；
  - 16/16 seeds 方向一致。
- cluster spread `3 → 24`：
  - `nearest_distance_mean_normalized` effect：`1.65 IQR`；
  - `mst_edge_mean_normalized` effect：`1.48 IQR`；
  - 16/16 seeds 方向一致。
- demand distribution `uniform_integer → bimodal`：
  - `demand_cv` effect：`1.46 IQR`；
  - 16/16 seeds 方向一致。
- demand-location coupling `anticorrelated → correlated`：
  - `demand_depot_correlation` effect：`2.06 IQR`；
  - `demand_weighted_depot_ratio` effect：`2.86 IQR`；
  - 16/16 seeds 方向一致。
- depot mode `center → corner`：
  - `depot_distance_mean_normalized` effect：`1.08 IQR`；
  - 16/16 seeds 方向一致。

这证明 QoI 确实能定量描述多个文档关心的实例变化角度。Axiom 4 的失败不是因为
QoI 对结构变化不响应，而是部分结构 QoI 同时系统性响应规模变化。

### 4.4 规模稳定与规模污染的轴

通过 scale-stability gate 的轴：

- `vehicle_lower_bound`
- `fleet_fill_ratio`
- `demand_cv`
- `depot_distance_mean_normalized`
- `depot_distance_iqr_normalized`
- `demand_depot_correlation`
- `demand_weighted_depot_ratio`

没有通过的轴：

- `convex_hull_fraction`
- `demand_mean_fraction`
- `mst_edge_mean_normalized`
- `nearest_distance_iqr_normalized`
- `nearest_distance_mean_normalized`

后五个量仍然可以用于固定规模切片或作为显式 scale-conditioned coordinates，但不能
在未经修正时直接承担与规模分离的 `cstruct`。

## 5. 已完成的历史研究 probes

旧 probe 实现已从主代码删除，负面结果和原 artifact SHA-256 被压缩保存在
`results/cvrp-qoi-research-summary-v1.json`，以避免未来重复已经失败的实验。

### 5.1 联合 QoI counterfactual

目标是检验完整协方差是否允许 customer count 与其他 QoI 轴合理补偿。192 个
references 上：

- median gain：`0.0`；
- positive gain fraction：`31.25%`；
- bootstrap 95% median lower bound：`0.0`。

结论为 **Falsified**。当前联合 covariance geometry 没有稳定证明跨轴补偿优于纯规模
变化。

### 5.2 冻结结构距离与 solver behavior retention

在已有 72 个 PyVRP runs 上，结构 matching 对主要 conditional-gap behavior distance
的 relative gain 为 `-13.9966%`，leave-one-pair 中只有一个正向结果。

结论为 **Not supported**。当前结构距离没有保留主要 solver quality behavior。

### 5.3 Scale-residual QoI

两个独立 fitting designs 的 primary gains 分别为：

- `+13.89%`；
- `-0.014%`。

pair Jaccard 为 `0.5`。结论为 **Promising but unstable**，不能晋级为默认 geometry。

### 5.4 OT planted-pair recovery

- OT median pair recovery：`1.0`；
- greedy median pair recovery：`1.0`；
- random median pair recovery：`0.0625`；
- median OT minus greedy recovery：`0.0`。

结论为 **No added value**。OT 能恢复 planted coupling，但测试域中的 greedy matching
已经同样好，未证明 OT 的增量价值。

### 5.5 QoI 与 solver behavior 的观察性关联

开发集 72 runs 上，只有 iterations 出现探索性信号，主要 conditional-gap gate 未通过。
随后使用 26 个 untouched CVRPLIB-X instances、3 seeds、两个 budgets，共创建并完成
156 个 PyVRP runs：

- development iterations：`r=0.457, p=0.0038`；
- holdout iterations：`r=0.136, p=0.116`；
- conditional-gap partial association：`r=0.015, p=0.418`；
- joint-QoI nearest-neighbor error 恶化 `12.7%`。

开发集信号没有复现，结论为 **Falsified**。

### 5.6 单轴独立信号扫描

在同一 26-instance holdout 上，控制 customer count 和其余 11 个 QoI，执行 4,999 次
label permutations、Benjamini–Hochberg FDR 和 leave-one-instance-out 检查，没有任何
当前 QoI 轴通过 independent-signal gate。

该结果只否定“当前轴的观察性差异能稳定预测 solver behavior”，不等价于证明受控
generator intervention 对 solver 没有因果影响。

## 6. 对 PDF 研究问题的判断

### RQ1：equivalence、scale、structure 能否被分离？

**部分支持。** Equivalence 和 scale 均通过，五个 structure directions 也都能被检测；
但完整结构轴集合未通过 structure–scale separability。因此当前 QoI 可以用于逐轴、
固定规模或 scale-conditioned 分析，不能直接宣称已经形成合格的统一 `cstruct`。

### RQ2：OT matching 是否优于简单 matching？

**当前测试域内不支持。** Exact OT 明显优于 random，但没有优于 greedy。只有在真实
heterogeneous populations 中证明 greedy 或 size-only matching 留下显著 nuisance
imbalance 后，继续使用 OT 才有研究价值。

### RQ3：Stability 的 distributional view 是否有增量价值？

**尚未得到决定性验证。** 该问题可独立于统一 QoI geometry 测试。应比较简单
MAD/IQR 与 orbit-level distribution distance 对 tail timeout、representation brittleness
或 failure onset 的增量解释力。

### RQ4：Scalability response 能否恢复被结构漂移混淆的 size effect？

**可行，且是当前最有价值的下一项实验。** Generator 已能使用相同随机源构造规模和
结构干预，天然 pairing 可作为 ground-truth coupling。应人为加入预注册的 correlated
structure drift，比较 naive size regression、简单 matching 和 matching 后 scaling
response 是否恢复已知 controlled effect。

在这一实验中，天然 pairing 是主要 coupling；OT 是待比较的方法，不应被预设为答案。

### RQ5：Generalization distance 是否具有 predictive validity？

**当前统一 QoI distance 已被 holdout 证伪。** 若继续研究，应改用预注册的 controlled
shift shells 和 oracle-relative loss degradation，不能继续在已打开的 holdout 上调整
联合权重。

### RQ6：Base/Human/Agent 的 gain surface 是否不同？

**原则上可行，但尚未完成。** 在固定 generator pairing 下分别测量
`G_p(rho_scale)`、`G_p(rho_struct)` 和 representation stability，不要求先拥有 universal
instance distance。只有比较 patch gain surface 时才需要新增严格配对的 solver runs。

## 7. 对文档最小研究目标的总体判断

PDF 的原始最小目标要求同时证明：

1. solver-independent、size-aware、structure-aware geometry 能稳定分开三类输入扰动；
2. OT matching 比 naive matching 更少 confounding；
3. 这种 coupling 能更好解释 patch sensitivity。

按当前实例化，第一项只部分通过，第二项没有通过，第三项尚未完成。因此不能声称
原始最小目标已经可行或得到验证；当前完整实现的结论是 **Falsified**。

但是，更基础的 axis-conditioned sensitivity 方法仍然 **Promising**：

- QoI 已经能稳定测量五个受控结构方向；
- 等价、规模、单位、solver independence 和 version freezing 已通过；
- generator pairing 可以直接支持 causal/paired response analysis；
- OT 可以保留为无天然 pairing 时的可选 coupling engine，而不是方法成立的前提。

更合理的贡献表述是：

> 使用 solver-independent QoI 定义受控实例变化方向，并测量 solver stochastic kernel
> 的 axis-conditioned response；当自然 pairing 不存在且简单 matching 无法控制
> nuisance imbalance 时，再使用 OT coupling。

## 8. 下一步建议

1. 对五个 scale-confounded QoI 预定义处理方式：固定规模使用、理论 scale correction，
   或固定大小的 deterministic subsampling；不得使用 target solver outcome 调整。
2. 原样重跑冻结的八公理 gate，要求 Axiom 4 达到至少 75% scale-stable axes。
3. 在 generator 上执行 common-random-number paired solver experiment，逐方向估计
   `Delta gap`、`Delta iterations`、feasible-rate 和 resource response。
4. 优先完成 RQ4 的 planted confounding recovery；它能直接判断 matching 是否恢复已知
   size effect。
5. 只有 simple matching 留下显著 imbalance 时才恢复 OT 实现，并把 OT 与 greedy、
   size-only 和 hand-crafted Euclidean baseline 同时比较。
6. 在新的 untouched instances/seeds 上冻结并验证结论，不继续调整已经打开的 26-instance
   holdout。

## 9. 复现入口与结果

运行八公理实验：

```bash
uv run pitbench qoi validate-cvrp-axioms \
  --output results/cvrp-qoi-axioms-v1.json
```

保留的结果：

- `results/cvrp-qoi-axioms-v1.json`：完整逐轴公理报告；
- `results/cvrp-qoi-research-summary-v1.json`：历史 probes 的压缩结论、关键指标和
  原 artifact SHA-256。

核心实现：

- `pitbench/qoi/schema.py`
- `pitbench/qoi/cvrp.py`
- `pitbench/instances/generate.py`
- `pitbench/distribution/transforms.py`
- `pitbench/distribution/qoi_axiom_validation.py`

当前 production geometry 未被修改；正式公理实验使用 0 个 solver runs。

# CVRP QoI v1.0：逐定义文献审计与干预边界

日期：2026-08-21

## 审计对象与结论

本审计只针对冻结的 `cvrp-instance-qoi` v1.0 的 16 个 QoI，不改写其定义，
也不把它们组合成 geometry。结论分为四类：

- `cvrp_primitive`：CVRP 问题定义中的原始量；
- `cvrp_established`：CVRP 文献中已有相同或非常接近的 descriptor；
- `cross_domain_adapted`：来自 TSP/通用 instance-feature 文献，再适配到 CVRP；
- `project_specific`：当前精确公式是 PitBench 自己的构造，不能写成“文献标准”。

干预状态另行区分：`exact_single_qoi`、`compound`、`descriptor_only`。文献支持一个
descriptor 可用于描述实例，并不自动意味着它能作为独立 treatment。

## 文献基线

Uchoa et al. 的 Set X 通过 depot positioning、customer positioning、demand
distribution、average route size 和规模系统性扩展 CVRP benchmark，明确目标是研究实例
特征如何影响算法表现 [Uchoa et al., 2017](https://doi.org/10.1016/j.ejor.2016.08.012)。
这支持“可干预 generator axis”，但不支持“每一个统计量都能单独操纵”。

Rasku, Kärkkäinen and Musliu 将 distance matrix、demand statistics、nearest-neighbour、
geometry、MST 等特征正式整理为 VRP instance descriptors
[Rasku et al., 2016](https://doi.org/10.4230/OASIcs.SCOR.2016.7)。其中一部分来自 TSP
instance-feature 文献，例如 [Mersmann et al., 2013](https://doi.org/10.1007/s10472-013-9341-2)。
因此，把 NN/MST/convex-hull 特征用于 CVRP 有文献基础，但 PitBench 当前使用 median 或
IQR 的具体归一化仍需单独验证。

近期 CVRP instance-space 工作继续采用 demand/capacity、depot distance、nearest
neighbour、MST、convex hull 和 customers-per-vehicle 等特征，并发现其中若干轴与预算下
算法表现相关 [Notice et al., 2025](https://doi.org/10.1145/3712256.3726405)。这是
observational feature evidence，不是单轴因果识别证据。

## 16 个定义逐项审计

| QoI | 当前定义 | 文献状态 | 定义审计 | 干预状态 |
|---|---|---|---|---|
| `customer_count` | \(n\) | CVRP primitive | 标准规模变量。加入正需求客户必然改变其他需求或有限样本空间统计。 | compound: `scale` |
| `capacity` | \(Q\) | CVRP primitive | 标准车辆容量；改变它会确定性改变容量压力派生量。 | compound: `capacity_pressure` |
| `total_demand` | \(D=\sum_i q_i\) | CVRP established | 标准需求总量，但不是独立于需求均值和容量压力的 treatment。 | compound: `demand_scale` |
| `vehicle_lower_bound` | \(\lceil D/Q\rceil\) | CVRP established | 是有效的 volume lower bound；不是 bin-packing fleet minimum。CVRP capacity cut 常用此下界。 | descriptor-only |
| `fleet_fill_ratio` | \(D/(\lceil D/Q\rceil Q)\) | project-specific | 文献有 loading \(D/(KQ)\)，但当前 \(K\) 是 volume LB、不是实际 fleet；名称容易过度解释。 | descriptor-only |
| `demand_mean_fraction` | \(\bar q/Q=D/(nQ)\) | CVRP established | demand-to-capacity 是常用特征；这里与 \(D,n,Q\) 完全共线。 | descriptor-only |
| `demand_cv` | \(\mathrm{sd}(q)/\bar q\) | CVRP established | 标准无量纲需求离散度。可在保持总需求、容量与 radial moments 时精确干预。 | exact single-QoI |
| `pairwise_distance_median` | positive customer-pair distance median | cross-domain adapted | distance-matrix summary 有明确先例；选择 median 是 PitBench 的 robust 版本。 | exact single-QoI via coordinate scaling |
| `depot_distance_mean_normalized` | mean depot radius / pairwise median | cross-domain adapted | depot-distance mean 有 CVRP 先例；以 pairwise median 归一化是项目定义。 | compound: `depot_position` |
| `depot_distance_iqr_normalized` | depot-radius IQR / pairwise median | project-specific | 文献多用 mean/range/std；当前 IQR 合理但没有独立 construct validation。 | compound: `depot_position` |
| `nearest_distance_mean_normalized` | mean NN distance / pairwise median | cross-domain adapted | NN summary 是成熟 instance feature；当前 denominator 是项目选择。 | compound: `customer_structure` |
| `nearest_distance_iqr_normalized` | NN-distance IQR / pairwise median | project-specific | NN 分布有先例，精确 IQR 版本缺少逐定义验证。 | compound: `customer_structure` |
| `mst_edge_mean_normalized` | mean MST edge / pairwise median | cross-domain adapted | MST summaries 有成熟先例；冻结结果已显示当前公式有明显 finite-\(n\) leakage。 | compound: `customer_structure` |
| `convex_hull_fraction` | hull customers / \(n\) | CVRP established | points-on-convex-hull proportion 已用于 CVRP instance-space analysis。 | compound: `customer_structure` |
| `demand_depot_correlation` | \(\mathrm{corr}(q_i,r_i)\) | project-specific | 语义清楚，但没有找到该精确 CVRP feature 的直接来源；只描述 radial coupling。 | compound: `radial_demand_coupling` |
| `demand_weighted_depot_ratio` | \(E_q[r]/E[r]\) | project-specific | demand-weighted radial first moment 有应用直觉，但当前 ratio 是 PitBench 构造。 | compound: `radial_demand_coupling` |

`vehicle_lower_bound` 的公式是经典 rounded capacity lower bound，而精确最少车辆数一般对应
bin-packing 子问题；两者不能混称。相关 CVRP 综述/算法文献明确使用
\(\lceil\sum q_i/Q\rceil\) 作为下界
[Letchford, Lysgaard and Eglese, 2004](https://www.lancaster.ac.uk/staff/letchfoa/articles/2004-cvrp-exact.pdf)。

## 为什么不能做 16 个“只变一个 QoI”的实验

前六个 QoI 包含以下恒等式：

\[
D=n\bar q,\qquad
L=\lceil D/Q\rceil,\qquad
F=D/(LQ),\qquad
M=\bar q/Q=D/(nQ).
\]

所以在非退化 CVRP 中，`capacity`、`total_demand`、`vehicle_lower_bound`、
`fleet_fill_ratio` 和 `demand_mean_fraction` 不能同时被当成独立 treatment。它们应属于一个
capacity/route-size mechanism block，分析时报告全部 realized changes。

两个 radial coupling 也存在直接关系。令 \(r_i=d(0,i)\)，使用总体矩：

\[
R_w=\frac{E[q r]}{E[q]E[r]}
=1+\frac{\operatorname{Cov}(q,r)}{\bar q\bar r},
\qquad
\rho=\frac{\operatorname{Cov}(q,r)}{\sigma_q\sigma_r}.
\]

固定 demand 和 radius 的边际分布时，\(R_w-1\) 与 \(\rho\) 只是常数倍关系。因此
`demand_depot_correlation` 与 `demand_weighted_depot_ratio` 不能各自形成单轴 treatment；
它们是同一个 radial covariance treatment 的两种读数。

NN、MST 与 convex hull 则共享同一 customer point pattern。一般位置中的点移动会同时改变
多个统计量。用优化器把 collateral delta 压小可以构造近似配对，但不能把它称为
ground-truth single-QoI intervention。

## 正式 generator 约定

代码目录 `pitbench/instances/cvrp_axis_spec.py` 是 v1.0 的机器可读 specification：

1. catalog 顺序和冻结的 16 个 QoI 完全一致；
2. 每项声明文献状态、treatment axis、干预状态和不可避免的 collateral QoI；
3. `build_exact_single_qoi_panel()` 只生成经 extractor 反向验证、确实只改变一个 QoI 的
   pairs；
4. 当前可接受的 exact axes 只有：
   - `pairwise_distance_median`：统一缩放所有坐标；
   - `demand_cv`：在等半径整数圆点上保持总需求不变，只改变需求离散度。
5. 每个 pair 固定 generator seed；solver 比较必须使用 matched solver seed。

第二个构造刻意使用等 depot radius，以固定两个 radial coupling。它是 identification
fixture，不代表真实 benchmark population。外部有效性应另外在 Uchoa Set X 上检验，不能
由这个 fixture 推出。


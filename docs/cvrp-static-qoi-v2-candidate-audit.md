# CVRP Static QoI v2 候选：文献特征审计与冻结建议

日期：2026-08-21

状态：**implemented candidate.0, not frozen**。35 轴定义已实现为
`2.0-candidate.0`，但不替换 v1.0/v1.1 schema，也不定义新的 unified instance
geometry。已完成 synthetic hidden-CRN solver-free matching pilot；没有运行 solver。
v1.0 对 unified ground geometry 的 **Falsified** 结论保持不变。

## 1. 审计结论

CVRP/TSP 文献已经提供了成熟的静态 instance-characterization taxonomy，PitBench
不需要重新发明一组平铺特征。但文献没有给出社区公认的 canonical CVRP distance。
同一批特征被用于 clustering、algorithm selection、instance-space visualisation 或
knowledge transfer 时，其距离含义也不等于 PitBench 的 treatment-confounder matching。

建议采用三层结构：

1. 冻结 solver-independent 的静态测量表；
2. 对实例对先报告 blockwise difference profile；
3. 由具体 experiment 决定哪些 block 是 treatment，哪些 block 进入 matching cost。

当前实现是 35 轴的 v2-candidate.0，而不是已经发布的 v2.0。35 轴包含 raw、scale、
candidate core、scale-conditioned 和 experimental 项；它们不是 35 个相互独立的
treatment，也不应无条件等权塞入一个欧氏距离。

## 2. 原始文献核实

| 来源 | 实际贡献 | 可借用内容 | 不能直接借用的内容 |
|---|---|---|---|
| [Rasku et al. 2016](https://doi.org/10.4230/OASIcs.SCOR.2016.7) | 76 个 extractors、合计 386 个 feature values；在 168 个 CVRPLIB instances 上用于 clustering 和配置 | distance、DBSCAN、MST、geometry、NN、demand/capacity 的静态 families | 386 维整体包含 local-search 和 branch-and-cut probing，既非纯静态，也非 target-solver independent |
| [Notice et al. 2025 supplementary material](https://ahmedkheiri.github.io/publications/GECCO2025_CVRP_SUPP.pdf) | 明确声明 105 个静态 CVRP features，分 demand、distance matrix、NN、MST、geometry、clustering 六组 | v2 候选池的主要 taxonomy | 补充材料只列 extractor 和 summary groups，没有逐名给出全部 105 个展开列；不能伪造逐列名称 |
| [Gouvêa et al. 2025](https://arxiv.org/abs/2507.10397) | 将 CVRPLIB、X、AGS 和 DIMACS real-world 放入同一 ISA；SIFTED 最终选出 23 个输入 | heterogeneous benchmark families 可被共同描述的经验依据；原始静态 feature families | 23 维和 PILOT 2D projection：23 项中有 12 项是 probing；SIFTED 和 PILOT 均使用 solver performance |
| [Uchoa et al. 2017](https://doi.org/10.1016/j.ejor.2016.08.012) | Set X 系统改变 $n$、depot positioning、customer positioning、demand distribution、average route size | semantic macro axes 和 external construct-validation semantics | 不能把 generator labels 当成连续距离，也不能假设每个统计 micro-QoI 可独立干预 |
| [Heins et al. 2023](https://doi.org/10.1016/j.tcs.2022.10.019) | 为 TSP 的 MST、kNN graph 和 NN-distance 特征推导理论上下界 normalization，并显示 size correlation 降低 | “NN/MST 必须单独处理规模污染”的直接依据和 normalization baseline | 不能把 PitBench 当前的 sqrt(n) 或 Clark-Evans 修正写成该论文验证过的同一公式 |
| [Wu et al. 2026](https://doi.org/10.1016/j.swevo.2026.102297) | 用 8 个 interpretable features 和 cosine similarity 初始化 many-task CVRP transfer similarity，并在线更新 | cosine 是 CVRP feature-vector similarity baseline | 动态 similarity 后续吸收 transfer effectiveness，不再是纯静态 ground cost |

Gouvêa 的 23 项具体由 11 个静态项和 12 个 probing 项组成。静态项是
NN3(sd)、ND8(var)、NN3(skew)、ND2、NN2(max)、NN2(skew)、VRP4、
MST3(median)、ND5(mean)、G2、MST2(mean)；其余 12 项均以 P 开头。
论文还明确说明，SIFTED 先保留与算法 Primal Integral 的绝对 Pearson 相关超过 0.5
的 features，再用 Random Forest OOB error 选择组合。因此，该 23 维不能作为
PitBench 静态 basis 的“文献选定真值”。

## 3. 定义约定

客户集合为 $V=\{1,\ldots,n\}$，depot 为 0，客户坐标为
$x_i\in\mathbb R^2$，需求为 $q_i>0$，车辆容量为 $Q>0$。定义：

$$
D=\sum_i q_i,\qquad
K_0=\left\lceil D/Q\right\rceil,\qquad
d_{ij}=\lVert x_i-x_j\rVert_2.
$$

令 $s$ 为所有正 customer-to-customer distances 的 median。除明确标为 raw 的轴外，
距离用 $s$ 消除 coordinate units。标准差使用 population convention（ddof=0）；
IQR 为 0.75 quantile 减 0.25 quantile；相关系数在任一变量方差为零时标记
undefined，不能静默填 0 后进入 matching。

复杂度栏给出简单、确定性 reference implementation 的上界。可以用 Delaunay、
KD-tree 等加速，但优化实现必须产生相同定义值。match 表示定义验证后可成为 matching
候选；report 表示只报告或用于分层；conditional 表示只能在固定规模或通过规模校正后
使用；experimental 表示不能进入默认 cost。

## 4. 35 轴 v2-candidate.0

### A. Scale / computational mass

| # | 候选轴 | 定义 | role / matching | scale 或表示风险 | 复杂度 | 文献支持 | v1 映射 |
|---|---|---|---|---|---|---|---|
| 1 | customer_count | $n$ | scale；treatment 或 confounder | intrinsic size | $O(1)$ | Uchoa；Rasku DC1；Notice | 保留 |
| 2 | capacity | $Q$ | raw；report | demand-unit dependent | $O(1)$ | CVRP primitive；Rasku/Notice | 保留 |
| 3 | total_demand | $D$ | raw；report | demand-unit dependent；与 $n\bar q$ 共线 | $O(n)$ | Rasku/Notice | 保留 |
| 4 | capacity_volume_lower_bound | $K_0=\lceil D/Q\rceil$ | scale；report/stratify | 非 bin-packing fleet minimum | $O(n)$ | Rasku DC11；Notice minimum fleet size | 重命名 vehicle_lower_bound |
| 5 | volume_lb_customers_per_route | $n/K_0$ | struct_core；match 或 route-size treatment | 与 $n,K_0$ 代数相关 | $O(n)$ | Uchoa average route size；Rasku DC10；Notice | v1.1 新增 |
| 6 | pairwise_distance_median | $s=\operatorname{median}_{i<j,d_{ij}>0}d_{ij}$ | raw；report | coordinate-unit dependent | $O(n^2)$ | Rasku ND1；Notice edge-cost summary | 保留 |

### B. Demand / capacity structure

| # | 候选轴 | 定义 | role / matching | scale 或表示风险 | 复杂度 | 文献支持 | v1 映射 |
|---|---|---|---|---|---|---|---|
| 7 | fleet_fill_ratio | $D/(K_0Q)$ | struct_core；match | $K_0$ 跳变；不是实际 fleet loading | $O(n)$ | 接近 Rasku/Notice total-demand-to-capacity | 保留；继续注明 project-specific |
| 8 | demand_mean_fraction | $\bar q/Q$ | scale_conditioned；match | 与 $D,n,Q$ 共线 | $O(n)$ | Rasku DC5/DC6；Notice demand summary | 保留 |
| 9 | max_demand_fraction | $\max_i q_i/Q$ | struct_core；match | 对单个 extreme 敏感 | $O(n)$ | Rasku DC9；Notice | v1.1 新增 |
| 10 | demand_cv | $\operatorname{sd}(q)/\bar q$ | struct_core；match | 小均值和离散 demand support 需报告 | $O(n)$ | Rasku DC5；Notice demand summary | 保留 |

demand_std_fraction 不单列，因为它严格等于
demand_mean_fraction × demand_cv。demand skewness/kurtosis 可留在完整 extractor
dump，但在 compact basis 中先排除：小样本和低离散 support 下不稳定，且没有覆盖
当前 macro axes 所必需的新增语义。

### C. Global spatial / depot geometry

| # | 候选轴 | 定义 | role / matching | scale 或表示风险 | 复杂度 | 文献支持 | v1 映射 |
|---|---|---|---|---|---|---|---|
| 11 | pairwise_distance_cv | $\operatorname{sd}(d_{ij})/\operatorname{mean}(d_{ij})$ | struct_core；match | 重复点需显式处理 | $O(n^2)$ | Rasku ND1；Notice | 新增 |
| 12 | distinct_distance_fraction_3dp | 将 $d_{ij}/s$ 四舍五入到 3 位后，distinct 值数除以 $\binom{n}{2}$ | experimental | 对 rounding、metric encoding 和重复点敏感 | $O(n^2\log n)$ | Rasku ND2；Notice | 新增；不能默认入 cost |
| 13 | depot_centroid_distance_normalized | $\lVert x_0-\bar x\rVert/s$ | struct_core；match | centroid 对 outlier 敏感 | $O(n)$ | Rasku DC3；Notice | 新增 |
| 14 | depot_distance_mean_normalized | $\operatorname{mean}_i d_{0i}/s$ | struct_core；match | 需 metric 与坐标一致 | $O(n)$ | Rasku DC4；Notice | 保留 |
| 15 | depot_distance_iqr_normalized | $\operatorname{IQR}_i(d_{0i})/s$ | struct_core；match | IQR 是 PitBench robust choice | $O(n\log n)$ | 文献支持 distribution summary | 保留 |

### D. Local spatial geometry

令 $r_i^{(1)}$ 为客户 $i$ 到其他客户的第一近邻距离，
$r_{CSR}=\tfrac12\sqrt{A_H/n}$，其中 $A_H$ 是 customer convex-hull area。

| # | 候选轴 | 定义 | role / matching | scale 或表示风险 | 复杂度 | 文献支持 | v1 映射 |
|---|---|---|---|---|---|---|---|
| 16 | nearest_neighbor_clark_evans_ratio | $\operatorname{mean}_i r_i^{(1)}/r_{CSR}$ | scale_conditioned；conditional | finite-$n$、边界和 hull-shape effect；不是 Heins normalization | $O(n^2)$ | Rasku NN1；Notice；Clark-Evans 跨域归一化 | v1 NN mean 的 v1.1 successor |
| 17 | nearest_neighbor_iqr_clark_evans_ratio | $\operatorname{IQR}_i(r_i^{(1)})/r_{CSR}$ | scale_conditioned；conditional | 同上；IQR 为项目选择 | $O(n^2)$ | Rasku/Notice family | v1 NN IQR 的 v1.1 successor |
| 18 | two_nearest_neighbor_angle_median | 每个客户指向前两近邻的夹角 median，再除以 $\pi$ | struct_core；match | ties 必须按稳定 node id 打破 | $O(n^2)$ | Rasku NN21；Notice | 新增 |
| 19 | depot_as_nearest_neighbor_fraction | 以 depot 为最近节点的客户比例 | struct_core；match | ties 必须固定规则 | $O(n^2)$ | Notice | 新增 |

Heins 的理论 normalization 应作为 #16–17 的正式 baseline。当前 Clark-Evans 版本只有
候选地位；不能因为 2023 论文证明“normalization 有必要”，就推断当前公式已经正确。

### E. MST topology

令 $T_c$ 为 customer-only Euclidean MST，边长和为 $L_T$；令 $T_0$ 为包含 depot、
以 depot 为根的 MST。ties 使用 (weight, min_id, max_id) 的稳定顺序。

| # | 候选轴 | 定义 | role / matching | scale 或表示风险 | 复杂度 | 文献支持 | v1 映射 |
|---|---|---|---|---|---|---|---|
| 20 | mst_total_length_n_corrected | $\sqrt n\,L_T/((n-1)s)$ | scale_conditioned；conditional | sqrt(n) 是 asymptotic correction，不是 Heins 精确 normalization | $O(n^2)$ | Rasku MST1；Notice；Heins | v1 MST mean 的 v1.1 successor |
| 21 | mst_edge_cv | $\operatorname{sd}_{e\in T_c}d_e/\operatorname{mean}_{e\in T_c}d_e$ | scale_conditioned；conditional | topology/finite-$n$ leakage 待测 | $O(n^2)$ | Rasku MST1；Notice | 新增 |
| 22 | mst_leaf_fraction | $|\{i:\deg_{T_c}(i)=1\}|/n$ | scale_conditioned；conditional | small-$n$ effect | $O(n^2)$ | Rasku MST2；Notice degree summary | 新增 |
| 23 | mst_depth_mean_n_corrected | $\operatorname{mean}_i depth_{T_0}(i)/\sqrt n$ | experimental | sqrt(n) 仅候选；root/tie sensitivity | $O(n^2)$ | Rasku MST3；Gouvêa selected MST3；Heins | 新增 |

树的 mean degree 不进入 basis，因为任何含 $m$ 个节点的树平均 degree 都是
$2(m-1)/m$，它主要重新编码规模。文献中 mean MST degree 的 predictive 出现不能
消除这个代数事实。

### F. Shape

| # | 候选轴 | 定义 | role / matching | scale 或表示风险 | 复杂度 | 文献支持 | v1 映射 |
|---|---|---|---|---|---|---|---|
| 24 | convex_hull_area_ratio | $A_H/s^2$ | scale_conditioned；conditional | finite-$n$ leakage；退化 hull undefined | $O(n\log n)$ | Rasku G2；Notice | v1.1 successor |
| 25 | convex_hull_perimeter_ratio | $P_H/s$ | scale_conditioned；conditional | finite-$n$ leakage；退化 hull undefined | $O(n\log n)$ | Notice | 新增 |
| 26 | convex_hull_fraction | hull customer count / $n$ | scale_conditioned；conditional | 对 $n$ 和 collinearity/tolerance 敏感 | $O(n\log n)$ | Rasku G3；Notice | 保留 |

Axis-aligned bounding-box coordinates、area 和 perimeter 不进入默认 basis，因为它们不满足
rotation invariance。若 clustering epsilon 需要 rectangle，只允许使用 minimum-area
oriented bounding rectangle，并记录这一 PitBench definition repair。

### G. Clustering

这一 block 是现有 16/19 轴最明显的缺口。候选统一采用 customer-only DBSCAN，
min_samples=4。为保证 translation、rotation 和 unit invariance，建议用 minimum-area
oriented bounding rectangle area $A_{OBR}$，并设

$$
\epsilon=\frac{\sqrt{A_{OBR}}}{\sqrt n-1}.
$$

该式是对 Rasku density-based parameterization 的 PitBench invariant refinement，不能在
完成 Uchoa R/C/RC construct validation 前写成已冻结标准。统计只使用非-outlier 客户；
无 cluster、单 cluster 或零方差时必须通过 axis_defined 表示，而不是填造数值。

| # | 候选轴 | 定义 | role / matching | scale 或表示风险 | 复杂度 | 文献支持 | v1 映射 |
|---|---|---|---|---|---|---|---|
| 27 | dbscan_cluster_fraction | 非噪声 cluster count / $n$ | experimental | 强依赖 epsilon/min_samples；绝对 count 与规模共变 | typical $O(n\log n)$，worst $O(n^2)$ | Rasku ND5；Notice | 新增 |
| 28 | dbscan_cluster_size_cv | 非噪声 cluster sizes 的 CV | experimental | 少于两个 clusters 时 undefined | 同上 | Rasku ND8；Notice | 新增 |
| 29 | dbscan_outlier_fraction | noise customers / $n$ | experimental | parameter sensitivity | 同上 | Rasku ND6；Notice | 新增 |
| 30 | dbscan_core_fraction | core customers / $n$ | experimental | border fraction 可由其余比例推出 | 同上 | Rasku ND6；Notice | 新增 |
| 31 | dbscan_within_cluster_distance_cv | 非噪声客户到所属 cluster centroid 距离的 CV | experimental | cluster definition 与零均值风险 | 同上 | Rasku ND7；Notice | 新增 |
| 32 | dbscan_max_cluster_demand_fraction | $\max_c\sum_{i\in c}q_i/Q$ | experimental | 同时依赖 clustering 与 capacity；可能大于 1 | 同上 | Rasku DC7；Notice | 新增 |

### H. Demand-spatial coupling

| # | 候选轴 | 定义 | role / matching | scale 或表示风险 | 复杂度 | 文献支持 | v1 映射 |
|---|---|---|---|---|---|---|---|
| 33 | demand_depot_radial_pearson | $\operatorname{corr}(q_i,d_{0i})$ | struct_core；match | 任一边际方差为零时 undefined | $O(n)$ | Uchoa structured demands 提供 construct motivation；精确公式为 PitBench | 重命名 v1 radial correlation |
| 34 | demand_spatial_quadrupole_coupling | 标准化 demand 与 centered、traceless 二阶 location tensor 的 Frobenius norm | experimental | 旋转不变但丢失 phase/sign；退化时 undefined | $O(n)$ | Uchoa quadrant-dependent demand motivation；PitBench definition | v1.1 新增 |
| 35 | demand_local_sparsity_spearman | $\operatorname{Spearman}(q_i,r_i^{(4)})$ | experimental | $n\le4$、ties 或零方差时 undefined | $O(n^2)$ | demand-location motivation；Rasku/Notice NN family；精确组合为 PitBench | 新增 |

旧的 demand_weighted_depot_ratio 不进入 v2 matching basis。固定 demand 与 radius
边际时，它与 radial covariance/Pearson 仅相差常数尺度；保留两者会双重计算同一方向。
它可以作为 audit-only derived value 输出，但不应成为独立 ground coordinate。

## 5. 被审计但暂不进入 35 轴 basis 的特征

| 文献特征 | 决定 | 原因 |
|---|---|---|
| centroid/depot raw (x,y) | 排除 | translation、rotation 和 coordinate-system dependent |
| axis-aligned bounding rectangle | 排除默认 basis | rotation dependent；只允许 oriented rectangle 作为 clustering 内部 normalization |
| demand sd/Q | 排除 | 严格等于 mean/Q × CV |
| demand skewness/kurtosis | full extractor dump only | 小样本/低 support 不稳定，当前 macro-axis coverage 增量有限 |
| sum of n shortest edges、expected tour length | 暂缓 | 与 scale/MST/NN 高度重叠；先做 unsupervised redundancy audit |
| complete kNN SCC/WCC suite，$k\in\{3,5,7\}$ | 暂缓 | 维数膨胀且 normalization 复杂；Heins variants 为第二阶段候选 |
| minimum bottleneck cost | 暂缓 | 与 MST/cluster block 语义重叠待检验 |
| hull inner-point distances、hull-edge summaries | 暂缓 | shape block 已有 area/perimeter/fraction；先检查增量覆盖 |
| all local-search / branch-and-cut probing | 排除 static basis | 依赖算法、预算、实现和随机性 |
| Gouvêa 23D / PILOT 2D projection | 排除 ground cost | performance-supervised selection 和 projection |
| Wu et al. adaptive similarity | 排除 ground cost | 在线吸收 transfer outcome；只保留初始 static cosine baseline |

## 6. Difference profile 与 experiment-conditioned cost

先按八个 block 形成：

$$
q(x)=(q_{\rm scale},q_{\rm demand},q_{\rm global},q_{\rm local},
q_{\rm MST},q_{\rm shape},q_{\rm cluster},q_{\rm coupling}).
$$

每个 feature 在 solver-free reference population 上用 median/MAD 标准化。MAD 为零时该轴
在该 population 中不可用于 matching。首先报告每个 block 的差异：

$$
\Delta_b(x,x')=z_b(x')-z_b(x),\qquad
d_b(x,x')=\sqrt{\frac{1}{|S_b|}\sum_{j\in S_b}\Delta_j^2}.
$$

block 内除以 $|S_b|$，防止 clustering 等维数较大的 block 自动获得更大权重。默认总
cost 只在预注册的 confounder blocks 上聚合：

$$
c_T(x,x')=
\sqrt{\frac{1}{|B\setminus T|}\sum_{b\in B\setminus T}d_b(x,x')^2},
$$

其中 $T$ 是本实验主动改变的 treatment blocks。比较 clustering treatment 时不最小化
cluster block；比较 scale treatment 时不最小化 scale block。

正式 metric study 至少比较：

- blockwise robust standardized Euclidean；
- 只用 instance features、带 shrinkage 的 blockwise Mahalanobis/whitened distance；
- normalized feature vector cosine distance（Wu et al. baseline）；
- 相同 ground cost 下的 nearest-neighbor、greedy assignment 与 OT coupling。

OT 是在 ground cost 上寻找 coupling 的方法，不是与 Euclidean/cosine 并列的 feature
distance。必须固定同一个 cost 后比较 OT 与 greedy，才能判断增量来自 coupling，还是
来自换了一套 features/normalization。

## 7. 冻结 v2.0 前的 solver-free gates

1. **Definition reproducibility**：ties、undefined、distance rounding、MST inclusion、
   DBSCAN 参数全部有测试和 fingerprint。
2. **Semantic invariance**：translation、rotation、reflection、customer relabeling、
   coordinate unit、demand/capacity unit。
3. **Scale audit**：在 $n\in\{50,100,200,500,1000\}$ 上比较当前 correction 与
   Heins theoretical normalization；未通过的轴降为 scale_conditioned。
4. **Uchoa construct validation**：用真实 Set X metadata 检验 C/E/R depot、R/C/RC
   customer、七类 demand、average route size 是否主要改变预期 blocks。
5. **Unsupervised redundancy**：只用 instance features 检查代数重复、rank correlation、
   condition number；不得看 target solver outcome。
6. **Hidden-pair recovery**：在 CRN generator 中隐藏真实 pairing，比较上述 costs 找回
   $G(\rho,u)\leftrightarrow G(\rho',u)$ 的 top-1、top-k、structural imbalance 和
   已知 treatment-effect recovery。
7. **Cross-family support**：在 CVRPLIB A/B/E/F/M/P/X 与 DIMACS real-world 上报告
   missingness、out-of-range 和 block coverage；不使用 performance-informed projection。

只有 gates 1–5 通过，才冻结 static QoI v2.0；gate 6 冻结 matching protocol；solver
response 实验在两者之后进行。若 DBSCAN parameterization 或 NN/MST normalization 未
通过，v2.0 仍可发布其余稳定 blocks，但失败轴必须保留为
experimental/scale_conditioned，不能靠事后调权掩盖。

## 8. 最终决定

PitBench 下一版不应叫“CVRP geometry v2”，而应叫 **CVRP Static QoI Basis v2**。
当前 35 轴候选覆盖 Uchoa 五个 macro characteristics，并补上现有 schema 缺失的
clustering、MST topology、local geometry 和 non-radial demand coupling。

但是现在还不冻结 v2.0，原因不是 literature support 不足，而是三个验证问题尚未解决：

- DBSCAN 的 invariant parameterization 尚未在真实 X 的 R/C/RC labels 上验证；
- 当前 Clark-Evans 与 sqrt(n) MST correction 尚未和 Heins normalization 正面对比；
- raw/report、scale-conditioned、experimental 与 default matching-eligible 的边界
  需要由 solver-free redundancy 和 hidden-pair recovery 固定。

其中 hidden-pair pilot 已使用完整 v2-candidate.0 extractor 运行；条件化 OT 平均恢复率
为 94.10%，普通 OT 为 85.76%。但是 customer-count treatment 两者均只有 72.92%，
继续显示 NN/MST/hull 等有限样本尺度问题。因此该 pilot 支持 experiment-conditioned
cost 的增量价值，但不满足冻结 v2.0 所需的 scale gate。

PitBench 的创新主张应是：**用文献支持、solver-independent 的静态 CVRP
characteristics 构造 experiment-conditioned matching cost，并在已知 pairing 上验证
metric，而不是宣称发明了 canonical CVRP distance。**

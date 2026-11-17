# 8 求解器 × 5 目标任务：实例设计与决策记录

日期：2026-11-09 起持续更新
状态：40 个任务的实例参数/结构配置和 reference import 已完成；solver Base/headroom 与部分 driver qualification 仍待完成；未冻结正式隐藏集，未启动新 Agent 开发。

## 范围与完成标准

本记录覆盖 configs/research/eight_solvers_five_targets_v1.json 中 40 个求解器×目标任务，合并为 15 种共享目标分布。同一目标族内的多个求解器使用完全相同的实例、参考值和隐藏划分。

“实例设计完成”要求：每项有实例结构、因素水平、实例角色与独立 seed 流；声明参数确实被生成器读取；实例语义有独立验证器；reference 有等级并按内容 hash 独立复核；生成和设计审计通过。仅有候选范围不算完成。

“任务可执行”是后续更严格的门槛：还需完成每个 solver driver 的 Base qualification，以及 discovery/qualification 难度和 headroom 分析。此项不改变本轮已经选择并生成的参数范围；在通过前不启动新的 Agent 开发。

## 统一生成与参考协议

1. 每种目标采用三个规模水平；未见规模至少达到可见开发最大规模的两倍。规模、结构、联合变化分层，避免只换 seed 冒充结构泛化。
2. 校准 discovery 5 个、qualification 5 个；Agent 可见开发每任务 5 个；隐藏 ID、规模、结构、联合层分开冻结。角色使用独立命名 seed 流，并保存生成参数、同源组和 hash。
3. 参考搜索沿用 configs/research/generalization_difficulty_v2.yaml：单核、0.1/1/5/20/60 秒档位，discovery 0–4 选档，qualification 5–9 独立核验。MIP 检查原模型可行解和参考目标；路由用独立验算的参考值，校准目标不超过 1% gap。最大档未达标记作删失挑战，保留失败和缺测原因。
4. 新任务预算与参考搜索预算分开。正式任务预算由 discovery 按既有 90% 覆盖规则选择，qualification 只确认一次；不改旧任务预算。求解器的 dual/bound 未经独立验证只记录来源，不称为证书。
5. 可证明时用独立可核验最优值；否则用独立验证的冻结 BKS，明确其等级并照样计算相对差距。求解器之间一致只作佐证。
6. 生成器参数未实现或无独立语义验证的目标须先扩展共享功能模块，再运行校准；不以 unsupported 结果充当普通求解失败。

## v4 参考解的实际构建方式与时限

- 先用确定性构造器、精确 DP、serial schedule 和可行路线建立 2,935 个初始 witness；每条保存输入 hash、解 hash、原问题 verifier 结果与最优性来源。MPS 族的初始 witness 覆盖也由独立重算验证。
- 对 calibration role 用单线程参考求解器按 `0.1/1/5/20/60s` 网格搜索；discovery 为每格前 5 个 replicate，qualification 为不重叠的后 5 个。参考 worker 的外层 watchdog 为 solver limit 加 120 秒 grace；求解器内部时限仍是声明的单次时限。
- 参考 campaign 另含最终 `300s` 提升档，用于改善已存在的 feasible witness；该档不是任务评分时限。主 campaign 共 7,285 个作业，预算上限为 217.13 单核 CPU 小时，g49 上分配 32 个 CPU 并行。repair-v1 共 2,325 个作业，亦使用单线程；修复范围包括 MIP 调度、JSSP 终止状态及 PyVRP routing variants。
- 固定参考后按 instance ID 和输入 hash 合并初始 witness、campaign-v3 和 repair-v1，再以原问题 verifier 逐条复核。final-v3 为 2,935/2,935 verified、0 missing。类别为 603 trusted_optimum、382 best_found、756 computed_optimum、1,194 feasible_reference；外部 solver 的 branch-and-bound 证明没有独立重放，dual/bound 只保留为来源信息。
- 任务 Base/Agent 时限与上面参考时限分开。沿用来源 task 的单线程预算：CBC 1/5s（primary 1s）；SCIP 5/20s（primary 20s）；HiGHS 10/30s（primary 30s）；Choco/Gecode 5/20s（primary 20s）；OR-Tools CP-SAT 10/30s（primary 30s）；PyVRP 5/10s（primary 10s）；VROOM 1/5/10s（primary 10s）。Base discovery/qualification 将按 task 原 seed policy 和这些 source budget 建立独立结果；不把 300s reference 档混入 Agent solve budget。

## 15 种目标设计

| 目标（求解器） | 实例结构与选定范围 | 参考与独立检查 | 当前状态 |
|---|---|---|---|
| 多维背包（CBC/SCIP/HiGHS） | 物品数 80/160/320，容量维度 3/5/8，容量比 0.35/0.50/0.70；利润模式 independent、weakly_correlated、inverse_correlated。 | 按原始物品向量重算容量可行性和目标；小例精确 DP，其余 proof 或 BKS。 | 基础生成器已支持；需核对观测特征和难度。 |
| 集合覆盖（CBC/SCIP/HiGHS） | 列数 120/250/500，行数 80/160/320，行密度 0.05/0.10/0.20；记录行列度分布。 | 在原集合上检查覆盖与成本；最优声明与独立界分开存证。 | 生成器存在；独立成本分布/度偏斜控制需实现，否则不纳入因素。 |
| 设施选址（CBC/SCIP/HiGHS） | 设施数 10/20/40，客户数 50/120/240，总容量比 1.05/1.25/1.60；固定费比例与地理聚类分层。 | 原需求、指派、设施容量、开设费、运输费独立复算。 | 基础生成器存在；需审计紧容量可行性并实现控制因素。 |
| MIP 图着色（CBC/SCIP/HiGHS） | planted 图：顶点 80/160/320，颜色固定 4，跨色边概率 0.02/0.05/0.10；结构-only 层固定顶点和颜色数，仅改变密度。 | 对原边验色；planted coloring 只证明可行，不证明最优。 | 图生成和 CP 验证存在；需加共享图到 MPS 建模、MIP 输入/输出验证。 |
| MIP 单机调度（CBC/SCIP/HiGHS） | 主层为单机加权迟期与释放时间；作业 30/60/120，释放跨度/总加工量 0/0.5/1.0，due tightness 0.5/1.0/1.5。structure-only 层固定规模/释放跨度，只收紧交期；序列相关准备时间不纳入本轮。原 SPT 总完成时间仅作正确性层。 | 从原始作业、开始时刻复算释放、完成、迟期与目标。 | 新 MIP 表示和独立 verifier 已接入；需用固定求解版本重新校准难度。 |
| 装箱（Choco/Gecode/CP-SAT） | 容量 150；物品 120/300/600；均匀、20–100 Falkenauer 风格、双峰 30–45/75–90 分层。 | 检查物品恰好一次、箱载和箱数；小例 proof，大例 BKS。 | 基础生成/验证存在；triplet 分布不受支持，不列入正式因素。 |
| CP 图着色（Choco/Gecode/CP-SAT） | planted 图：顶点 80/160/320，颜色固定 4，边概率 0.02/0.05/0.10；结构-only 层固定顶点和颜色数，仅改变密度。 | 验证原图所有边；仅理论下界和可行色数闭合时标 proven optimum。 | 生成器和验证器存在；待参考质量、难度校准。 |
| Job-shop（Choco/Gecode/CP-SAT） | 随机机器序列为主层；共享机器序列的 flow-shop 单独报告。作业×机器为 10×5、20×10、40×10、40×15；加工时长 1–99 均匀/双峰分层。 | 验证作业内先后、机器互斥、makespan；保存公开来源/hash 或 BKS/bound。 | 生成器和验证器存在；待生成特征审计与难度校准。 |
| RCPSP（Choco/Gecode/CP-SAT） | 活动 30/60/120，资源 3/5/8；分层 DAG 与随机拓扑 DAG；资源紧度 0.70/0.85/0.95，时长 1–20。非抢占、renewable、makespan。 | 建通用实例和逐时资源/先后 verifier；串行排程给可行 BKS，只有可信界闭合才称最优。 | 生成、适配、验证均需实现。 |
| 平衡子集和（Choco/Gecode/CP-SAT） | n=80/160/320，最大权重固定 100，k/n=0.25/0.50/0.75，均匀/重复值结构层，目标偏移总和的 −5%/0/+5%。不引入高精度权重轴，以免 DP 参考成本过高且与规模因素混淆。 | fixed-cardinality DP 参考；独立核验 cardinality、目标距离及并列规则。 | 基础生成/验证存在；重复值生成待审计，DP 上界成本待确认。 |
| CVRP（PyVRP/VROOM） | 单仓闭环 EUC_2D；客户 50/150/300；uniform/clustered/mixed；uniform/depot-correlated 需求；车辆容量固定 50，车数为最少可行车数的 1.0/1.25/1.5 倍向上取整。 | 统一取整距离；独立验客户覆盖、容量、车队上限、路线成本。 | 基础生成/验证存在；新增显式车数字段并令 VROOM 与 PyVRP 使用同一车队上限。 |
| VRPTW（PyVRP/VROOM） | 单仓闭环整数窗；客户 50/150/300；窗松弛参数 20/100/500，服务时间 5/10；uniform/mixed 坐标。窗口由构造的可行路线给出，随宽松程度形成难度层。 | 验到达、等待、服务、时间窗、回仓 horizon、容量及目标。 | PyVRP 支持整数窗导出；VROOM 当前请求漏传时间窗，需实现；窗口参数先按生成器的绝对单位定义，不伪称相对比例。 |
| 异构车队 VRP（PyVRP/VROOM） | 单仓闭环；2 种车（容量 40/80、固定成本 0/20）与 3 种车（30/60/100、0/10/30）分别分层；客户 50/150/300，车辆类型须影响可行性或成本。 | 记录每条路线车辆类型；验车型容量、数量、固定费用和路线成本。 | 规范实例与独立 verifier 已实现；PyVRP/VROOM 求解输入输出适配待完成。 |
| 多仓 VRP（PyVRP/VROOM） | 闭环；2/4 仓，客户分布平衡/偏斜；客户 50/150/300；车辆显式绑定起终仓。 | 验起终仓、客户唯一覆盖、容量、成本。 | 规范实例与独立 verifier 待实现；两种求解输入输出适配待完成。 |
| 开放路径 VRP（PyVRP/VROOM） | 单仓出发、不回仓；客户 50/150/300；uniform/clustered/mixed；容量 50，车辆数为容量下界的 1/1.25/1.5 倍。不得用零成本回仓边替代开放路径。 | 验起点、唯一覆盖、容量；目标不计回仓边。 | 规范实例与独立 verifier 待实现；两种求解输入输出适配待完成。 |

## 自主决策

- 同一目标族的多个求解器共享完全相同的分布，避免输入差异混入算法比较。
- 保留简单正确性层，但不让已知易解的 SPT 结构承担主要优化目标。
- 参考值不因求解器自报 optimal 而自动升级；原问题独立验证优先。
- g49 参考阶段按五档预算、单核、5+5 discovery/qualification 运行。机器资源可并行排程，单次求解时限不变。Agent 开发既有 4 小时、8 CPU、32 GiB 配额不作为参考搜索时限。
- CVRP 容量比的旧生成参数实际上定义单车容量相对总需求的比例，并不直接表达车队松弛。正式设计改为固定单车容量并显式配置车队数量，新增字段前不把该范围视为已实现。
- 40 个任务各自的候选评测时限和线程沿用来源任务原配置：CBC 1/5 秒（主时限 1），SCIP 5/20 秒（主时限 20），HiGHS 10/30 秒（主时限 30），Choco/Gecode 5/20 秒（主时限 20），OR-Tools CP-SAT 10/30 秒（主时限 30），PyVRP 5/10 秒（主时限 10），VROOM 1/5/10 秒（主时限 10）；全部单线程。参考搜索另行沿用五档 protocol。
- 缺失的共享模型/验证能力先补齐，再冻结相应目标；不把目标退化成适配器恰好支持的别的问题。
- 仓库已有参数鲁棒性修改和 g49 参数补测独立保留，不覆盖、不提交。

## 执行记录

| 日期 | 决策/动作 | 证据 |
|---|---|---|
| 2026-11-09 | 初始探测仅覆盖 4 个 MIP 族、8 个实例、16 次 10 秒求解；不作最终配置。 | g49 family-expansion-v1/initial-probe/summary.json：15/16 个可行解；调度实例 260、416 的 SPT 最优可解析证明；6 个 MIP 目标仅为 CBC/HiGHS 交叉佐证且原语义仍待审计。 |
| 2026-11-09 | 较大调度实例另存 HiGHS 超时原解，独立验得 421；可信最优为 416，421 不称最优。 | g49 family-expansion-v1/reference-qualification/reference-ledger-v2.json 与 raw solution hash。 |
| 2026-11-09 | 确认沿用五档预算、discovery/qualification 分离；10 秒探测不是正式任务预算。 | configs/research/generalization_difficulty_v2.yaml。 |
| 2026-11-09 | 为 40 行任务绑定 15 个共享目标设计，并逐项继承来源任务的 solver budgets、primary budget 和线程数；参考搜索仍另用五档校准协议。 | configs/research/eight_solvers_five_targets_v1.json；配置分别保留两种预算，防止混用。 |
| 2026-11-09 | 生成总体草案展开为 30 个结构配置、107 个条件，预计 2,945 个文件：每目标 5 个 Agent 可见开发实例，hidden ID/scale/structure/joint 各 30 个，校准按 10 个/条件。 | configs/research/eight_solvers_five_targets_generation_v1.yaml。此为计划量；当前仍有未实现 generator kind，不能视为已生成。 |
| 2026-11-09 | 将实例规模审计从 MIP/CP/CVRP 三类改为 15 个目标族各自比较，避免把“变量数”和“客户数”等不同单位交叉比较；为 bin packing、图着色、子集和、RCPSP 与 job-shop 增加生成特征提取入口。 | pitbench/research.py 与 pitbench/instances/generate.py。完整生成审计尚未运行。 |
| 2026-11-09 | 开始实现 RCPSP 通用结构：确定 JSON 数据合同、生成可行 DAG/resource 实例、逐时资源/precedence verifier 和 CP-SAT 参考模型；任务层 Choco、Gecode、CP-SAT 调用适配及完整验证仍待完成。 | pitbench/instances/generate.py、pitbench/problem_families/public_verification.py、pitbench/problem_families/verification.py、pitbench/solver_drivers/reference.py。尚未运行求解确认。 |
| 2026-11-09 | RCPSP 30 活动样本的串行可行排程通过原问题 verifier；容量逐时检查及 precedence 均通过。 | 本地确定性生成/验证调用；这不是 g49 参考求解，也不构成性能校准。 |
| 2026-11-09 | g49 探测批次保留构造失败记录；先修正传输路径，再根据实际报错补齐 MPS 背包特征解析和 scalar/level 参数统一读取。 | g49 /data3/gushengda/pitbench-runs/family-expansion-v1/populations-supported-probe-v3 与 -v6 的 suite 不完整记录。期间没有启动 solver 或 Agent。 |
| 2026-11-09 | 平衡子集和按 item 数成比例设置 cardinality，避免 n=80 与固定 k=80 不可行；只用 max_weight=100，校准 cardinality_fraction 0.25/0.5/0.75。 | generation_v1 的 BSS base、range sweep 和特征约束。 |
| 2026-11-09 | 图着色和调度的泛化 scale 特征改用原问题顶点数/作业数，而非 MPS 列数；编码额外变量不能挤掉真实的 2× 问题规模外推。 | g49 子集设计审计首次只因 MIP 图着色变量列 1,284/644 未达到 2×而失败；改用 vertices 后达标。 |
| 2026-11-09 | 在 g49 上完成 12 目标族的实例生成和设计审计：2,380 个实例、880 校准、60 开发、四个隐藏角色各 360。规模外推、结构分离、hash 去重通过。 | /data3/gushengda/pitbench-runs/family-expansion-v1/populations-supported-probe-v12/population-construction.json 与 design-audit.json。仅生成/审计通过，solver references、Base difficulty/headroom 仍未完成；heterogeneous/multi-depot/open VRP 尚未生成。 |
| 2026-11-10 | 全 15 目标族、40 任务绑定生成完毕：2,945 个实例，构造完整，设计审计通过。 | g49 /data3/gushengda/pitbench-runs/family-expansion-v1/populations-v3/population-construction.json 与 design-audit.json；107 条件，校准 1,070、Agent development 75、hidden 四层各 450。 |
| 2026-11-10 | 为 1,950 个 CP/路由实例独立构造初始可行参考；其中 353 个参考被解析为计算最优，1,597 个为已验证可行 BKS。995 个 MIP 实例等待固定 Highs 搜索。 | g49 populations-v3-initial-anchors-v3/references.json；该文件状态 reference_search_pending，不能称正式参考集合完成。 |
| 2026-11-10 | g49 host highspy 1.15.1 wheel 导入失败（undefined C++ symbol）；保持求解器镜像未迁移，改用同一固定 Highs 1.15.1 镜像的 CLI 参考 worker。其首个 1 秒参考试跑返回独立验证可行目标 -6375、solver dual -6389，状态 time limit。 | g49 reference-probe-highs-cli-wrapper-v2 与固定镜像 sha256:810ae162704bc1d04bd21bd5c62bfc54d4687976b074147c8f5ea02157edd581。此 dual 仍是求解器报告值，不是独立证书。 |
| 2026-11-10 | 参考 campaign 准备验证通过：2,945 个实例生成 7,345 个作业，预算上限 217.13 单核 CPU 小时；固定五档与 300 秒最终参考，使用 g49 CPU 96–127。 | configs/research/eight_solvers_five_targets_reference_v1.yaml 与 g49 reference-campaign-v2/campaign.json。Campaign 已启动。 |
| 2026-11-10 | reference-campaign-v2 运行中；最近轮询完成 4,081/7,345，29 个作业运行，BKS 已覆盖 2,312 个实例。 | g49 reference-campaign-v2/status.json；执行失败仍完整保留。 |
| 2026-11-10 | 首轮 CP-SAT UNKNOWN（0.1 秒 JSSP）属于无 incumbent 的预算终止，工作区已改为保留 TIME_LIMIT/no-solution，而非抛 solver_error；加权迟期 MIP 时间截断时把松弛的迟期变量按实际排程重算。随后发现更根本的旧 setup formulation 把 setup 施加在所有先后作业对，过度约束了相邻准备时间语义。 | campaign-v2 原始失败和 design review 保留；新设计把 setup 固定为 0，避免错误模型进入最终参考。 |
| 2026-11-10 | 保留 campaign-v2 输出和错误；在发现 MIP 调度 formulation 过度约束后，于最终 MIP phase 写 STOP，等待正在运行的 300 秒作业收尾，不再派发新作业。新总体将使用无 setup 的标准 weighted-tardiness 模型与独立输入 hash。 | g49 reference-campaign-v2/STOP、status.json；v2 结果不得作为新调度分布的参考。 |
| 2026-11-10 | 参考导入接受 campaign incumbent 的 verified_feasible/verification 标记，并在导入时仍按原输入再次核验；不会要求旧 worker 使用新增字段 locally_verified。 | pitbench/research.py；ReferenceCollection bks.json 的记录字段与初始 population importer 不同。 |
| 2026-11-10 | MIP scheduling pairwise setup rows 对所有先后作业对添加 setup，会比相邻 setup 语义过度约束；决定不把这一近似称作 sequence-dependent scheduling。新目标固定 setup=0，保留标准单机 weighted tardiness、release dates、due-date tightness；可见规模 30 jobs，scale/joint 为 60（2×），校准仍覆盖 30/60/120。 | old campaign-v2 的 scheduling inputs/输出保留但不进入最终分布；修正定义在 generation_v2。 |
| 2026-11-10 | old campaign-v2 已在 STOP 边界收尾：5,395 个作业有运行结果，1,950 个 stopped-before-start；在途 300 秒作业已结束，原始错误与 BKS 全保留。 | g49 reference-campaign-v2/status.json；weighted-tardiness setup 模型结果不作为最终参考。 |
| 2026-11-10 | corrected population-v4 完成构造和设计审计：2,935 个实例、106 个条件，结构、规模与 hash 检查全部通过。 | g49 populations-v4/population-construction.json 与 design-audit.json。 |
| 2026-11-10 | 为 corrected population-v4 建立 2,145 个初始参考，其中 353 个计算最优、其余是独立验证的可行 BKS；790 个 MIP search cases 留待固定 Highs 搜索。 | g49 populations-v4-initial-anchors-v1/references.json。 |
| 2026-11-10 | 扩展 constructive MPS witnesses 后，为 v4 全部 2,935 个实例建立并独立验算初始参考；352 个有最优证明，其余均有可行 BKS，不再有 reference_search_required 项。 | g49 populations-v4-initial-anchors-v2/references.json 与 initial-anchor-prior-v3.json。Final reference campaign 继续改进这些锚点并保留 solver bounds。 |
| 2026-11-10 | 修复 reference verifier 的 near-integral start/t/order 归一化后，campaign-v3 产生的加权迟期 incumbent 可独立重验；Highs CLI warm start 接口也通过 1 秒端到端试跑。 | 重验 campaign-v3 保存候选得到原问题目标 53,258、可行；reference-probe-scheduling-warm-v3 返回可行 BKS 55,731 和 dual 1,107。 |
| 2026-11-10 | 为旧 campaign-v3 的 weighted tardiness/JSSP 误报建立 repair-v1：重跑调度五档、JSSP 0.1 秒终止状态，以及 PyVRP 的异构/多仓/开放路径参考；其 prior 复用 hash 匹配的 anchor 与 campaign-v2 BKS。 | g49 reference-repair-v1 已准备 2,325 作业，在 CPU32–63 运行；截至最近检查已完成 520，0 个 solver_error。campaign-v3 继续在 CPU96–127 运行。 |
| 2026-11-10 | repair-v1 的 PyVRP 0.14.1 adapter 对异构车队、多仓、开放路径分别完成 2 秒单核 smoke solve；三个结果均由新的 routing-variant verifier 独立验算通过。 | g49 /data3/.../adapter-probes/pyvrp-heterogeneous-v2、pyvrp-multi-depot-v1、pyvrp-open-v1。此为接口资格化，不是最终 BKS 校准。 |
| 2026-11-10 | VROOM 规范 request/solution 路径已按统一 vehicle type schema 接入 driver；g49 的 VROOM 1.15.0 source clone 经 HTTP/2 framing 错误和 HTTP/1.1 连接超时均未成功，因此尚未构建该 Base binary；路由 BKS 仍由固定 PyVRP/native backend 计算。 | task source commit 为 43dd7d0b8b560431eb555bf335cf4797eb7343c；VROOM driver 的 compile/Base qualification 继续标 pending。 |
| 2026-11-10 | 最新轮询：campaign-v3 已完成 7,285/7,285，0 个运行或待处理；2,935 个实例均有参考，其中 1,522 个 solver-reported optimum、1,413 个 BKS；88 个旧 solver/verifier error 保留在执行记录中。repair-v1 已完成 2,048/2,325，32 个 PyVRP 300 秒作业运行、245 个待处理，repair BKS 覆盖 2,819 个实例。 | g49 `/data3/gushengda/pitbench-runs/family-expansion-v1/reference-campaign-v3/status.json` 与 `reference-repair-v1/status.json`；repair 尚未终态，当前参考集合不可冻结。 |
| 2026-11-10 | reference-campaign-v3 与 repair-v1 均到终态：主 campaign 7,285/7,285，保留 88 个 solver_error；repair 2,325/2,325，0 solver_error，覆盖 2,819 个实例。repair 结果在至少 617 个重叠实例上比主 campaign/anchor 有更低目标值。 | g49 `reference-campaign-v3/status.json`、`reference-repair-v1/status.json`；原始结果和失败记录都保留。 |
| 2026-11-10 | 将初始 anchor、主 campaign、repair 按 instance ID 和 hash 合并并导入；2,935/2,935 条 reference 通过原问题复核，missing=0。等级计数：603 trusted_optimum、382 best_found、756 computed_optimum、1,194 feasible_reference。solver optimal 返回及 primal/dual provenance 保留；未独立重放外部 branch-and-bound 证明。 | g49 `/data3/gushengda/PitBench-family-expansion-20261109/private-reference-artifacts/populations-v4-reference-import-final-v3/references.json`。985 个 MPS reference oracle 单独复核全通过；全量导入状态为 `references_verified`。 |
| 2026-11-10 | 全量导入初次失败后发现两个独立问题：MPS 浮点目标需用既有绝对误差容差比较；加权迟期模型有合法的松弛 tardiness 变量，其 MPS primal 目标可高于按原始调度语义复算的目标。修正 trusted MIP oracle 检查：先检查 MPS primal 可行性，再对带调度元数据的模型按源调度语义复算 reference objective。 | g49 工作副本 `pitbench/problem_families/verification.py` 与本地同路径未提交修改；旧导入失败目录 final-v1/final-v2 保留，最终 final-v3 全量复核通过。 |
| 2026-11-10 | 完成 40 行任务到 15 个 profile 的绑定检查；每行的参数范围、结构、角色、source budget/primary budget、线程和 seed policy 均已填写。v4 有 2,935 个实例、106 个条件，设计审计通过；reference 已 hash 对应且全量复核。 | `configs/research/eight_solvers_five_targets_v2.json`、`eight_solvers_five_targets_design_v2.json`、`eight_solvers_five_targets_generation_v2.yaml` 与 g49 v4 construction/design/reference receipts。任务 readiness 仍标记 solver Base/headroom 或 driver qualification pending。 |

## 实例配置完成；任务执行准备仍待办

本轮配置目标已通过逐项审计：40 个 solver×target 行绑定 15 种共享目标设计；所有行有结构、选定范围、实例角色/seed、继承预算和线程；2,935 个实例通过生成审计，2,935 条 reference 按实例 hash 导入并通过原问题独立复核。错误尝试、solver_error、缺测和 reference provenance 均保留。

下列事项是正式启动 solver Base/Agent 评测前的执行资格门槛，不再阻止本轮实例参数/结构配置完成：

1. 完成 Choco、Gecode、CP-SAT 的 RCPSP driver qualification，以及 VROOM 的编译和异构/多仓/开放路径 qualification。
2. 用固定版本 Base 完成 difficulty discovery/qualification 和 headroom 分析；source task 的单线程、seed policy、budget grid 与 primary budget 已沿用，实际是否能作为新任务预算仍待 Base 结果确认。
3. 完成上述资格检查后再冻结 Agent 开发和正式 hidden views；本轮没有启动新 Agent。

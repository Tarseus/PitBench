# Configuration Robustness & Tunability evidence

## M1：参数空间与反向搜索的依据

核对日期：2026-09-10。本轮承接用户确认的 [M0 方案](scope.md)，进行针对性
文献阅读、固定版本源码核对与已有运行记录检查。以下区分原始证据、对
PitBench 的推论和待确认建议。没有运行新的求解实验或确定正式参数范围。

文献与源码快照保存在 [papers/configuration-robustness](../../../papers/configuration-robustness/sources.json)。
这是定向调研，不是穷尽性的系统综述。

## 1. 文献能支持什么

### R1：传统求解器的参数空间需要显式定义

Hutter、Hoos、Leyton-Brown、Stützle（2009），*ParamILS: An Automatic
Algorithm Configuration Framework*，JAIR 36，267–306。
[原文](https://www.cs.ubc.ca/~hoos/Publ/HutEtAl09.pdf) ·
[本地全文](../../../papers/configuration-robustness/sources/hutter-et-al-2009-configuration.pdf)。
重点阅读第 2、5.2.1、7、10 节。

论文研究传统 SAT 求解器和 CPLEX。配置问题的输入包括参数空间、实例、
单次运行截止时间、观测成本和汇总目标。参数可以具有条件依赖与组合限制。
作者记录配置、实例、seed、截止时间和成本，并用未参与配置选择的实例及
seed 评估搜索得到的配置，明确指出选择阶段的表现存在偏差。

在 SPEAR 实验中，作者按合理上下界离散数值参数，并包含默认值；在 SAPS
实验中，区间包含默认值但不一定以默认值为中心。在 CPLEX 实验中，作者
选择影响搜索轨迹的参数，特意排除了会改变问题解释或解数值精度的参数。
结论还明确说明，配置器需要使用者提供参数允许值，不能替使用者决定其含义。

**对本项目的支持：**显式声明参数范围、固定精度要求、记录配对 seed、
区分搜索和复测，都有直接的传统求解器研究依据。该论文寻找好配置，未定义
本项目的反向压力测试得分，也未规定应使用默认值上下浮动某个百分比。

### R2：参数交互与研究区域都会改变结论

Hutter、Hoos、Leyton-Brown（2014），*An Efficient Approach for Assessing
Hyperparameter Importance*，ICML，PMLR 32(1)，754–762。
[原文](https://proceedings.mlr.press/v32/hutter14.pdf) ·
[本地全文](../../../papers/configuration-robustness/sources/hutter-et-al-2014-parameter-importance.pdf)。
重点阅读第 2、3.3、4.3 节。

作者明确指出，一次只改变一个参数得到的是其他参数固定时的局部信息。
其 fANOVA 方法分别研究主效应和交互效应，并在传统 SAT、MIP、ASP 求解器
上实验。全配置空间和表现较好的区域可能得到不同的参数重要性结论；
CPLEX 的部分参数很能解释变差，却不能解释超过默认配置的改善。

**对本项目的支持：**联合搜索有必要考虑的真实机制；“寻找导致退化的参数”
与“寻找能够改善默认表现的参数”可能关注不同变量。论文的重要性统计量
没有被选为 PitBench 指标，其区域限制和方差分解也不是最坏情况保证。

### R3：SMAC3 提供搜索能力，反向评测目标由项目定义

Lindauer 等（2022），*SMAC3: A Versatile Bayesian Optimization Package
for Hyperparameter Optimization*，JMLR 23(54)，1–9。
[原文](https://jmlr.org/papers/volume23/21-0888/21-0888.pdf) ·
[本地全文](../../../papers/configuration-robustness/sources/lindauer-et-al-2022-configuration.pdf)。
重点阅读第 2.1、2.2、2.4 节；同时核对了官方
[目标函数示例](https://automl.github.io/SMAC3/latest/4_minimal_example/)。

SMAC3 支持自定义标量目标、结构化参数空间及多实例算法配置。论文中的
algorithm configuration 模式通过 racing 分配实例评估，并使用针对其目标
设计的代理模型和成本处理。

**对本项目的支持：**可把“寻找相对默认配置退化更大的组合”交给黑箱搜索器。
反向搜索是本项目已批准的评测设计，不是该论文已经验证的标准鲁棒性指标。

实现时仍需核对选定软件版本的目标方向、成本编码与失败返回规则。论文中的
logEI、运行时间截断及插补处理不能直接当作固定预算质量退化的协议。
搜索器的随机 seed 与求解器的重复运行 seed 也是不同的记录项。

### D1：自动调参工具同样要求用户给出范围

irace 官方 *User Guide*，版本 4.4.9000，2026-09-09。
[官方手册](https://mlopez-ibanez.github.io/irace/irace-package.pdf) ·
[本地快照](../../../papers/configuration-robustness/sources/configuration-user-guide.pdf)。
核对第 5.1.1–5.1.5 节。

手册要求为数值参数提供区间，为类别参数提供取值集合，并支持依赖域、条件
参数和禁止组合。第 5.1.2 节建议选择与调参任务相关的值；范围扩大可能增加
搜索难度。它在不确定时倾向较宽区间，是为了寻找好的配置，不能直接作为
本项目“合理扰动多远”的依据。

手册也允许排除已知表现不好的组合。对本项目主动寻找弱点的目标，不能仅因
某个合法组合表现差就把它排除；约束需要来自事先声明的参数范围与合法性要求。

## 2. PyVRP 0.14.0：机制、默认值与可用参照

任务固定源码提交为 `5d9776a954b810bdb3fe71d47b1e7de7cffe90d2`。
参数入口为 `SolveParams`，当前算法是 iterated local search。
Wouda、Lan、Kool（2024）的 [PyVRP 论文](../../../papers/resource-efficiency/sources/wouda-et-al-2024-routing-solver.pdf)
实验使用 0.5.0 的 HGS 实现，其种群参数不能直接搬到当前版本；本轮核对了
第 4、6 节和附录 A 的参数列表。

### 有直接依据的候选参数

- `neighbourhood.num_neighbours`：默认 **50**，类型为正整数，控制候选邻域
  大小。当前普通 CVRP 最多保留其他客户数那么多邻居；继续增大输入值会被
  截断成相同邻域，不能将这些值视为不同强度的实际扰动。
- `ils.history_length`：默认 **300**，正整数，控制 late acceptance 的历史长度。
  当前源码在每轮使用该历史判断候选解是否可接受。
- `penalty.penalty_increase`：默认 **1.50**，构造检查要求至少为 1。
  可行注册比例不足时用于提高搜索罚系数。
- `penalty.penalty_decrease`：默认 **0.90**，构造检查范围为 `[0,1]`。
  可行注册比例足够时用于降低搜索罚系数。

上述默认值来自当前源码和安装版本的参数接口，不是拟定的搜索范围。
原始源码快照分别见
[邻域构造](../../../papers/configuration-robustness/sources/neighbourhood-construction.txt)、
[迭代搜索](../../../papers/configuration-robustness/sources/iterated-search.txt)、
[罚系数管理](../../../papers/configuration-robustness/sources/penalty-management.txt)。

### 历史默认值可以提供参照，但没有验证整段区间

上游 [PR #989](https://github.com/PyVRP/PyVRP/pull/989) 调整了：

- 邻域大小 **60 → 50**；
- 历史长度 **500 → 300**；
- 罚系数增加倍率 **1.25 → 1.50**；
- 罚系数降低倍率 **0.85 → 0.95**。

当前 0.14.0 的降低倍率为 **0.90**，不能把 PR 合并时的 0.95 写成当前默认值。
PR 同时改变了历史缺失时采用的接受参考，因此上游基准结果不是纯调参的因果
证据。其主要预算也长于本项目 5/10 秒。

[本地讨论记录](../../../papers/configuration-robustness/sources/upstream-tuning-discussion.json)
与 [改动记录](../../../papers/configuration-robustness/sources/upstream-tuning-diff.json)
保留了原文。**近期上游实际使用过这些值**，可以作为讨论范围的依据；
“两个值之间全部合理”以及“适用于本项目短预算”仍须单独说明和评测。

### 容易产生无效变化的参数

`ils.num_iters_no_improvement` 默认 **150,000**，达到连续无改善阈值才触发
重启。复查此前规模实验中 50/100/200 客户的原版运行：5 秒的 540 次运行
总迭代数为 **10,268–19,699**；10 秒的 540 次为 **19,796–39,949**。
这些运行均不可能达到默认重启阈值。
[检查记录](../../../papers/configuration-robustness/sources/existing-iteration-observations.json)。
这不是本轮新实验，也不能据此推断所有实例或改参后的运行次数。

`penalty.solutions_between_updates` 默认 **500**，每积累指定次数的注册才
更新罚系数。范围若远超实际注册次数，也会让改变相关参数失去作用。
`target_feasible` 默认 **0.65**，接口允许 `[0,1]`，但两个极端值的搜索含义
不同于默认附近微调。`feas_tolerance` 在这里是罚系数更新的比例容差，
不是最终解的可行性检查容差，不能仅因名称相同就混淆。

对于当前无时间窗、对称距离的普通 CVRP，`weight_wait_time` 的等待项为零，
`symmetric_proximity` 的对称化也不改变距离邻近度；不建议仅因接口存在就将
它们作为这批实例的主要扰动参数。其他 VRP 变体须重新检查这一条件。

`perturbation.min_perturbations` 和 `max_perturbations` 默认 **1、25**，
必须为非负整数且满足前者不大于后者。源码允许两者都为零，此时扰动步骤
直接返回。这是明显的机制开关边界，是否允许进入首轮压力范围属于协议决定。

## 3. HiGHS 1.15.1：策略开关与数值投入

源码提交为 `04024d701f79feb8e2f18bc3df0dffc04ef05088`；当前 highspy 报告版本
1.15.1，绑定哈希 `04024d7`。使用 `Highs.writeOptions` 导出了
[有效默认选项](../../../papers/configuration-robustness/sources/effective-default-options.txt)，
并核对固定提交的 [选项定义](../../../papers/configuration-robustness/sources/solver-options.txt)。
接口允许范围如下，尚未作为全部批准的压力测试范围：

- `presolve`：默认 `choose`，允许 `off/choose/on`，控制预处理策略。
  已读 MIP 代码中的多处判断只区分 `off` 和非 `off`；不能未经核对就将
  `choose` 与 `on` 算作两个行为不同的扰动等级。
- `mip_detect_symmetry`：默认 `true`，布尔开关。需要实例具备适用结构；
  代码还会依据是否存在二元变量等条件决定后续检测。
- `mip_heuristic_effort`：默认 **0.05**，允许 `[0,1]`。控制与启发式 LP
  工作量估计相关的投入判断，不能解释为实际墙钟时间的严格百分比。
- `mip_heuristic_run_feasibility_jump`：默认 `true`，布尔开关，决定是否进入
  根节点求解前的 feasibility jump 步骤。
- `mip_pscost_minreliable`：默认 **8**，接口允许整数 `[0,2147483647]`。
  它控制伪成本可信所需观测数；整数接口上限没有提供合理实验上限的依据。
- `mip_heuristic_run_rens`、`mip_heuristic_run_rins`：默认均为 `true`。
  调用受 incumbent 是否存在、搜索阶段和工作量等条件影响，适合作为后续
  机制候选，不能假定每次运行都执行过。

`mip_heuristic_effort=0` **不等于关闭所有启发式**。源码在早期主 MIP 搜索
允许额外的 10,000 次启发式 LP 迭代额度，feasibility jump 也由独立开关控制。
见 [搜索状态](../../../papers/configuration-robustness/sources/tree-search-state.txt)
的 `moreHeuristicsAllowed()` 和
[搜索流程](../../../papers/configuration-robustness/sources/tree-search.txt) 的根节点前步骤。

按已确认范围，时间、线程和精度要求固定。`mip_rel_gap`、`mip_abs_gap`、
`mip_feasibility_tolerance` 及其他精度选项不作为搜索参数；节点上限、
改进解数量上限等额外停止限制也不能用来替代约定的求解预算。

## 4. 供 M2 讨论的首轮候选空间

以下是基于上述证据提出的建议，**尚未选定为实验协议**：

**PyVRP：**先考虑邻域大小、历史长度、罚系数增加倍率、罚系数降低倍率四项。
近期上游值提供的区间候选分别是整数 `[50,60]`、整数 `[300,500]`、
实数 `[1.25,1.50]`、实数 `[0.85,0.95]`，均包含当前默认值。
这是一种有历史参照的保守范围；不覆盖更大扰动，默认值也不都处于区间中心。
把端点之间连续化或整数化仍是本项目建议，并非上游验证过的标准。

**HiGHS：**先考虑 `presolve` 的 `choose/off`、对称性检测布尔开关、
feasibility jump 布尔开关及启发式投入。前三项有明确的公开取值；
投入的完整公开域 `[0,1]` 可作为较强压力范围的候选，也可以另选较窄范围，
本轮未作选择。先不把伪成本观测数的整个整数合法域用于搜索。

两套候选空间的强度不同。即使以后采用，也只能解释各自声明空间内的结果，
不能把两种求解器的退化量直接当成同等扰动下的跨求解器排名。

参数范围、搜索成本与失败反馈、跨实例和 seed 的汇总、搜索预算及独立复测
安排，仍须在 M2 明确。本轮没有选择统一百分比扰动、最终统计量或验收阈值。

## 5. 本轮核查范围

本轮整理并重点阅读了三篇算法配置原始论文与 irace 手册，复用了已收集的 PyVRP
论文、上游变更记录和原版运行数据，并核对两个固定求解器版本的参数源码。
没有安装 SMAC3、运行参数搜索、修改求解器驱动、修改任务配置或更新台账。
归档的三个 Python 参数相关模块与已安装 PyVRP 0.14.0 的对应文件逐字节一致。

后续应先确定参数空间和反馈协议，再接入已有按功能组织的采集、驱动、
独立验证和存储模块。

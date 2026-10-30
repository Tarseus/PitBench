# Algorithmic Generalization and Adaptation 诊断实验

本文件规定下一轮实验的比较方式和实施前提。当前状态是研究设计与准备，尚未运行新的 Agent 实验，也未完成新分布、参考干预和全部场景的验证。历史 28 个完成条件不能据此扩充为完整指标覆盖。

## 比较对象和预算

优先研究 CBC、SCIP、PyVRP，沿用 release 0.0.1 为它们选定的各两个版本，共六个有固定 commit/tree 的任务。相邻版本是相关观测，不能当作六个独立算法仓库。

模型设置为 gpt-6.1-sol high/xhigh、gpt-6-luna medium、gpt-6.1-terra high/xhigh。每个任务、模型设置、实验组只启动一次新 Agent；暂不做独立 Agent 重复或挑选最优尝试。

每次沿用 standard 开发预算：4 小时、8 vCPU、32 GiB，默认开发实验并发 4。模型等待、重试、编译、数据生成、分析和开发评测均计入同一墙钟预算；镜像准备与最终隐藏 Judge 在预算外。三个实验组使用相同模型设置、工具、硬件隔离和资源限制，同时记录实际 CPU、token、请求、失败及缺失的用量信息。相同墙钟上限不等于相同实际消耗。

六个版本任务 × 五种模型设置 × 三组 × 一次尝试 = **90 次计划运行，最多 360 Agent 墙钟小时**。这不是服务器耗时或费用估算，也不包含前置验证和最终 Judge。

## 三组比较

- A / `original`：原始 Agent，原始任务开发视图。允许自行诊断、生成实验和设计条件化策略。
- B / `diverse`：原始 Agent，预先准备的多样化开发视图。
- C / `mechanism_aware`：与 B 完全相同的数据和工具，额外要求声明机制、适用条件、失效假设、算法状态、正确性不变量及可证伪的实验。

B−A 衡量提供多样化开发信息的效果。如果开发样本数量也不同，该比较包含数量效应，不能命名为纯粹的多样性效应。C−B 衡量机制感知要求在相同信息下的附加效果；C−A 只能表示整体组合效果。三组都能使用机制声明工具，只有 C 被要求使用，避免同时改变工具访问权。

记录 A/B 是否自发识别适用条件、设计反例和采取条件化策略。评价可观察的声明、代码触发条件和实验记录，不要求 Agent 提供私有思维过程。C 的结构化声明是否有内容、预测是否被支持，需要按预先定义的审阅规则检查；JSON 合法或文字较长不能证明机制理解。

## 分布设计与可获得的优化空间

先清点公开基准的完整合格库存，记录来源、许可、排除理由与同源实例组，再用系统生成设计补齐因素。不能依据候选补丁表现选择实例。保留容易的正确性回归实例和困难实例，分别报告。

CBC/SCIP 的同族层可从多维 knapsack 出发，改变 item/constraint 数、实际非零密度、系数与 profit 的相关性、capacity tightness，并单独改变 solve budget。当前生成器是稠密矩阵，且 profit 固定为权重列和；需先实现真实稀疏结构和独立 profit 分布，再验证生成后的观测特征。在 YAML 中加入未被读取的 density 参数不构成分布变化。

PyVRP 的 CVRP 层改变客户数、空间混合与簇尺度、depot 位置、需求分布及其空间相关性、容量紧张程度和时间预算。完整图 CVRP 不具有独立的输入“约束密度”轴；邻域宽度属于求解器配置实验。现有某一固定 shift 配方下的规模切片不能解释为 shift 严重程度曲线。

先做单因素变化以解释机制，再评测未见因素组合。开发与测试须包含不同参数水平或组合，不能只更换 seed。类别变化没有天然严重程度顺序；距离基准分布的坐标、归一化方式和每条曲线所控制的其他因素必须在测量前定义。

跨问题族另设困难层：CBC/SCIP 可比较 knapsack 到 scheduling/set covering；PyVRP 可比较 CVRP 到 VRPTW。后者需要对应驱动及独立时间窗验证器，不能沿用 CVRP 路线合法性检查声称完成跨变体迁移。各层分别汇报，不能让跨族失败掩盖同族结论。

分割为 evaluator 的 protocol calibration、Agent 的 diverse development、隐藏 ID、隐藏同族 shift、隐藏跨族/变体。记录实例内容 hash、生成器版本、参数、随机种子命名空间和同源组；相关变体及等价表示必须同组分割。测试水平、测试 seed、参考干预及答案不进入 Agent 视图。旧实验中已被当前设计者查看的面板只作为历史诊断，不冒充新研究的未见证据。

在观察 Agent 结果之前，用独立校准实例资格化参考源代码干预，确认每个计划计分的机会层在相同 solve budget 下存在可获得的改进。保留参考干预失败及未达资格的层，并用独立实例和反向运行顺序确认。原始求解器有 gap、BKS 更好、延长时间或调参有效，都不足以单独证明存在源代码优化空间。参考干预及其实现对 Agent 隐藏；隐藏测试不用于反复选择参考策略。未资格化的层可以保留为探索或正确性诊断，但必须明确标识。

具体因素水平、实例数量、新增 solve budget、target、容差及统计归约尚未冻结；应在数据与参考资格化后写入有 hash 的实例清单和 scoring/scenario 协议，再开始 Agent 实验。已有任务的预算、seed、原生指标和阈值保持原定义。

## 评测维度和统一入口

复用 [指标场景协议](../metrics/scenarios.md)、`ScenarioSession`、独立问题验证器和已有报告器。当前目录中的 26 个指标按以下职责组织；“要求评测”不代表所有求解器目前支持：

- 主性能：`exact_solved_coverage`、`exact_penalized_runtime_sec`、`fixed_budget_normalized_gap`。按任务类型适用，精确求解与启发式指标不混成同一收益比例。
- 搜索轨迹：`time_to_first_verified_solution_sec`、`time_to_target_sec`、`primal_integral`、`dual_progress_integral`、`search_work`。每个计分 incumbent 导出 assignment 并独立验证；目标、初始 loss、积分尺度和超时删失按冻结协议处理。无独立证书的 bound 只作诊断，搜索工作量不能替代质量收益。
- 资源与生命周期：`peak_rss_ratio`、`cpu_time_ratio`、`retained_memory_series`、`memory_growth_per_solve`、`setup_time_sec`、`setup_peak_memory`。长期增长须在同一进程和保留的 solver 对象中测量，规定 warmup、清理/reset、采样和归约。一次进程峰值不是长期增长。
- 并行：`thread_scaling_curve`、`parallel_efficiency`。确认每个构建实际支持的线程机制，在共同质量 target 下比较 wall time、CPU 和内存。独立多进程 portfolio 另作资源分配实验，不能代替原生线程扩展性。
- 增量：`sequence_solve_time_sec`、`update_overhead_sec`。cold/warm 使用相同内容 hash 的模型更新序列和逐步质量目标；warm 保留 solver 对象与其算法状态。重新启动 CLI 或仅输入外部 incumbent 不证明原地重求解。
- 正确性与压力：`independent_violation`、`numerical_failure_counts`、`boundary_pass_counts`、`operational_failure_counts`。MIP 按原模型单位检查 row/bound/integrality/objective 与错误 infeasible 状态；整数路线问题另测溢出、距离舍入和容量边界，不能称为浮点 MIP 条件数测试。
- 重复与表示：`seed_dispersion`、`representation_dispersion`、`same_seed_replay_consistency`。沿用原生 dispersion 定义；replay 固定输入、seed、线程和模式。特征关闭的隔离改动需检查固定 seed 轨迹身份。
- 泛化：`controlled_retention`，结合各分布格子的原生性能及失败率解释。不能从 aggregate improved 推出所有分布均改善。

现有 source-bound 扩展场景实现是 `HighsScenarioSession`。CBC、SCIP、PyVRP 尚缺本轮所需、经验证的统一场景接入；每个任务须明确 supported、unsupported、not applicable、failed 或 missing 及证据。PyVRP 没有 MIP dual bound 属于不适用；实现缺失不能标为不适用。适用的缺失维度不能填零、填正收益或沿用主性能结论。

开发使用同一个 `pitbench bench`，最终评估继续使用 `pitbench evaluate`，扩展场景通过 `metric_scenario_configs` 接入既有 collector；生成、构建、运行、验证、存储及报告共用功能模块。统一接口不能改变场景的算法语义。

## 机制声明与适应性检查

C 在每个非空候选补丁正式 bench 前执行：

```bash
pitbench hypothesis --plan candidate.json
pitbench bench --build --hypothesis-id candidate-id --experiment-id experiment-id
```

`candidate.json` 包含 `id`、`mechanism`、`applicability`、`failure_hypotheses`、`algorithm_state_features`、`correctness_invariants`，以及 `experiments`。每个实验含 `id`、`controlled_change`、`expected_observation`、`falsifies_if`、`metric_ids`。静态改动也需明确说明不依赖状态，而不是编造自适应机制。

声明绑定当前真实源代码 patch 的 SHA-256，并在反馈中记录声明 hash 和 experiment ID。补丁变化后须重新声明；同一补丁下声明 ID 不覆盖写入。空补丁、Base 和 dry-run 不需要声明。A/B 保留自主使用权。此入口检查记录完整性，不自动判断假设质量，也不阻止 Agent 在 shell 自行运行实验；遵循情况需结合 host trace 审阅，不能把 Agent 可写的本地文件当作可信防篡改存储。

每个适应性候选需在 evaluator 诊断病例上比较 feature-off、always-on、conditional。记录触发率、触发前后状态、决策开销、命中与失效条件，并检查同一实例内部是否随状态改变。按可观察输入特征分支和按算法状态调整应分开标记；按实例文件名、隐藏 ID 或答案分支不算算法适应。纯吞吐优化可以不改变策略，但收益需在相同决策或工作量轨迹下验证。诊断补丁需固定、保留 diff，不能测试后再回调 Agent。

## 达到上游 PR 审阅要求

按每个固定版本的真实 CI 和测试入口建立记录，先验证 Base，再验证最终候选及关闭特征的控制构建。编译成功不算测试通过；保留收集数、执行数、skip/xfail/deselect 原因、测试日志及构建 flags。检查求解器跑到的是候选构建而非系统安装的原版 binding。

已经检查的上游最新版本工作流提供了具体参照：

- [CBC 2.10.13 Linux CI](https://github.com/coin-or/Cbc/blob/releases/2.10.13/.github/workflows/linux-ci.yml) 使用 coinbrew 的 `--tests main`，README 记录 `make test`，并包含静态/共享、编译器及系统组合。本地插件目前只构建，尚未补齐 suite 调用。
- [SCIP 10.1.0 CI](https://github.com/scipopt/scip/blob/c8e5737a847219d2811f7dfbd7cc6aa5b96df684/.github/workflows/build_and_test.yml) 构建后执行 `ctest --output-on-failure`。其单元测试依赖 Criterion；仅有一般 ctest 结果不能证明单元测试已注册。本地插件目前没有执行 ctest。
- [PyVRP 0.14.1 CI](https://github.com/PyVRP/PyVRP/blob/a1a9a700e104e53ba908cf81b76a3d9f7132a43c/.github/workflows/CI.yml) 包含 debug、pytest、编译器组合、coverage 和 Valgrind smoke。本地 validation 有 pytest/检测构建，但现存 golden/轨迹测试排除需逐项审查，必要时用 feature-off 身份检查补足。

以上是对工作流的检查，未表示这轮在六个版本上已运行完这些流程。另一个版本必须独立确认兼容入口，不能直接套用最新版命令。固定 seed 行为测试应区分被允许改变的策略轨迹和必须保持的正确性契约；不能为了让优化通过而直接删除失败的测试。

候选还需通过相关回归、边界、独立可行性/目标重算、状态回滚、增量更新和诊断病例。根据实际改动选择上游已有 sanitizer、内存检查和编译器配置，记录覆盖范围；此研究不声称复刻每个上游的所有操作系统矩阵。

## 执行、报告与结论边界

先完成数据与参考资格化，再完成源代码绑定的场景及上游测试接入，最后运行 Base 预检和 90 次计划实验。正式 Agent 开发不接收隐藏评估反馈；按既有 checkpoint 机制留存补丁和捕获失败。原始模型名、provider 修订、scaffold、源码、镜像、数据、配置、运行顺序和主机身份都进入实验记录。

Base/候选及三组之间按共同实例、solver seed、预算配对，交替或按记录的随机顺序执行以控制漂移。新增主要对比、置信水平、重采样单位、层间归约及多重比较方式必须在计分计划中冻结。沿用已有任务的原生 CI 与 confirmation 规则；solver seed 不能替代独立实例，相关版本不能作为独立仓库。报告完整失败、超时、非法解、基础设施错误和缺测，保留计划分母。

复用现有自动报告，新增适配完成后输出：各模型/组的分布格子效应与 CI、预定义 shift 坐标曲线、规模和线程扩展曲线、incumbent/bound 轨迹、长期内存序列、cold/warm 对比、数值压力与失败、feature-off/always-on/conditional 消融。轨迹/严重程度不存在的数据不作插值假装完成。报告 better/equal/worse、尾部损失和去除最大影响实例后的敏感性；总体均值不能替代逐层退化检查。

本轮每条件只有一次 Agent 开发，实例/solver seed 区间描述该次实际补丁的效果，不能估计 Agent 生成方差，也不能证明某模型总体上优于另一模型。结论限于诊断试验；不因缺少 Agent 重复而擅自增加运行。

## 配置和复现

设计在 `configs/research/generalization.json`，模型矩阵在 `configs/experiments/generalization_model_matrix.yaml`，总体与待办证据在 `configs/research/generalization_populations.json`。准备入口：

```bash
.venv/bin/python -m scripts.prepare_research_cases study \
  --plan configs/research/generalization.json --root . \
  --output-dir gushengda/experiments/generalization/design-v1
```

入口生成 evaluator 的 `study.json` 和十八个任务/组公开视图，保留预算、来源 hash、90 个计划 job 及缺口；不会调用 Agent 或 solver。它是设计物化与缺口清单，始终标记 `ready_for_agent_execution: false`，尚不是可以替代原生数据、场景和计分验证器的完整执行门禁。当前 null 的人口清单、headroom、上游测试证据和 scoring 不能仅填一个 ready 字段就视为通过。

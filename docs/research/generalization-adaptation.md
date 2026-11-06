# Algorithmic Generalization and Adaptation 诊断实验

2026-11-01 更新：问题集、难度和逐条件时限已扩展为 [难度校准版](difficulty-calibration.md)，共 78 个条件、3,230 个实例，六个固定版本均有实测分层。原配置为 `configs/research/archive/generalization_v2.json`；本文下方保留原设计及历史校准记录。此前首批 Agent 已试跑并按用户要求暂停，现场及基础设施失败记录保留。

当前缩减配置为 `configs/research/generalization_latest_high_v2.json`，模型矩阵为 `configs/experiments/generalization_high_matrix.yaml`：仅比较 `gpt-6.1-sol/high` 和 `gpt-6.1-terra/high`，仅保留清单已冻结的最新版本 CBC 2.10.13、SCIP 10.1.0 和 PyVRP 0.14.1，三组各一次，共 18 个条件。开发预算、问题集、指标、seed 和逐条件求解时限保持不变；历史 90 条件配置不覆盖修改。

每个实例、每个预算的正式性能与资源评测中，三种求解器统一使用 30 个隐藏评测 seed。CBC、SCIP 保留原来的 5 个开发 seed，评测另外抽取 30 个正整数 seed 并冻结在私有存储中；PyVRP 保留既有两组各 30 个 seed。开发与评测列表互不重叠，同一任务的 Base、候选和三组共享冻结评测列表。历史任务文件保留原样，新策略只写入本轮评测视图。`pitbench bench` 默认运行单个 seed，需指定 `--all-seeds` 才覆盖可见开发 seed 网格。扩展场景诊断目前每个分布条件取 2 个预定隐藏实例、1 个 solver seed；replay 重放 3 次，不代表 3 个独立 seed。实例生成随机流、表示重标号、solver seed 和 Agent 独立重复是不同维度，本轮 Agent 独立重复仍为 1 次。

统一入口为 `scripts.run_benchmark_release --phase study-models`。机器本地配置 `config/study.local.yaml` 可从 `configs/experiments/generalization_launch.yaml` 配置，包含计划、任务视图、运行配置、CPU 池和输出目录；代理、认证路径和镜像由其引用的本地 evaluation 配置提供，不要求调用者设置环境变量。配置就绪后直接执行：

```bash
.venv/bin/python -m scripts.run_benchmark_release
```

默认后台执行，日志和进程回执写入输出目录；`--prepare-only` 只冻结清单，`--foreground` 用于前台运行。每个开发结束的最终补丁通过现有独立 evaluator 进入隐藏评测，使用独立 CPU 池；无需新一次 Agent 调用。全部条件处理完后使用原生报告入口自动汇总，包括失败和缺测。实际源代码 API 构建和指标采集随候选执行，尚未资格化的优化空间在结果中标记为待确认，不能提前宣称能力结论。

经用户确认，先前暂停的全部运行保留为中断预运行，不并入正式样本。新的 18 条件统一从头开始，每个正式条件仅一次 Agent 开发，不根据预运行补丁效果选择重启对象。

本文件规定下一轮实验的比较方式和实施前提。截至 2026-10-31，已实现数据构造、三种求解器的场景接口与基础验证入口，构造并独立验证了 2,150 个新实例。六个固定版本的首轮参考干预校准产生了 76,800 条运行记录，其中有未求解和采集失败；集合打包与新增集合覆盖已修复驱动后完成有效校准。90 次新 Agent 实验尚未启动。实现验证与研究结果分别记录。历史 28 个完成条件不能据此扩充为完整指标覆盖。

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

CBC/SCIP 的同族层可从多维 knapsack 出发，改变 item/constraint 数、实际非零密度、系数与 profit 的相关性、capacity tightness，并单独改变 solve budget。生成器已支持真实非零密度、容量紧张程度及多种 profit 分布，并记录观测特征。带加权列和 profit 的 planted 模型可独立验证最优性；独立 profit 模型不沿用该证明。在 YAML 中加入未被读取的 density 参数仍不构成分布变化，构造入口会检查声明的观测特征。

PyVRP 的 CVRP 层改变客户数、空间混合与簇尺度、depot 位置、需求分布及其空间相关性、容量紧张程度和时间预算。完整图 CVRP 不具有独立的输入“约束密度”轴；邻域宽度属于求解器配置实验。现有某一固定 shift 配方下的规模切片不能解释为 shift 严重程度曲线。

先做单因素变化以解释机制，再评测未见因素组合。开发与测试须包含不同参数水平或组合，不能只更换 seed。类别变化没有天然严重程度顺序；距离基准分布的坐标、归一化方式和每条曲线所控制的其他因素必须在测量前定义。

跨问题族另设困难层：CBC/SCIP 从 knapsack 迁移到 single-machine scheduling、set packing、set covering；PyVRP 从 CVRP 迁移到 VRPTW。以上问题类型已接入通用生成、执行和独立验证入口。集合覆盖为二元最小化模型，采用互不相交的必覆盖行给出下界，并用满足全部额外覆盖行的原模型解达到该下界；不能把集合打包的互斥约束改名当作集合覆盖。VRPTW 独立验证时间窗、服务时间和车队上限。各层分别汇报，不能让跨族失败掩盖同族结论。

`populations-v3` 包含校准 520、开发 70、隐藏 ID 420、隐藏同族 shift 840、隐藏跨族/变体 300 个实例。相对 v1 增加集合覆盖的两个规模条件：80 变量/50 约束和 160 变量/100 约束，额外覆盖行密度均为 0.2；共增加 20 个校准与 60 个隐藏实例，不新增开发数据。v3 将新增模型写为 CBC/SCIP 均接受的标准 MPS 字段宽度，修复 CBC 对首条记录的格式误判。原 2,070 个实例的 SHA-256 均保持一致。额外行密度与整体矩阵密度分别记录。

分割为 evaluator 的 protocol calibration、Agent 的 diverse development、隐藏 ID、隐藏同族 shift、隐藏跨族/变体。记录实例内容 hash、生成器版本、参数、随机种子命名空间和同源组；相关变体及等价表示必须同组分割。测试水平、测试 seed、参考干预及答案不进入 Agent 视图。旧实验中已被当前设计者查看的面板只作为历史诊断，不冒充新研究的未见证据。

在观察 Agent 结果之前，用独立校准实例资格化参考源代码干预，确认每个计划计分的机会层在相同 solve budget 下存在可获得的改进。保留参考干预失败及未达资格的层，并用独立实例和反向运行顺序确认。原始求解器有 gap、BKS 更好、延长时间或调参有效，都不足以单独证明存在源代码优化空间。参考干预及其实现对 Agent 隐藏；隐藏测试不用于反复选择参考策略。未资格化的层可以保留为探索或正确性诊断，但必须明确标识。

因素水平、实例数量和独立 seed 命名空间已写入 `configs/research/archive/generalization_generation.json`，实例清单带内容 hash。参考干预固定为历史已完成 sol-high 运行的补丁，在新测量前选定，不按新结果挑选。首轮原生分类在 CBC 两版本分别为 improved 6 / inconclusive 19 / regressed 2 和 improved 7 / inconclusive 20；SCIP 分别为 improved 10 / inconclusive 17 和 improved 11 / inconclusive 16；PyVRP 两版本均为 improved 17 / inconclusive 6。这些是保存的原生统计输出，不等于研究层资格化。

逐条审计发现首轮 69,107 条通过独立解校验，6,093 条标记 no_solution，1,600 条集合打包记录在调用求解器前因变量名提取器拒绝非背包 MPS 而失败。首次追加的 1,600 条集合覆盖记录也受同一问题影响。旧驱动还只保存声明最优的解，因而 no_solution 不能解释为求解器从未找到 incumbent。现已从原模型提取通用 MPS 变量，并保留时间限制下的可行 incumbent；exact 指标仍按原来的最优声明、独立最优值和 PAR2 规则计分，未把可行解当作已求解。原始失败记录保留，受影响条件在相同求解预算、seed、补丁下重跑；批次并发和 CPU 隔离写入各次 command.json。报告附有效解数、状态与错误计数；双方全部采集失败的面板不画性能效应；不同批次分别画曲线，不合并重复坐标。

修复后，四个固定 MIP 版本的集合打包、集合覆盖各取得 1,600 条通过独立验证的记录。qualification 原生分类如下：集合打包在 CBC 2.10.12 两条件均 regressed，CBC 2.10.13 两条件均 inconclusive，SCIP 10.0.3 两条件均 improved，SCIP 10.1.0 为一个 improved、一个 inconclusive；集合覆盖在 CBC 2.10.13 的 10 组条件为 regressed，其余七个版本/分布条件均 inconclusive。这是固定历史参考补丁在各分布上的测量，不是新 Agent 效果。调度和 VRPTW 的首轮 qualification 条件尚未确认改善。独立校准分割、反向顺序确认等完整 headroom 资格化仍须另行验证，不能仅凭上述分类声称泛化通过。

本次集合覆盖八个版本/分布条件在主预算下的平均 PAR2 时间差绝对值均小于 0.5 毫秒。这些输入已验证适配与迁移诊断可运行，尚未建立有意义的源代码优化空间；不能把统计分类直接解释为实质算法收益。

新增 solve budget、target、容差、场景归约及跨分布计分规则仍须在正式 Agent 实验前冻结到 scoring/scenario 协议。已有任务的预算、seed、原生指标和阈值保持原定义；首轮校准保留 MIP 的 5 个和 PyVRP 的 30 个既定求解 seed。

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

已新增并在六个固定版本上执行原生场景接口资格化：`CbcScenarioSession`、`ScipScenarioSession`、`PyVRPScenarioSession`，与既有 `HighsScenarioSession` 共用采集与验证器。CBC 增量接口复用 LP basis 并重置 MIP 树；SCIP 使用原生重优化。PyVRP 没有原生线程数或修改 ProblemData 后原地重求解的 API。每个任务须明确 supported、unsupported、not applicable、failed 或 missing 及证据。PyVRP 没有 MIP dual bound 属于不适用；实现缺失不能标为不适用。适用的缺失维度不能填零、填正收益或沿用主性能结论。

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

按每个固定版本的真实测试入口建立基础门槛，先验证 Base，再验证最终候选及关闭特征的控制构建。依据用户后续要求，上游测试采用构建、相关单元测试和独立解合法性检查；耗时的全量优化回归与示例程序不作为默认门槛。编译成功不算测试通过；保留选取范围、执行数、skip/xfail/deselect 原因、测试日志及构建 flags。泛化、轨迹、长期内存、并行、增量和数值压力的研究评测保持独立、完整。检查求解器跑到的是候选构建而非系统安装的原版 binding。

已经检查的上游最新版本工作流提供了具体参照：

- [CBC 2.10.13 Linux CI](https://github.com/coin-or/Cbc/blob/releases/2.10.13/.github/workflows/linux-ci.yml) 使用 coinbrew 的 `--tests main`，README 记录 `make test`。插件已接入实际测试调用，两个固定版本均通过基础验证。
- [SCIP 10.1.0 CI](https://github.com/scipopt/scip/blob/c8e5737a847219d2811f7dfbd7cc6aa5b96df684/.github/workflows/build_and_test.yml) 构建后执行 `ctest --output-on-failure`，单元测试依赖 Criterion。插件现按 `^unittest-` 同时筛选注册清单与执行，零测试时报错；10.0.3 的 288 项、10.1.0 的 290 项分别在约 63/69 秒通过。
- [PyVRP 0.14.1 CI](https://github.com/PyVRP/PyVRP/blob/a1a9a700e104e53ba908cf81b76a3d9f7132a43c/.github/workflows/CI.yml) 包含 debug、pytest、编译器组合、coverage 和 Valgrind smoke。0.13.4 在兼容环境中 844 项全通过；0.14.1 的 debugoptimized/检测构建 966 项通过，1 项仅适用 Debug 日志的 golden 测试不适用。完整 Debug 构建存在 fmt 编译兼容问题，此结果不声称完整 Debug 矩阵通过。

SCIP 先前一小时超时来自整个串行 CTest 的外层上限：两个版本前 1,120 项均通过，累计约 3,547/3,398 秒，最慢单项约 57 秒；不是单项卡住一小时。之后续跑被 g52 主机重启中断。依用户要求已采用上述相关单元测试门槛，不重启全量优化/示例回归。固定 seed 行为测试应区分被允许改变的策略轨迹和必须保持的正确性契约；不能为了让优化通过而直接删除失败的测试。上述记录是 Base/控制构建验证，正式候选仍须执行相应检查。

候选还需通过相关回归、边界、独立可行性/目标重算、状态回滚、增量更新和诊断病例。根据实际改动选择上游已有 sanitizer、内存检查和编译器配置，记录覆盖范围；此研究不声称复刻每个上游的所有操作系统矩阵。

## 执行、报告与结论边界

先完成数据与参考资格化，再完成源代码绑定的场景及上游测试接入，最后运行 Base 预检和 90 次计划实验。正式 Agent 开发不接收隐藏评估反馈；按既有 checkpoint 机制留存补丁和捕获失败。原始模型名、provider 修订、scaffold、源码、镜像、数据、配置、运行顺序和主机身份都进入实验记录。

Base/候选及三组之间按共同实例、solver seed、预算配对，交替或按记录的随机顺序执行以控制漂移。新增主要对比、置信水平、重采样单位、层间归约及多重比较方式必须在计分计划中冻结。沿用已有任务的原生 CI 与 confirmation 规则；solver seed 不能替代独立实例，相关版本不能作为独立仓库。报告完整失败、超时、非法解、基础设施错误和缺测，保留计划分母。

复用现有自动报告，新增适配完成后输出：各模型/组的分布格子效应与 CI、预定义 shift 坐标曲线、规模和线程扩展曲线、incumbent/bound 轨迹、长期内存序列、cold/warm 对比、数值压力与失败、feature-off/always-on/conditional 消融。轨迹/严重程度不存在的数据不作插值假装完成。报告 better/equal/worse、尾部损失和去除最大影响实例后的敏感性；总体均值不能替代逐层退化检查。

本轮每条件只有一次 Agent 开发，实例/solver seed 区间描述该次实际补丁的效果，不能估计 Agent 生成方差，也不能证明某模型总体上优于另一模型。结论限于诊断试验；不因缺少 Agent 重复而擅自增加运行。

## 配置和复现

设计在 `configs/research/archive/generalization.json`，模型矩阵在 `configs/experiments/archive/generalization_model_matrix.yaml`，总体与待办证据在 `configs/research/archive/generalization_populations.json`。准备入口：

```bash
.venv/bin/python -m scripts.prepare_research_cases study \
  --plan configs/research/archive/generalization.json --root . \
  --output-dir gushengda/experiments/generalization/design-v1
```

入口生成 evaluator 的 `study.json` 和十八个任务/组公开视图，保留预算、来源 hash、90 个计划 job 及缺口；不会调用 Agent 或 solver。它是设计物化与缺口清单，始终标记 `ready_for_agent_execution: false`，尚不是可以替代原生数据、场景和计分验证器的完整执行门禁。缺失文件以及尚未资格化的 headroom、场景和 scoring 不能仅填一个 ready 字段就视为通过。

生成、参考值与视图依次使用同一脚本的 `populations`、`references`、`views` 子命令；求解校准使用 `headroom`。新增问题类型可通过重复的 `--condition ID` 仅追加相应条件，保留已有测量。`pitbench report-distributions SOURCE --generation-specification configs/research/archive/generalization_generation.json --output OUTPUT` 从已保存的原生 CI 生成分布曲线和 SVG/PNG/PDF；参考校准报告明确标记为 evaluator reference，不混入新 Agent 对照结果。

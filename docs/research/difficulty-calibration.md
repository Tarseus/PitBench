# 结构、规模与难度分层

本协议用于 CBC、SCIP、PyVRP 的新诊断问题集。原 release 任务的源码版本、主面板、指标、预算、seed 和已有结果保留。新增面板采用单独的分布定义、参考解与时限；本协议不表示 90 次 Agent 对照或全部科学指标已经完成。

## 设计参照

- [NeurIPS 2021 ML4CO](https://github.com/ds4dm/ml4co-competition) 的 DATA、TASKS、PLATFORM 说明分别记录公开/隐藏划分、预计算参考界、primal/dual/config 指标及计时起点。求解提前结束后仍需处理剩余积分区间，单核平台的秒数需要在目标机器上重新校准。
- [PyVRP/Instances](https://github.com/PyVRP/Instances) 提供 CVRPLIB X/XXL，以及 Solomon、Gehring–Homberger 时间窗分类。本次清点 110 个 CVRP、356 个 VRPTW 文件及其解文件。CVRP 采用最近整数距离，VRPTW 文档规定 DIMACS 的一位小数截断；这些公开文件在此提供设计参照，未冒充本轮新生成的隐藏测试。

参照来源的 commit、说明文件 hash、完整库存及舍入规则写入实验目录。难度标签由本项目实测产生，未照搬竞赛平台上的时限或难度标签。

## 控制结构与规模

生成定义为 `configs/research/generalization_generation_v2.yaml`，新总体为 78 个条件、3,230 个实例。角色分别为校准 780、共享多样化开发 110、隐藏 ID 660、隐藏规模 360、隐藏结构 780、隐藏结构与规模组合 540。

MIP 使用独立利润背包、非预置最优解的随机加权覆盖、随机独立集、设施选址、固定费用流。设施选址和网络流包含整数变量与连续变量耦合，具有等式、混合符号约束；不强制目标系数等于约束系数列和。实际变量数、连续/整数变量数、等式与非零密度均从原模型解析。背包与覆盖在开发中出现，独立集、选址和网络流保留未见结构层。

路由采用 uniform、clustered、mixed 空间分布及不同时间窗宽度。mixed CVRP 是匹配的无时间窗对照，避免同时改变空间分布和时间窗。6、12、50、100 客户提供小规模梯度；250、750 客户及原始公开开发集提供开发覆盖；2,000、4,000 客户构成隐藏外推层。

`audit` 同时检查新开发集和所有组原有开发实例，观测上限为 MIP 120 个变量、路由 978 个客户。规模外推至少达到所有可见开发实例上限的两倍；仅结构变化的条件保持在可见规模范围内。角色与条件使用独立命名 seed 流，内容 hash、参数与同源组保留。先前 2,210 个新设计实例以及历史问题集未被覆盖写入。

## 难度与时限的实际校准

统一入口为 `scripts.collect_reference_solutions`，经 g52 跳转到 10.10.100.40–47。每次 solver 搜索固定单核，同一实例的版本、seed 和预算组合分在同一机器，记录 CPU 绑定、构建、包版本及运行负载。并发属于调度资源，不改变单次求解时限。部署包包含代码、指定求解器及生成数据，凭据只保留在本地。

探索预算为 0.1、1、5、20、60 秒，先覆盖各条件的 10 个校准实例，再搜索独立参考值。CBC、SCIP 使用四个已固定 commit 的 Release 构建；两版 PyVRP 分别加载对应版本。CBC 的零 seed 会取时钟随机数，因此保留零 seed 试跑诊断，预算选择使用正 seed 的确认批次。

每个条件的校准实例按事先确定的序号分为 discovery 0–4、qualification 5–9。MIP 目标为固定求解器返回最优且其原模型可行解与参考值吻合；校准目标匹配沿用原模型验证器的 1e-6 精度，原有 exact 指标的严格相等规则不变。路由目标为相对已验证参考值的 gap 不超过 1%。这些是难度探测坐标，不是 Agent 改善的显著性标准。

预算选择使用 discovery 中达到目标的 90% 覆盖要求，取首次满足的探测档位；该分位覆盖要求不表示置信水平。qualification 原样检验该选择并报告命中数，不根据 qualification 重新调参。未命中最大探测档位的条件保留为右删失挑战层，同时保留可行性及解质量变化，不能当成已求至最优或缺测。

新增面板保留原任务的主预算作为公共固定时间坐标，并单独声明校准后的主预算。跨规模的固定时间比较采用共同坐标；采用各自校准预算的结果另行报告。新预算只应用于新增面板，历史主面板保持原定义。

难度分为 regression、short、medium、long 和 censored challenge，完整的每条件选择及 qualification 计数由 `difficulty` 输出。若实测某一问题族缺少低难度层，先补充系统生成的小规模条件再冻结；新增条件及其理由保留版本记录。小规模层的 6/12 客户 CVRP 使用独立子集动态规划，6 客户时间窗使用完整顺序与路线分割枚举；未完成枚举时只保留可行解，不声称最优。

## 参考值、异常与计分接口

每个回收解都在本地重新检查原模型约束与目标。外部求解器的最优返回及 bound 作为来源记录，未独立重放其搜索证明；未知最优值保留为 BKS。存在数值不一致时不强行升级为严格 exact 目标。已知 BKS 的 MIP 面板使用现有固定预算解质量估计器，不能以 BKS 代替独立最优值计算 exact solved coverage。

原生状态、超时、非法解、collector 失败和缺测分别保留。短预算下 CBC 的预处理“不可行或无界”消息与输出的 LP 松弛不能当作合法整数 incumbent，也不能自动改写成正常超时；原始输出和诊断注释保留。

新增面板的预算进入统一 Judge、expected run grid、性能和资源报告。seed 分散度仍使用原生估计器，新增面板按自己的协议与预算记录，主面板结果不合并改写。B/C 开发文件完全一致，三组共享隐藏数据。

构造、检查和导入入口为同一 `scripts.prepare_research_cases` 的 `populations`、`audit`、`difficulty`、`import-references`、`views` 子命令。`views --difficulty FILE` 使用冻结的实测预算。正式 Agent 启动前仍需完成场景与 headroom 的完整资格化；本协议不以参考搜索更长或 BKS 更好替代源代码优化空间的证据。

## 完成的校准结果

四个批次共处理 17,420 个计划任务：17,030 次实际求解，390 个已知最优后的重复参考搜索按预定规则跳过。所有计划记录已回收，没有缺测。原 CBC 零 seed 试跑的 12 条解析异常和正 seed 批次的 11 条原生状态/松弛输出诊断均保留；后者有原始输出及独立验证记录，不作为合法整数 incumbent。

3,230 个输入全部有通过原模型验证的参考值。严格导入的 MIP 最优目标 1,006 个，小规模路由独立精确参考 390 个；其余为 MIP BKS 214 个、路由可行参考 1,620 个。外部求解器另有少量最优返回因浮点目标/bound 的严格一致性要求保守地保留为 BKS。完整来源与原始数值仍在参考记录中。

六个固定版本合计形成 216 个版本/分布校准单元，对应 10,800 条规范校准记录。难度为 regression 54、short 38、medium 22、long 42、censored challenge 60；每个版本均覆盖五档。非删失层的独立 qualification 命中 745/780；挑战层完整保留，未伪装成目标已命中。此计数说明校准分档与时限的表现，不是 Agent 成功率或模型能力排名。

各条件最终预算为 0.1、1、5、20、60 秒中的实测选择。公开配置为 `configs/research/generalization_latest_high_v2.json`、`generalization_populations_v2.json` 和 `generalization_difficulty_v2.yaml`。实验目录 `gushengda/experiments/generalization/difficulty-v2` 中保留 `execution-audit.json`、`difficulty-final.json`、`difficulty-summary.json`、`reference-summary.json` 及 `population-freeze.json`，分别记录完整分母、逐条件预算、分层汇总、参考种类和冻结 hash。

上述数据校准完成时的代码完整单元验证为 830 passed、18 deselected；固定版本的真实求解范围为上述四个批次。后续 Agent 试跑已按用户要求暂停，见 generalization-adaptation.md；未重复全量上游测试矩阵。

新环境可按 `configs/experiments/generalization_canonical_v2.yaml` 与 `generalization_canonical_remote_v2.yaml` 重建统一流程；实际四批次配置保留原貌。规范配置直接使用最终总体、CBC 正 seed、两版 PyVRP 和独立精确/长参考阶段，省去已定位问题的原试跑。配置中的运行时、wheel、原生构建及私有 inventory 路径需要提供对应资产，构建来源与 hash 均有记录；凭据文件不进入代码或部署包。本次未重复执行规范配置。

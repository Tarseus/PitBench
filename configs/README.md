# 配置入口

配置按用途维护。求解器版本固定在 `benchmarks/` 和 `tasks/`；实例定义、参考干预分别在 `instance_sets/` 和 `references/`。

## 当前泛化对照

- `research/generalization_latest_high_v2.json`：当前唯一的 Agent 对照方案，三个最新版求解器、Sol/Terra high、三组各一次，共 18 个条件；补充面板使用 5 个配对隐藏 seed 和校准主预算，原始主面板保留 30 个 seed，取消自动复测。
- `experiments/generalization_high_matrix.yaml`：上述方案的两个模型设置。
- `research/generalization_generation_v2.yaml`、`generalization_populations_v2.json`：问题生成定义及总体索引。
- `research/generalization_difficulty_v2.yaml`：基于独立校准实例选择求解预算。
- `research/generalization_scenarios_v2.yaml`：轨迹、内存、线程、增量、数值及重放诊断的冻结协议。
- `research/scenario_qualification.yaml`：API 语义检查用的小型解析病例，不是研究结果。

运行配置模板为 `experiments/generalization_launch.yaml`。在执行机器准备 `config/study.local.yaml` 后，一条 `.venv/bin/python -m scripts.run_benchmark_release` 即可后台启动开发、独立隐藏评测和最终原生报告，无需手动设置环境变量。`--prepare-only` 仅验证和冻结计划，不启动模型。未经优化空间资格化的分布保留待确认标记，详见 [研究说明](../docs/research/generalization-adaptation.md)。

## 数据准备与通用工具

`experiments/generalization_canonical_v2.yaml` 与 `generalization_canonical_remote_v2.yaml` 是最终问题集的统一参考搜索和难度校准配置。它们包含六个源版本以保留数据校准依据，不会增加当前方案的 Agent 条件数。

其余生成、参考搜索、配置鲁棒性及通用模型矩阵继续供各自命令使用。`research/budgets.json` 是共享开发预算目录；`protocol.json`、`pilot.json` 保留 Research Case 构造接口的定义。构造结果记录在 [pilot-construction.json](../docs/research/records/pilot-construction.json)，不再混放于运行配置中。

## 历史配置

`research/archive/` 保存被替代的初版和六版本、五模型方案，以及历史总体和生成定义。历史生成文件使用实际内容对应的 `.json` 后缀。

`experiments/archive/` 保存对应五模型矩阵，以及已经完成的初始参考、正 seed 确认、版本兼容和小规模补充批次配置。这些文件用于解释与重放旧记录，不作为当前默认运行入口。

归档仅调整维护位置及仓库内引用，保留预算、seed、版本和变换参数。原始配置另存于本地实验记录，冻结实验回执、失败记录和已部署快照均保持原貌；旧回执中的路径和 hash 按当时快照解释。凭据、私有实例、二进制和实验产物继续按 `.gitignore` 管理。

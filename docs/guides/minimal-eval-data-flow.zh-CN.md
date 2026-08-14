# 最小 Eval 流程与数据流：nop → oracle → Claude Code

这份文档记录当前仓库中最容易观察的端到端 eval 流程。目标不是跑完整
benchmark，而是用一个 FormulaCode task 看清楚：配置如何进入 CLI、task 如何被
装入 Docker、三个 agent 如何执行，以及结果最终写到哪里。

## 1. 当前工作区的最小配置

当前 `examples/config.json` 依次定义了三个 agent：

1. `nop:nop`：不修改代码，用来得到未优化基线。
2. `oracle:oracle`：执行 task 自带的 `solution.sh` 或 `solution.yaml`。
3. `claude-code:claude-haiku-4-5-20251001`：在 task 容器内安装 Claude Code，
   再让 Haiku 完成 task。

Claude Code 条目还设置了：

```json
{
  "base_url": "https://api.buzzgw.com",
  "version": "latest"
}
```

因此当前链路会从项目根目录的 `.env` 读取 `ANTHROPIC_API_KEY`，把
`https://api.buzzgw.com` 作为 Anthropic API base URL，并在容器中安装最新版
`@anthropic-ai/claude-code`。`.env` 已被 `.gitignore` 忽略，不应提交。

## 2. 运行前检查

在项目根目录执行：

```bash
uv sync --group dev
docker info
test -n "$ANTHROPIC_API_KEY" || grep -q '^ANTHROPIC_API_KEY=' .env
```

说明：

- eval 本身在 Docker 容器里运行；宿主机上是否已经安装 `claude` 不是必要条件。
- 容器首次构建以及容器内安装 Claude Code 都需要网络。
- `fc-eval` 会自动调用 `dotenv.load_dotenv()`，所以把 key 放在项目根目录 `.env`
  即可，不要求事先 `source .env`。
- 不要打印 `.env`，也不要把 key 写进 JSON 配置或命令行。

## 3. 一条命令观察完整流水线

仓库当前已经缓存了 `formulacode/head`，其中包含
`networkx_networkx_1`。用一个 task、单并发和直播日志最容易观察：

```bash
RUN_ID="flow-demo-$(date +%Y%m%d-%H%M%S)"

uv run fc-eval run \
  --dataset formulacode \
  --config examples/config.json \
  --task-id networkx_networkx_1 \
  --n-concurrent 1 \
  --livestream \
  --run-id "$RUN_ID"
```

如果只是验证 Docker、数据集和 test harness，不调用任何 LLM，先运行：

```bash
uv run fc-eval run \
  --dataset formulacode \
  --agent nop \
  --model nop \
  --task-id networkx_networkx_1 \
  --n-concurrent 1
```

## 4. 真实数据流

```text
.env + examples/config.json + CLI 参数
                  │
                  ▼
fceval/cli/fceval/runs.py
  - 读取 dataset、task-id 和三个 agent 条目
  - 构造 Harness
                  │
                  ▼
Dataset(name="formulacode", version="head")
  - registry 数据缓存到 ~/.cache/fc-eval/formulacode/head/
  - task-id 筛选出 networkx_networkx_1/
  - task 目录提供 task.yaml、Dockerfile、run-setup.sh、
    run-tests.sh、tests/ 和 solution.sh
                  │
                  ▼
Harness.run()
  - 先写 runs/<run-id>/run_metadata.json
  - 再写 runs/<run-id>/fc.lock
  - 为选中的 task 创建 TrialHandler 和 Docker terminal
                  │
                  ▼
同一个 task 容器内依次执行
  1. nop         → 不改代码          → run-tests.sh → parser → agent-1 results
  2. oracle      → 执行 solution.sh  → run-tests.sh → parser → agent-2 results
  3. claude-code → 安装并运行 Claude → run-tests.sh → parser → agent-3 results
                  │
                  ▼
合并结果
  - 每个 agent 的 sub-trial 保留自己的 results.json 和日志
  - 聚合 trial 按顺序深度合并，冲突字段偏向后执行的 agent
  - token 和 cost 对所有 agent 求和
  - run 级 results.json 再聚合所有 task/attempt
```

对应的主要代码入口：

- CLI 与配置解析：`fceval/cli/fceval/runs.py::create`
- dataset 加载和缓存：`fceval/dataset/dataset.py::Dataset`
- 总调度：`fceval/harness/harness.py::Harness.run`
- 多 agent 顺序执行：`Harness._run_multi_agent_trial`
- 单 agent 的 setup/agent/test/parse：`Harness._run_single_agent_trial_terminal`
- task 与输出路径建模：`fceval/handlers/trial_handler.py`
- agent 类型映射：`fceval/agents/agent_factory.py`
- Claude Code 环境与命令：
  `fceval/agents/installed_agents/claude_code/claude_code_agent.py`
- 容器内 Claude Code 安装：
  `fceval/agents/installed_agents/claude_code/claude-code-setup.sh.j2`

## 5. 一个必须知道的语义：三个 agent 共享容器状态

当前 `--config examples/config.json` 走的是 multi-agent 模式。对同一个 task，
`Harness._run_multi_agent_trial` 只启动一个 terminal/container，然后让三个 agent
依次在其中运行；本地模式不会在 agent 之间重建或重置容器。

这意味着：

- `nop` 看到原始仓库。
- `oracle` 在原始仓库上应用标准答案。
- `claude-code` 随后会看到 oracle 已经留下的代码状态。

所以这条命令适合观察数据流和产物，但不能被解释为三个 agent 的独立、公平对比。
若要独立评测，应使用三个不同的 run，让每个 run 各自创建容器：

```bash
uv run fc-eval run --dataset formulacode --agent nop --model nop \
  --task-id networkx_networkx_1 --n-concurrent 1 --run-id flow-nop

uv run fc-eval run --dataset formulacode --agent oracle --model oracle \
  --task-id networkx_networkx_1 --n-concurrent 1 --run-id flow-oracle

uv run fc-eval run --dataset formulacode \
  --agent claude-code --model claude-haiku-4-5-20251001 \
  --agent-kwarg base_url=https://api.buzzgw.com \
  --agent-kwarg version=latest \
  --task-id networkx_networkx_1 --n-concurrent 1 --run-id flow-claude
```

重复运行时换一个 `--run-id`；相同 ID 会触发 resume 逻辑，而不是创建全新实验。

## 6. 关键结果落盘位置

一次 multi-agent run 完成后，先看：

```bash
find "runs/$RUN_ID" -maxdepth 4 -type f | sort
```

最重要的文件如下：

```text
runs/<run-id>/
├── fc.lock
├── run_metadata.json
├── run.log
├── results.json
└── networkx_networkx_1/
    ├── <trial-name>/
    │   └── results.json                 # 三个 agent 的合并结果
    ├── <trial-name>.agent-1-nop/
    │   ├── results.json                 # nop 独立结果
    │   ├── commands.txt
    │   ├── panes/{pre-agent,post-agent,post-test}.txt
    │   └── sessions/*.cast
    ├── <trial-name>.agent-2-oracle/
    │   └── ...                          # oracle 独立产物
    └── <trial-name>.agent-3-claude-code/
        ├── results.json                 # Claude 独立结果、token、cost
        ├── commands.txt                 # 发到终端的命令历史
        ├── panes/post-agent.txt         # Claude 执行后的终端状态
        ├── panes/post-test.txt          # 测试原始输出，排错最关键
        ├── sessions/*.cast              # asciinema 终端录制
        └── agent-logs/                   # agent 轨迹（如果 agent 产生）
```

推荐按这个顺序阅读结果：

1. `run_metadata.json`：这次究竟跑了哪个 dataset、task、agent/model。
2. 顶层 `results.json`：run 总结、成功率、总 cost、每个 task 的合并结果。
3. 三个 `.agent-N-*` 目录内的 `results.json`：逐 agent 对比。
4. 某个 agent 的 `panes/post-test.txt`：理解测试或 parser 失败的原始依据。
5. `commands.txt` 和 `sessions/*.cast`：还原 agent 实际执行过程。
6. `fc.lock`：精确复现实验或理解 resume 判断使用的锁定配置。

快速提取关键字段（需要 `jq`）：

```bash
jq '{run_id,dataset_name,dataset_version,agent_name,model_name,task_ids,accuracy}' \
  "runs/$RUN_ID/run_metadata.json"

jq '{accuracy,n_resolved,n_unresolved,total_cost,cost_by_agent_model}' \
  "runs/$RUN_ID/results.json"

find "runs/$RUN_ID/networkx_networkx_1" -path '*/results.json' -print
```

如果希望每次实验只先打开一个文件，可以在 eval 结束后生成一个只读摘要：

```bash
SUMMARY="runs/$RUN_ID/key-results.txt"

{
  printf '%s\n' '=== RUN METADATA ==='
  jq '{run_id,dataset_name,dataset_version,agent_name,model_name,task_ids,accuracy}' \
    "runs/$RUN_ID/run_metadata.json"

  printf '%s\n' '=== RUN RESULT ==='
  jq '{accuracy,n_resolved,n_unresolved,total_cost,cost_by_agent_model}' \
    "runs/$RUN_ID/results.json"

  printf '%s\n' '=== PER-AGENT RESULT FILES ==='
  find "runs/$RUN_ID/networkx_networkx_1" -path '*/results.json' -print | sort
} > "$SUMMARY"

printf 'Summary written to %s\n' "$SUMMARY"
```

`key-results.txt` 是导航摘要，原始事实仍以各级 `results.json`、pane 输出和录制
文件为准。

## 7. 放到自己的 private GitHub 仓库

可以。仓库根目录的 `LICENSE` 是 BSD 3-Clause，允许复制和修改，但需要保留版权
声明、许可证条款和免责声明。推荐保留官方仓库为 `upstream`，把自己的 private
仓库设为新的 `origin`。

先在 GitHub 创建一个**空的 private 仓库**（不要自动添加 README、LICENSE 或
`.gitignore`），然后执行：

```bash
git remote rename origin upstream
git remote add origin git@github.com:<your-account>/fc-eval-explore.git
git push -u origin main
```

以后同步官方代码：

```bash
git fetch upstream
git merge upstream/main
```

当前 `.gitignore` 已排除 `.env`、`runs/`、`dataset/*` 和 `data/`，所以 API key、
实验结果及大体积 dataset 默认不会被推送。`~/.cache/fc-eval/` 位于仓库之外，
同样不会推送。真正 push 前仍建议执行：

```bash
git status --short
git diff --cached
git remote -v
```

注意：未提交的工作区修改不会随第一次 `git push` 上传。可以先推官方历史，再新建
探索分支，把自己的配置、代码和这份数据流笔记按主题提交：

```bash
git switch -c explore/eval-flow
git add docs/guides/minimal-eval-data-flow.zh-CN.md
git commit -m "docs: document minimal eval data flow"
git push -u origin explore/eval-flow
```

不要使用 `git add -A` 盲目提交；先确认没有任何 secret 或不希望上传的实验产物。

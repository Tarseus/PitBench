# Configuration Robustness Evaluation Environment Test Record

日期：2026-09-18

本记录保留一次跨求解器 Antigravity 评估启动过程中发现的环境问题、根因、
修复方式和后续前置检查。它记录的是评估基础设施证据，不改变任何 metric
定义、预算、seed、变换或统计量。

## 测试范围

评估任务：

- pyvrp_v0_12_2
- pyvrp_v0_13_0
- pyvrp_v0_13_4
- pyvrp_v0_14_0
- vroom_v1_15_0
- highs_v1_15_1
- choco_v6_0_1
- ortools_cp_sat_solve_v9_15

Agent：Antigravity；模型：gemini-3.8-flash-high；并发数：1；
livestream：启用。

## 发现的问题与根因

1. 多行 Shell 粘贴曾把 task ID 当作独立命令，造成 TASK_IDS 缺失和
   agy 参数解析错误。评估命令必须作为一条完整命令执行。
2. pyvrp_v0_14_0 的本地 release snapshot 与任务声明的 GitHub commit
   不一致，产生 repository release identity mismatch。正式评估不能依赖
   未验证的本地 release snapshot。
3. VROOM 曾引用不存在的 private asset，且曾使用不能作为 Dockerfile
   FROM 的 digest-only agent image reference。
4. VROOM、HiGHS、Choco 的实际配置 agent image 缺少 livestream 所需的
   asciinema（部分 image 也缺少 tmux）。修改 solver Dockerfile 后仍
   必须更新实际配置的基础 image，并重建 task wrapper image。
5. GitHub 直连曾出现 TLS early EOF、低速超时和无法连接。仓库 snapshot
   fetch 已改为 HTTP/1.1、--depth=1 和有限重试；当前机器最终使用
   Clash Verge 127.0.0.1:7897 作为 Git proxy。
6. /home/gsd/.cache/uv 的只读错误来自受限执行沙箱，不是 PitBench 评估
   逻辑问题；正常用户终端不应通过额外的 UV_CACHE_DIR 设置掩盖它。
7. livestream 暂无终端字符不等于 agent 没有启动。必须结合 run.log、
   agent 进程和 agent-trace/events.jsonl 判断。
8. 8 个任务配置为串行执行且单任务最长 90 分钟，完整 run 可能持续很久；
   必须使用新的唯一 run-id，不能重复启动同一 run。

## 已确认的修复证据

- 38 个 VROOM private cases 已通过 checksum 校验。
- 三个配置基础 image 均已验证存在 /usr/bin/tmux 和
  /usr/bin/asciinema。
- 8 个 GitHub source 已成功 materialize。
- agy-gemini-3.8-flash-high-2026091-retry8 已正式启动第一个 agent
  task；记录时完整 8-task 评估尚未结束。

## 后续 doctor 前置检查

pitbench doctor 和 pitbench evaluate 的前置检查应覆盖：

- 禁止 formal evaluation 配置本地 repository_source；
- 检查 GitHub source 的可访问性以及 Git proxy/HTTP transport；
- 检查 agent image 中的 tmux 和 asciinema；
- 检查 Configuration Robustness 已启用且实验配置文件存在；
- 保留 private assets、image reference、image presence、credentials、
  runner 和 Docker 检查。

## 相关实现

- GitHub snapshot fetch：adapters/pitbench/git_snapshot.py
- doctor：pitbench/cli/doctor.py
- 配置：config/evaluate.example.yaml

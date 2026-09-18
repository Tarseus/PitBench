"""Shared descriptive reporting for retained nuisance experiment formats."""

from __future__ import annotations

import csv
import html
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from pitbench.schema.observation import RunObservation

REPRESENTATION_PROTOCOL_VERSION = "0.0.1"
REPRESENTATION_STATISTICS_NAME = "type_7_iqr_equal_instance_mean"
REPRESENTATION_SOLVER_SEED = 0
REPRESENTATION_RELABELINGS_PER_INSTANCE = 30


class RepresentationInstanceResult(BaseModel):
    instance_id: str
    expected_relabelings: int = Field(gt=0)
    observed_relabelings: int = Field(ge=0)
    valid_relabelings: int = Field(ge=0)
    iqr: float | None = None


class RepresentationCodeStateResult(BaseModel):
    instance_count: int = Field(ge=0)
    complete_instance_count: int = Field(ge=0)
    mean_iqr: float | None = None
    instances: list[RepresentationInstanceResult]


class RepresentationBudgetResult(BaseModel):
    budget_sec: float = Field(gt=0)
    base: RepresentationCodeStateResult
    agent: RepresentationCodeStateResult
    change: float | None = None


class RepresentationRobustnessReport(BaseModel):
    task_id: str
    metric: Literal["representation_robustness"] = "representation_robustness"
    protocol_version: str = REPRESENTATION_PROTOCOL_VERSION
    statistics: str = REPRESENTATION_STATISTICS_NAME
    solver_seed: int = REPRESENTATION_SOLVER_SEED
    relabelings_per_instance: int = Field(
        default=REPRESENTATION_RELABELINGS_PER_INSTANCE, gt=0
    )
    instance_ids: list[str]
    budgets_sec: list[float]
    by_budget: dict[str, RepresentationBudgetResult]


def _type_7_quantile(values: Sequence[float], probability: float) -> float:
    ordered_values = sorted(values)
    if not ordered_values:
        raise ValueError("cannot compute a quantile without values")
    position = probability * (len(ordered_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered_values[lower_index]
    upper_weight = position - lower_index
    return (
        ordered_values[lower_index] * (1 - upper_weight)
        + ordered_values[upper_index] * upper_weight
    )


def representation_iqr(values: Sequence[float]) -> float:
    """Compute the approved Type 7 IQR for equivalent representations."""
    return _type_7_quantile(values, 0.75) - _type_7_quantile(values, 0.25)


def _finite_gap(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _representation_instance_result(
    records: Sequence[dict],
    *,
    instance_id: str,
    relabelings_per_instance: int,
) -> RepresentationInstanceResult:
    valid_gaps = [
        record["normalized_gap"]
        for record in records
        if record.get("verified_feasible") is True
        and _finite_gap(record.get("normalized_gap"))
    ]
    complete = len(records) == relabelings_per_instance and len(valid_gaps) == (
        relabelings_per_instance
    )
    return RepresentationInstanceResult(
        instance_id=instance_id,
        expected_relabelings=relabelings_per_instance,
        observed_relabelings=len(records),
        valid_relabelings=len(valid_gaps),
        iqr=representation_iqr(valid_gaps) if complete else None,
    )


def compute_representation_robustness_report(
    records: Sequence[dict],
    *,
    task_id: str,
    instance_ids: Sequence[str],
    budgets_sec: Sequence[float],
    solver_seed: int = REPRESENTATION_SOLVER_SEED,
    relabelings_per_instance: int = REPRESENTATION_RELABELINGS_PER_INSTANCE,
) -> RepresentationRobustnessReport:
    """Compute the formal customer-relabeling Representation Robustness report."""
    declared_instances = list(instance_ids)
    declared_budgets = list(budgets_sec)
    if len(declared_instances) != 10 or len(set(declared_instances)) != 10:
        raise ValueError("representation protocol requires ten distinct instances")
    if not declared_budgets or any(
        not _finite_gap(budget) or budget <= 0 for budget in declared_budgets
    ):
        raise ValueError("budgets_sec must contain positive finite budgets")
    if len(set(declared_budgets)) != len(declared_budgets):
        raise ValueError("budgets_sec must not contain duplicates")
    if solver_seed != REPRESENTATION_SOLVER_SEED:
        raise ValueError("representation protocol requires solver seed 0")
    if relabelings_per_instance != REPRESENTATION_RELABELINGS_PER_INSTANCE:
        raise ValueError("representation protocol requires 30 relabelings per instance")

    expected_instances = set(declared_instances)
    expected_budgets = set(declared_budgets)
    indexed: dict[tuple[str, float, str, int, int], dict] = {}
    for record in records:
        if record.get("task_id") not in (None, task_id):
            raise ValueError("representation record belongs to another task")
        if record.get("axis") != "representation":
            raise ValueError("representation report requires representation records")
        instance_id = record.get("instance")
        budget_sec = record.get("budget_sec")
        code_state = record.get("code_state")
        replicate = record.get("replicate")
        observed_solver_seed = record.get("solver_seed")
        if instance_id not in expected_instances:
            raise ValueError("representation record uses an undeclared instance")
        if budget_sec not in expected_budgets:
            raise ValueError("representation record uses an undeclared budget")
        if code_state not in {"base", "agent"}:
            raise ValueError("representation record uses an undeclared code state")
        if observed_solver_seed != solver_seed:
            raise ValueError("representation record uses an undeclared solver seed")
        if not isinstance(replicate, int) or isinstance(replicate, bool):
            raise ValueError("representation replicate must be an integer")
        if not 0 <= replicate < relabelings_per_instance:
            raise ValueError("representation replicate is outside the declared grid")
        key = (instance_id, budget_sec, code_state, replicate, observed_solver_seed)
        if key in indexed:
            raise ValueError("duplicate representation observation")
        indexed[key] = record

    by_budget: dict[str, RepresentationBudgetResult] = {}
    for budget_sec in declared_budgets:
        state_results: dict[str, RepresentationCodeStateResult] = {}
        for code_state in ("base", "agent"):
            instances = [
                _representation_instance_result(
                    [
                        indexed[key]
                        for key in indexed
                        if key[:3] == (instance_id, budget_sec, code_state)
                    ],
                    instance_id=instance_id,
                    relabelings_per_instance=relabelings_per_instance,
                )
                for instance_id in declared_instances
            ]
            complete_iqrs = [
                result.iqr for result in instances if result.iqr is not None
            ]
            state_results[code_state] = RepresentationCodeStateResult(
                instance_count=len(instances),
                complete_instance_count=len(complete_iqrs),
                mean_iqr=(
                    statistics.fmean(complete_iqrs)
                    if len(complete_iqrs) == len(instances)
                    else None
                ),
                instances=instances,
            )
        base_mean = state_results["base"].mean_iqr
        agent_mean = state_results["agent"].mean_iqr
        by_budget[f"{budget_sec:g}"] = RepresentationBudgetResult(
            budget_sec=budget_sec,
            base=state_results["base"],
            agent=state_results["agent"],
            change=(
                agent_mean - base_mean
                if base_mean is not None and agent_mean is not None
                else None
            ),
        )

    return RepresentationRobustnessReport(
        task_id=task_id,
        solver_seed=solver_seed,
        relabelings_per_instance=relabelings_per_instance,
        instance_ids=declared_instances,
        budgets_sec=declared_budgets,
        by_budget=by_budget,
    )


def _relative(source: Path, output: Path, value: str | None) -> str | None:
    return os.path.relpath(source / value, output) if value else None


class DirectoryRunResults:
    """Read an explicit job manifest and one result file per run."""

    @staticmethod
    def load(source: Path, output: Path, manifest: dict) -> tuple[dict, list[dict]]:
        state = manifest.get("code_state", "base")
        jobs = [{"code_state": state, **job} for job in manifest["jobs"]]
        records = []
        for job in jobs:
            path = source / "runs" / job["run_id"] / "result.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            verification = record.get("verification") or {}
            normalized_gap = record.get("normalized_gap")
            if normalized_gap is None:
                normalized_gap = (record.get("observation") or {}).get(
                    "normalized_gap"
                )
            records.append(
                {
                    **job,
                    **record,
                    "normalized_gap": normalized_gap,
                    "verified_objective": verification.get("objective")
                    if verification.get("feasible")
                    else None,
                    "result_path": os.path.relpath(path, output),
                    "trajectory_path": _relative(
                        source, output, record.get("trajectory_path")
                    )
                    if record.get("trajectory_path")
                    else (
                        os.path.relpath(path.with_name("trajectory.jsonl"), output)
                        if path.with_name("trajectory.jsonl").exists()
                        else None
                    ),
                }
            )
        return {**manifest, "jobs": jobs}, records


class JudgeRunResults:
    """Read shared judge observations with independently mapped solution checks."""

    @staticmethod
    def load(source: Path, output: Path, manifest: dict) -> tuple[dict, list[dict]]:
        transformations = json.loads((source / "transformations.json").read_text())
        config = manifest.get("configuration", manifest)
        states = manifest["code_states"]
        seed = config["solver_seed"]
        counts = Counter()
        by_transform = {}
        jobs = []
        for identity, transformation in transformations.items():
            original = transformation["original_instance_id"]
            by_transform[identity] = {
                "instance": original,
                "axis": "representation",
                "replicate": counts[original],
                "solver_seed": seed,
                "permutation": transformation["transform_id"],
            }
            counts[original] += 1
            for budget in manifest["budgets_sec"]:
                for state in states:
                    jobs.append(
                        {
                            **by_transform[identity],
                            "budget_sec": budget,
                            "code_state": state,
                        }
                    )
        records = []
        checkpoint = source / "results.jsonl"
        if checkpoint.exists():
            for line in checkpoint.read_text().splitlines():
                if not line.strip():
                    continue
                raw = json.loads(line)
                observation = RunObservation.model_validate(raw["observation"])
                if observation.task_id != manifest["task_id"]:
                    raise ValueError("nuisance observation belongs to another task")
                verification = raw.get("verification") or {}
                mapped = verification.get("mapped_original") or {}
                artifacts = raw.get("artifacts") or {}
                feasible = (
                    observation.valid
                    and mapped.get("feasible") is True
                    and verification.get("objective_preserved") is True
                )
                records.append(
                    {
                        "task_id": observation.task_id,
                        **by_transform[observation.instance_id],
                        "budget_sec": observation.budget_sec,
                        "code_state": observation.code_state.value,
                        "solver_seed": observation.solver_seed,
                        "execution_status": observation.status.value,
                        "model_status": observation.solver_status,
                        "verified_feasible": feasible,
                        "normalized_gap": (
                            observation.normalized_gap if feasible else None
                        ),
                        "verified_objective": mapped.get("objective")
                        if feasible
                        else None,
                        "solver_objective": observation.objective,
                        "dual_bound": observation.dual_bound,
                        "solver_runtime_sec": observation.wall_time_sec,
                        "nodes": observation.nodes,
                        "solution_value_valid": bool(artifacts.get("solution")),
                        "result_path": _relative(
                            source, output, artifacts.get("solver_result")
                        ),
                        "trajectory_path": _relative(
                            source, output, artifacts.get("trajectory")
                        ),
                        "observation": raw["observation"],
                        "verification": verification,
                    }
                )
        return {
            **manifest,
            "jobs": jobs,
            "instances": [{"name": name} for name in counts],
        }, records


def _identity(record: dict) -> tuple:
    return tuple(
        record[key]
        for key in (
            "instance",
            "budget_sec",
            "axis",
            "code_state",
            "replicate",
            "solver_seed",
        )
    )


def _group(record: dict) -> tuple:
    return tuple(
        record[key] for key in ("instance", "budget_sec", "axis", "code_state")
    )


def _representation_configuration(manifest: dict) -> dict:
    configuration = manifest.get("config") or manifest.get("configuration") or {}
    if "representation" in configuration:
        return configuration["representation"]
    if {
        "relabeling_generation_seed",
        "relabelings_per_instance",
    } <= manifest.keys():
        return {
            "kind": "customer_relabeling",
            "solver_seed": manifest.get("solver_seed", REPRESENTATION_SOLVER_SEED),
            "relabelings_per_instance": manifest["relabelings_per_instance"],
        }
    return configuration


def _representation_instance_ids(manifest: dict) -> list[str]:
    instance_ids = []
    for instance in manifest.get("instances", []):
        instance_id = instance.get("name", instance.get("id"))
        if instance_id is None:
            raise ValueError("representation manifest instance has no identifier")
        instance_ids.append(instance_id)
    return instance_ids


def report_nuisance_results(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    manifest_path = source / "experiment.json"
    if not manifest_path.exists():
        manifest_path = source / "details.json"
    manifest = json.loads(manifest_path.read_text())
    reader = DirectoryRunResults if "jobs" in manifest else JudgeRunResults
    manifest, records = reader.load(source, output, manifest)
    expected = {_identity(job) for job in manifest["jobs"]}
    observed = {_identity(record) for record in records}
    if len(expected) != len(manifest["jobs"]):
        raise ValueError("duplicate planned run in nuisance manifest")
    if len(observed) != len(records):
        raise ValueError("duplicate nuisance observation")
    if observed - expected:
        raise ValueError("nuisance observation is outside the declared run grid")
    counts = Counter(_group(job) for job in manifest["jobs"])
    grouped = defaultdict(list)
    for record in records:
        grouped[_group(record)].append(record)
    summary = {
        "expected_runs": len(expected),
        "completed_runs": len(records),
        "missing_runs": len(expected - observed),
        "complete": expected == observed,
        "execution_statuses": dict(Counter(row["execution_status"] for row in records)),
        "model_statuses": dict(
            Counter(row.get("model_status") or "unavailable" for row in records)
        ),
        "statistics": "deferred",
        "groups": [],
    }
    for key, expected_count in counts.items():
        group = grouped[key]
        item = dict(
            zip(("instance", "budget_sec", "axis", "code_state"), key, strict=True)
        )
        item.update(
            {
                "completed": len(group),
                "expected": expected_count,
                "missing": expected_count - len(group),
                "verified_feasible": sum(
                    bool(row.get("verified_feasible")) for row in group
                ),
                "optimal_with_configured_tolerances": sum(
                    bool(row.get("optimal_with_configured_tolerances")) for row in group
                ),
                "no_solution_vector": sum(
                    row.get("solution_value_valid") is False for row in group
                ),
                "rejected_solution_vectors": sum(
                    row.get("solution_value_valid") is True
                    and not row.get("verified_feasible")
                    for row in group
                ),
            }
        )
        for measure in (
            "verified_objective",
            "dual_bound",
            "solver_gap",
            "solver_runtime_sec",
            "time_to_solver_optimal_sec",
            "nodes",
        ):
            values = [
                row[measure]
                for row in group
                if row.get(measure) is not None
                and (
                    measure not in ("verified_objective", "solver_gap")
                    or row.get("verified_feasible")
                )
            ]
            item[measure] = (
                {"count": len(values), "min": min(values), "max": max(values)}
                if values
                else None
            )
        summary["groups"].append(item)
    formal_representation_report = None
    planned_axes = {job.get("axis") for job in manifest["jobs"]}
    representation_configuration = _representation_configuration(manifest)
    if (
        planned_axes == {"representation"}
        and representation_configuration.get("kind") == "customer_relabeling"
    ):
        formal_representation_report = compute_representation_robustness_report(
            records,
            task_id=manifest["task_id"],
            instance_ids=_representation_instance_ids(manifest),
            budgets_sec=manifest["budgets_sec"],
            solver_seed=representation_configuration.get(
                "solver_seed", REPRESENTATION_SOLVER_SEED
            ),
            relabelings_per_instance=representation_configuration.get(
                "relabelings_per_instance",
                representation_configuration.get(
                    "count", REPRESENTATION_RELABELINGS_PER_INSTANCE
                ),
            ),
        )
        summary["statistics"] = REPRESENTATION_STATISTICS_NAME
        summary["representation_robustness"] = (
            formal_representation_report.model_dump(mode="json")
        )
    manifest["code_states"] = list(
        dict.fromkeys(job["code_state"] for job in manifest["jobs"])
    )
    manifest["report_metadata"] = " · ".join(
        f"{key}: {json.dumps(manifest[key], ensure_ascii=False)}"
        for key in (
            "solver",
            "solver_options",
            "verification",
            "timing",
            "equivalence_checks",
        )
        if key in manifest
    )
    title = manifest.get("task_id") or manifest.get("experiment") or "无关扰动实验"
    embedded = {"manifest": manifest, "records": records, "summary": summary}
    data = json.dumps(embedded, ensure_ascii=False, allow_nan=False).replace(
        "<", "\\u003c"
    )
    page = (
        PAGE.replace("__TITLE__", html.escape(title))
        .replace(
            "__MANIFEST__",
            html.escape(os.path.relpath(manifest_path, output), quote=True),
        )
        .replace("__DATA__", data)
    )
    output.mkdir(parents=True, exist_ok=True)
    if formal_representation_report is not None:
        (output / "representation_robustness.json").write_text(
            formal_representation_report.model_dump_json(indent=2) + "\n"
        )
    fields = [
        "instance",
        "axis",
        "code_state",
        "replicate",
        "solver_seed",
        "permutation",
        "budget_sec",
        "execution_status",
        "model_status",
        "verified_feasible",
        "verified_objective",
        "solver_objective",
        "dual_bound",
        "solver_gap",
        "solver_runtime_sec",
        "time_to_solver_optimal_sec",
        "optimal_with_configured_tolerances",
        "nodes",
        "simplex_iterations",
        "solve_cpu_sec",
        "solve_wall_sec",
        "objective_mapping_difference",
        "objective_reporting_difference",
        "cpu",
        "result_path",
    ]
    with (output / "observations.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "report.html").write_text(page)
    return summary


PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ · 无关扰动探索实验</title><style>
body{margin:0;background:#fafaf8;color:#242a30;font:15px/1.65 system-ui,sans-serif}main{max-width:1240px;margin:40px auto;padding:0 24px}h1{font-size:25px;margin:10px 0}h2{font-size:20px;margin-top:32px}a{color:#24639a}p{max-width:1050px}.muted,small{color:#626970}.facts{font-size:18px;border-block:1px solid #d7dade;padding:16px 0;margin:20px 0}.filters{display:flex;gap:20px;flex-wrap:wrap;align-items:end;margin:24px 0 12px}label{display:flex;flex-direction:column;font-size:13px}select{padding:8px;border:1px solid #aeb5bb;background:white;font:inherit}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #ddd;white-space:nowrap}th{background:#eef0f1;position:sticky;top:0}.scroll{overflow:auto;max-height:600px}.note{border-left:3px solid #586773;padding-left:15px}svg{width:100%;height:auto;display:block}#plot{background:white;border:1px solid #d7dade}code{font-size:12px}nav{display:flex;gap:20px;flex-wrap:wrap}.legend{display:flex;gap:24px;margin:10px 0}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}#plotNote{font-size:13px}.subtitle{letter-spacing:1px;font-size:12px;color:#626970}
</style></head><body><main>
<div class="subtitle">PITBENCH · __TITLE__</div>
<h1>固定实例下，seed 与等价表示会造成多大变化？</h1>
<p id="protocol"></p>
<div class="facts" id="facts"></div>
<nav><a href="observations.csv" download>全部运行 CSV</a><a href="summary.json">逐实例结果摘要</a><a href="__MANIFEST__">实验清单与原始资产位置</a></nav>
<p class="note">保留原始分布；正式统计量及其版本见 summary.json。可行性、最优性界和终止状态按原始记录展示；求解器报告最优不等同于独立核验证明。未提供的观察量显示为空。</p>
<h2>逐实例观察</h2>
<div class="filters"><label>实例<select id="instance"></select></label><label>预算<select id="budget"></select></label><label>代码状态<select id="state"></select></label><label>观察量<select id="measure">
<option value="verified_objective">验证后的可行解目标值</option><option value="solver_gap">求解器 primal–dual gap（%）</option><option value="dual_bound">求解器最优性界</option><option value="solver_runtime_sec">记录的运行耗时（秒，含时限退出）</option><option value="time_to_solver_optimal_sec">达到最优容差的耗时（秒，仅已达标运行）</option><option value="nodes">搜索节点数</option></select></label></div>
<div class="legend" id="legend"></div>
<div id="plot"></div><p id="plotNote" class="muted"></p>
<div class="scroll"><table><thead><tr><th>组别</th><th>已运行</th><th>验证可行</th><th>最优容差达标</th><th>未返回解</th><th>解验证未通过</th><th>目标值范围</th><th>求解器 gap 范围</th><th>求解耗时范围</th></tr></thead><tbody id="groupRows"></tbody></table></div>
<h2>每次运行</h2><p class="muted">随上方实例和预算筛选。点击记录查看完整状态、独立验证结果和实际 seed；轨迹链接指向采集时保留的原始文件。</p>
<div class="scroll"><table><thead><tr><th>组别 / 编号</th><th>状态</th><th>验证可行</th><th>目标值</th><th>最优性界</th><th>求解器 gap</th><th>秒</th><th>节点</th><th>原始证据</th></tr></thead><tbody id="runRows"></tbody></table></div>
<h2>实例及解释边界</h2><div class="scroll"><table><thead><tr><th>实例与来源</th><th>变量</th><th>整数变量</th><th>约束</th><th>非零系数</th></tr></thead><tbody id="instances"></tbody></table></div>
<p id="metadata"></p>
<p>这些实例用于探索，结果只描述清单内的运行。未出解、未证明最优和运行异常均保留；缺失值不补成零。</p>
</main><script>
const DATA=__DATA__;
const names={seed:'Seed',representation:'表示',control:'相同条件复跑'};
const colors={seed:'#276a9d',representation:'#b45c27',control:'#747b80'};
const axes=[...new Set(DATA.manifest.jobs.map(x=>x.axis))];
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=v=>v==null?'—':(v!==0&&Math.abs(v)<1e-4?Number(v).toExponential(5):Number(v).toLocaleString('en-US',{maximumSignificantDigits:9}));
const range=(cell,scale=1)=>cell?`${num(cell.min*scale)} ～ ${num(cell.max*scale)}`:'—';
const validValue=(row,key)=>((key==='solver_gap'||key==='verified_objective')&&!row.verified_feasible)?null:row[key];
document.getElementById('facts').textContent=`${DATA.summary.completed_runs} / ${DATA.summary.expected_runs} 次运行 · ${DATA.summary.complete?'采集完成':'采集中，本页为当前快照'} · 统计量待定`;
document.getElementById('budget').innerHTML=DATA.manifest.budgets_sec.map(x=>`<option value="${x}">${x} 秒</option>`).join('');
document.getElementById('state').innerHTML=DATA.manifest.code_states.map(x=>`<option>${esc(x)}</option>`).join('');
document.getElementById('legend').innerHTML=axes.map(x=>`<span><i class="dot" style="background:${colors[x]}"></i>${esc(names[x]||x)}</span>`).join('');
document.getElementById('protocol').textContent=`${DATA.manifest.instances.length} 个实例；预算 ${DATA.manifest.budgets_sec.join('、')} 秒。组别、seed、重复次数和代码状态以实验清单为准，各组计划次数见下表。`;
document.getElementById('metadata').textContent=DATA.manifest.report_metadata;
document.getElementById('instance').innerHTML=DATA.manifest.instances.map(x=>`<option>${esc(x.name)}</option>`).join('');
document.getElementById('instances').innerHTML=DATA.manifest.instances.map(x=>`<tr><td>${x.details_url?`<a href="${esc(x.details_url)}">${esc(x.name)}</a>`:esc(x.name)}</td><td>${num(x.columns)}</td><td>${num(x.integer_columns)}</td><td>${num(x.rows)}</td><td>${num(x.nonzeros)}</td></tr>`).join('');
function draw(){
 const instance=document.getElementById('instance').value,budget=Number(document.getElementById('budget').value),key=document.getElementById('measure').value,state=document.getElementById('state').value;
 const label=document.getElementById('measure').selectedOptions[0].textContent;
 const rows=DATA.records.filter(x=>x.instance===instance&&x.budget_sec===budget&&x.code_state===state);
 const groups=DATA.summary.groups.filter(x=>x.instance===instance&&x.budget_sec===budget&&x.code_state===state);
 document.getElementById('groupRows').innerHTML=groups.map(x=>`<tr><td>${names[x.axis]}</td><td>${x.completed}/${x.expected}</td><td>${x.verified_feasible}</td><td>${x.optimal_with_configured_tolerances}</td><td>${x.no_solution_vector}</td><td>${x.rejected_solution_vectors}</td><td>${range(x.verified_objective)}</td><td>${range(x.solver_gap,100)}%</td><td>${range(x.solver_runtime_sec)}</td></tr>`).join('');
 document.getElementById('runRows').innerHTML=rows.slice().sort((a,b)=>axes.indexOf(a.axis)-axes.indexOf(b.axis)||a.replicate-b.replicate).map(x=>`<tr><td>${names[x.axis]} / ${x.replicate+1}</td><td>${esc(x.model_status||x.execution_status)}</td><td>${x.verified_feasible?'是':'否'}</td><td>${num(x.verified_objective)}</td><td>${num(x.dual_bound)}</td><td>${num(x.verified_feasible&&x.solver_gap!=null?x.solver_gap*100:null)}%</td><td>${num(x.solver_runtime_sec)}</td><td>${num(x.nodes)}</td><td>${x.result_path?`<a href="${esc(x.result_path)}">记录</a>`:"—"} · ${x.trajectory_path?`<a href="${esc(x.trajectory_path)}">轨迹</a>`:"—"}</td></tr>`).join('');
 const scale=key==='solver_gap'?100:1;
 const samples=axes.map(axis=>({axis,all:rows.filter(x=>x.axis===axis),values:rows.filter(x=>x.axis===axis&&validValue(x,key)!=null).sort((a,b)=>validValue(a,key)-validValue(b,key))}));
 const values=samples.flatMap(s=>s.values.map(x=>validValue(x,key)*scale));
 const plot=document.getElementById('plot');
 if(!values.length){plot.innerHTML='';plot.style.display='none';document.getElementById('plotNote').textContent='该观察量尚无可绘制的有效值；运行状态见下表。';return;}
 plot.style.display='block';
 let low=Math.min(...values),high=Math.max(...values);const padding=(high-low||Math.abs(high)*.02||1)*.08;low-=padding;high+=padding;
 if(['solver_gap','solver_runtime_sec','time_to_solver_optimal_sec','nodes'].includes(key))low=Math.max(0,low);
 const left=135,right=1060,top=65,bottom=355;
 const sampleCount=Math.max(1,...samples.map(s=>s.values.length));
 const x=rank=>left+(rank-1)/Math.max(1,sampleCount-1)*(right-left),y=value=>bottom-(value-low)/(high-low)*(bottom-top);
 let svg=`<svg viewBox="0 0 1110 420" role="img" aria-label="${esc(instance+' '+budget+'秒 '+label)}"><text x="20" y="28" font-size="17">${esc(instance)} · ${budget} 秒 · ${esc(label)}</text>`;
 for(let i=0;i<=4;i++){const value=low+(high-low)*i/4,py=y(value);svg+=`<line x1="${left}" y1="${py}" x2="${right}" y2="${py}" stroke="#e5e7e9"/><text x="${left-12}" y="${py+4}" text-anchor="end" font-size="12" fill="#626970">${esc(num(Number(value.toPrecision(4))))}</text>`;}
 for(const rank of [...new Set(Array.from({length:6},(_,i)=>1+Math.round((sampleCount-1)*i/5)))])svg+=`<text x="${x(rank)}" y="380" text-anchor="middle" font-size="12">${rank}</text>`;
 svg+=`<text x="600" y="407" text-anchor="middle" font-size="13">组内按观察值从小到大排列的样本序号（不是运行时间顺序）</text>`;
 for(const sample of samples){sample.values.forEach((row,i)=>{const value=validValue(row,key)*scale;svg+=`<circle cx="${x(i+1)}" cy="${y(value)}" r="4.2" fill="${colors[sample.axis]}" opacity="0.78"><title>${esc(names[sample.axis]+' / '+(row.replicate+1)+'，值='+num(value)+'，状态='+row.model_status+'，seed='+row.solver_seed)}</title></circle>`;});}
 plot.innerHTML=svg+'</svg>';
 document.getElementById('plotNote').textContent=`来源：本次固定面板的原始运行记录。当前观察值跨度为 ${num(Math.max(...values)-Math.min(...values))}（单位同纵轴）。${samples.map(s=>names[s.axis]+'：绘制 '+s.values.length+'/'+s.all.length+' 个已采集值').join('；')}。缺失值未补齐；“仅已达标”耗时不包含尚未达到最优容差的运行。`;
}
for(const id of ['instance','budget','measure','state'])document.getElementById(id).addEventListener('change',draw);draw();
</script></body></html>"""

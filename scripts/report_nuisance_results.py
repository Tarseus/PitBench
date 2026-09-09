"""Render retained HiGHS observations without selecting a robustness statistic."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from collections import Counter
from pathlib import Path


class HighsNuisanceReport:
    """Render the retained HiGHS observation schema without changing its statistics."""

    @staticmethod
    def report(source: Path, output: Path) -> dict:
        output.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((source / "experiment.json").read_text())
        records = []
        for job in manifest["jobs"]:
            path = source / "runs" / job["run_id"] / "result.json"
            if path.exists():
                record = json.loads(path.read_text())
                verification = record.get("verification") or {}
                records.append(
                    {
                        **record,
                        "verified_objective": verification.get("objective")
                        if verification.get("feasible")
                        else None,
                        "result_path": os.path.relpath(path, output),
                        "trajectory_path": os.path.relpath(
                            path.with_name("trajectory.jsonl"), output
                        ),
                    }
                )
        fields = [
            "instance",
            "axis",
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
        summary = {
            "expected_runs": len(manifest["jobs"]),
            "completed_runs": len(records),
            "complete": len(records) == len(manifest["jobs"]),
            "execution_statuses": dict(
                Counter(row["execution_status"] for row in records)
            ),
            "model_statuses": dict(
                Counter(row.get("model_status", "unavailable") for row in records)
            ),
            "statistics": "deferred",
            "groups": [],
        }
        for instance in manifest["instances"]:
            for budget in manifest["budgets_sec"]:
                for axis in ("seed", "representation", "control"):
                    group = [
                        row
                        for row in records
                        if row["instance"] == instance["name"]
                        and row["budget_sec"] == budget
                        and row["axis"] == axis
                    ]
                    if not group:
                        continue
                    item = {
                        "instance": instance["name"],
                        "budget_sec": budget,
                        "axis": axis,
                        "completed": len(group),
                        "expected": 3 if axis == "control" else 30,
                        "verified_feasible": sum(
                            bool(row.get("verified_feasible")) for row in group
                        ),
                        "optimal_with_configured_tolerances": sum(
                            bool(row.get("optimal_with_configured_tolerances"))
                            for row in group
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
                    for key in (
                        "verified_objective",
                        "dual_bound",
                        "solver_gap",
                        "solver_runtime_sec",
                        "time_to_solver_optimal_sec",
                        "nodes",
                    ):
                        valid = [
                            row[key]
                            for row in group
                            if row.get(key) is not None
                            and (
                                key not in ("verified_objective", "solver_gap")
                                or row.get("verified_feasible")
                            )
                        ]
                        item[key] = (
                            {"count": len(valid), "min": min(valid), "max": max(valid)}
                            if valid
                            else None
                        )
                    summary["groups"].append(item)
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        embedded = {"manifest": manifest, "records": records, "summary": summary}
        data = json.dumps(embedded, ensure_ascii=False, allow_nan=False).replace(
            "<", "\\u003c"
        )
        source_link = html.escape(
            os.path.relpath(source / "experiment.json", output), quote=True
        )
        page = PAGE.replace("__DATA__", data).replace("__MANIFEST__", source_link)
        (output / "report.html").write_text(page)
        return summary


PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HiGHS · 无关扰动探索实验</title><style>
body{margin:0;background:#fafaf8;color:#242a30;font:15px/1.65 system-ui,sans-serif}main{max-width:1240px;margin:40px auto;padding:0 24px}h1{font-size:25px;margin:10px 0}h2{font-size:20px;margin-top:32px}a{color:#24639a}p{max-width:1050px}.muted,small{color:#626970}.facts{font-size:18px;border-block:1px solid #d7dade;padding:16px 0;margin:20px 0}.filters{display:flex;gap:20px;flex-wrap:wrap;align-items:end;margin:24px 0 12px}label{display:flex;flex-direction:column;font-size:13px}select{padding:8px;border:1px solid #aeb5bb;background:white;font:inherit}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #ddd;white-space:nowrap}th{background:#eef0f1;position:sticky;top:0}.scroll{overflow:auto;max-height:600px}.note{border-left:3px solid #586773;padding-left:15px}svg{width:100%;height:auto;display:block}#plot{background:white;border:1px solid #d7dade}code{font-size:12px}nav{display:flex;gap:20px;flex-wrap:wrap}.legend{display:flex;gap:24px;margin:10px 0}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}#plotNote{font-size:13px}.subtitle{letter-spacing:1px;font-size:12px;color:#626970}
</style></head><body><main>
<div class="subtitle">PITBENCH · 原版 HiGHS 1.15.1 · 2026-09-09</div>
<h1>固定实例下，seed 与等价表示会造成多大变化？</h1>
<p>10 个固定的公开 MIPLIB collection 实例。Seed 组保持原模型顺序，改变 30 个求解 seed；表示组固定求解 seed=0，改变 30 组变量与约束排列；对照组使用原顺序、seed=0，重复 3 次。每组分别运行 10 秒和 30 秒预算。</p>
<div class="facts" id="facts"></div>
<nav><a href="observations.csv" download>全部运行 CSV</a><a href="summary.json">逐实例结果摘要</a><a href="__MANIFEST__">实验清单与原始资产位置</a></nav>
<p class="note">保留原始分布，本轮未选择 IQR、置信区间、总分或验收阈值。HiGHS 的“Optimal”表示达到固定求解容差：相对 gap 0.0001、绝对 gap 0.000001。可行解在原模型上独立检查；最优性界和终止状态由 HiGHS 报告，未独立核验证明。</p>
<h2>逐实例观察</h2>
<div class="filters"><label>实例<select id="instance"></select></label><label>预算<select id="budget"><option value="10">10 秒</option><option value="30">30 秒</option></select></label><label>观察量<select id="measure">
<option value="solver_gap">求解器 primal–dual gap（%）</option><option value="verified_objective">验证后的可行解目标值</option><option value="dual_bound">求解器最优性界</option><option value="solver_runtime_sec">求解耗时（秒，含时限退出）</option><option value="time_to_solver_optimal_sec">达到最优容差的耗时（秒，仅已达标运行）</option><option value="nodes">搜索节点数</option></select></label></div>
<div class="legend"><span><i class="dot" style="background:#276a9d"></i>Seed（30 次）</span><span><i class="dot" style="background:#b45c27"></i>表示（30 次）</span><span><i class="dot" style="background:#747b80"></i>相同条件复跑（3 次）</span></div>
<div id="plot"></div><p id="plotNote" class="muted"></p>
<div class="scroll"><table><thead><tr><th>组别</th><th>已运行</th><th>验证可行</th><th>最优容差达标</th><th>未返回解向量</th><th>解验证未通过</th><th>目标值范围</th><th>求解器 gap 范围</th><th>求解耗时范围</th></tr></thead><tbody id="groupRows"></tbody></table></div>
<h2>每次运行</h2><p class="muted">随上方实例和预算筛选。点击记录查看完整状态、独立验证结果和实际 seed；轨迹保留原生 MIP 日志事件、改进可行解事件及终止状态。</p>
<div class="scroll"><table><thead><tr><th>组别 / 编号</th><th>状态</th><th>验证可行</th><th>目标值</th><th>最优性界</th><th>求解器 gap</th><th>秒</th><th>节点</th><th>原始证据</th></tr></thead><tbody id="runRows"></tbody></table></div>
<h2>实例及解释边界</h2><div class="scroll"><table><thead><tr><th>实例与来源</th><th>变量</th><th>整数变量</th><th>约束</th><th>非零系数</th></tr></thead><tbody id="instances"></tbody></table></div>
<p>所有模型、变量类型、上下界、目标系数、目标方向和常数项均保留，300 份排列逐一通过完全还原检查。所有运行都使用同一个数值模型输入接口；记录的是求解性能，模型读取和解验证时间分开保存。固定单线程，在 6 个不同物理核心上并行运行，顺序预先打乱；对照复跑反映本次环境中的计时及限时终止波动。</p>
<p>这些实例用于探索，不代表整个 HiGHS 仓库的功能覆盖或生产分布。未出解、未证明最优和运行异常均保留。缺失值不补成零；只展示已求解样本的耗时时，不能将其解释为全组完成时间。原版单独运行，没有 Agent patch 对比。</p>
</main><script>
const DATA=__DATA__;
const names={seed:'Seed',representation:'表示',control:'相同条件复跑'};
const colors={seed:'#276a9d',representation:'#b45c27',control:'#747b80'};
const axes=['seed','representation','control'];
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=v=>v==null?'—':(v!==0&&Math.abs(v)<1e-4?Number(v).toExponential(5):Number(v).toLocaleString('en-US',{maximumSignificantDigits:9}));
const range=(cell,scale=1)=>cell?`${num(cell.min*scale)} ～ ${num(cell.max*scale)}`:'—';
const validValue=(row,key)=>((key==='solver_gap'||key==='verified_objective')&&!row.verified_feasible)?null:row[key];
document.getElementById('facts').textContent=`${DATA.summary.completed_runs} / ${DATA.summary.expected_runs} 次运行 · ${DATA.summary.complete?'采集完成':'采集中，本页为当前快照'} · 统计量待定`;
document.getElementById('instance').innerHTML=DATA.manifest.instances.map(x=>`<option>${esc(x.name)}</option>`).join('');
document.getElementById('instances').innerHTML=DATA.manifest.instances.map(x=>`<tr><td><a href="${esc(x.details_url)}">${esc(x.name)}</a></td><td>${num(x.columns)}</td><td>${num(x.integer_columns)}</td><td>${num(x.rows)}</td><td>${num(x.nonzeros)}</td></tr>`).join('');
function draw(){
 const instance=document.getElementById('instance').value,budget=Number(document.getElementById('budget').value),key=document.getElementById('measure').value;
 const label=document.getElementById('measure').selectedOptions[0].textContent;
 const rows=DATA.records.filter(x=>x.instance===instance&&x.budget_sec===budget);
 const groups=DATA.summary.groups.filter(x=>x.instance===instance&&x.budget_sec===budget);
 document.getElementById('groupRows').innerHTML=groups.map(x=>`<tr><td>${names[x.axis]}</td><td>${x.completed}/${x.expected}</td><td>${x.verified_feasible}</td><td>${x.optimal_with_configured_tolerances}</td><td>${x.no_solution_vector}</td><td>${x.rejected_solution_vectors}</td><td>${range(x.verified_objective)}</td><td>${range(x.solver_gap,100)}%</td><td>${range(x.solver_runtime_sec)}</td></tr>`).join('');
 document.getElementById('runRows').innerHTML=rows.slice().sort((a,b)=>axes.indexOf(a.axis)-axes.indexOf(b.axis)||a.replicate-b.replicate).map(x=>`<tr><td>${names[x.axis]} / ${x.replicate+1}</td><td>${esc(x.model_status||x.execution_status)}</td><td>${x.verified_feasible?'是':'否'}</td><td>${num(x.verified_objective)}</td><td>${num(x.dual_bound)}</td><td>${num(x.verified_feasible&&x.solver_gap!=null?x.solver_gap*100:null)}%</td><td>${num(x.solver_runtime_sec)}</td><td>${num(x.nodes)}</td><td><a href="${esc(x.result_path)}">记录</a> · <a href="${esc(x.trajectory_path)}">轨迹</a></td></tr>`).join('');
 const scale=key==='solver_gap'?100:1;
 const samples=axes.map(axis=>({axis,all:rows.filter(x=>x.axis===axis),values:rows.filter(x=>x.axis===axis&&validValue(x,key)!=null).sort((a,b)=>validValue(a,key)-validValue(b,key))}));
 const values=samples.flatMap(s=>s.values.map(x=>validValue(x,key)*scale));
 const plot=document.getElementById('plot');
 if(!values.length){plot.innerHTML='';plot.style.display='none';document.getElementById('plotNote').textContent='该观察量尚无可绘制的有效值；运行状态见下表。';return;}
 plot.style.display='block';
 let low=Math.min(...values),high=Math.max(...values);const padding=(high-low||Math.abs(high)*.02||1)*.08;low-=padding;high+=padding;
 if(['solver_gap','solver_runtime_sec','time_to_solver_optimal_sec','nodes'].includes(key))low=Math.max(0,low);
 const left=135,right=1060,top=65,bottom=355;
 const x=rank=>left+(rank-1)/29*(right-left),y=value=>bottom-(value-low)/(high-low)*(bottom-top);
 let svg=`<svg viewBox="0 0 1110 420" role="img" aria-label="${esc(instance+' '+budget+'秒 '+label)}"><text x="20" y="28" font-size="17">${esc(instance)} · ${budget} 秒 · ${esc(label)}</text>`;
 for(let i=0;i<=4;i++){const value=low+(high-low)*i/4,py=y(value);svg+=`<line x1="${left}" y1="${py}" x2="${right}" y2="${py}" stroke="#e5e7e9"/><text x="${left-12}" y="${py+4}" text-anchor="end" font-size="12" fill="#626970">${esc(num(Number(value.toPrecision(4))))}</text>`;}
 for(const rank of [1,5,10,15,20,25,30])svg+=`<text x="${x(rank)}" y="380" text-anchor="middle" font-size="12">${rank}</text>`;
 svg+=`<text x="600" y="407" text-anchor="middle" font-size="13">组内按观察值从小到大排列的样本序号（不是运行时间顺序）</text>`;
 for(const sample of samples){sample.values.forEach((row,i)=>{const value=validValue(row,key)*scale;svg+=`<circle cx="${x(i+1)}" cy="${y(value)}" r="4.2" fill="${colors[sample.axis]}" opacity="0.78"><title>${esc(names[sample.axis]+' / '+(row.replicate+1)+'，值='+num(value)+'，状态='+row.model_status+'，seed='+row.solver_seed)}</title></circle>`;});}
 plot.innerHTML=svg+'</svg>';
 document.getElementById('plotNote').textContent=`来源：本次固定面板的原始运行记录。当前观察值跨度为 ${num(Math.max(...values)-Math.min(...values))}（单位同纵轴）。${samples.map(s=>names[s.axis]+'：绘制 '+s.values.length+'/'+s.all.length+' 个已采集值').join('；')}。缺失值未补齐；“仅已达标”耗时不包含尚未达到最优容差的运行。`;
}
for(const id of ['instance','budget','measure'])document.getElementById(id).addEventListener('change',draw);draw();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = HighsNuisanceReport.report(args.source.resolve(), args.output.resolve())
    print(json.dumps({key: value for key, value in summary.items() if key != "groups"}))


if __name__ == "__main__":
    main()

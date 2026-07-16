#!/usr/bin/env python3
"""Create interactive quality-versus-throughput plots and dashboard."""

import html
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROFILE_ORDER = {"fast": 0, "balanced": 1, "thorough": 2}
PROFILE_MARKERS = {"fast": "triangle-up", "balanced": "circle", "thorough": "square"}


def suite_label(suite: str) -> str:
    return {
        "public-api": "Synthetic public API",
        "phototourism-val": "PhotoTourism validation",
        "homography-ransac-val": "Homography validation",
    }.get(suite, suite)


def mode_label(mode: str) -> str:
    return {
        "ransac": "RANSAC",
        "msac": "MSAC",
        "magsac": "MAGSAC",
        "magsac_pp": "MAGSAC++",
    }.get(mode, mode)


def percentage(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "-"


def update_history(output: Path, dataset_rows: list[dict]) -> dict:
    path = output / "history" / "index.json"
    try:
        history = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        history = {"schema_version": 1, "runs": []}

    if os.environ.get("BENCHMARK_SCOPE", "local") != "full":
        return history

    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    record = {
        "run_id": run_id,
        "revision": os.environ.get(
            "BENCHMARK_TARGET_REVISION", os.environ.get("GITHUB_SHA", "local")
        ),
        "repository": os.environ.get("BENCHMARK_TARGET_REPOSITORY", "soraxas/inlier-benchmarks"),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "groups": [
            {
                field: row.get(field)
                for field in (
                    "suite",
                    "estimator",
                    "scoring_mode",
                    "profile",
                    "trials",
                    "scene_count",
                    "mean_runtime_ms",
                    "runtime_se_ms",
                    "auc_pose_10",
                    "auc_pose_10_se",
                )
            }
            for row in dataset_rows
            if row["suite"] == "phototourism-val"
        ],
    }
    history["runs"] = [entry for entry in history.get("runs", []) if entry.get("run_id") != run_id]
    history["runs"].append(record)
    history["runs"] = history["runs"][-100:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2) + "\n")
    return history


def quality_metric(suite: str) -> tuple[str, str, str]:
    if suite == "homography-ransac-val":
        return "auc_homography_3", "auc_homography_3_se", "Transfer AUC @ 3 pixels"
    return "auc_pose_10", "auc_pose_10_se", "Pose AUC @ 10 degrees"


def make_plot(rows: list[dict], title: str, suite: str) -> dict:
    """Build a Plotly figure following the SuperRANSAC speed/AUC convention."""
    metric, metric_se, metric_label = quality_metric(suite)
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_mode[row["scoring_mode"]].append(row)
    traces = []
    for mode, points in sorted(by_mode.items()):
        points.sort(key=lambda row: PROFILE_ORDER.get(row["profile"], 99))
        traces.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": mode_label(mode),
                "x": [point["mean_runtime_ms"] / 1_000.0 for point in points],
                "y": [point[metric] for point in points],
                "error_x": {
                    "type": "data",
                    "array": [point["runtime_se_ms"] / 1_000.0 for point in points],
                    "visible": True,
                },
                "error_y": {
                    "type": "data",
                    "array": [point[metric_se] or 0.0 for point in points],
                    "visible": True,
                },
                "marker": {
                    "size": 10,
                    "line": {"color": "#fff", "width": 1},
                    "symbol": [PROFILE_MARKERS.get(point["profile"], "circle") for point in points],
                },
                "customdata": [
                    [point["profile"], point["trials"], point["scene_count"], point["success_rate"]]
                    for point in points
                ],
                "hovertemplate": (
                    "%{fullData.name}<br>Budget: %{customdata[0]}<br>Time: %{x:.4f} s"
                    f"<br>{metric_label}: %{{y:.4f}}<br>Trials: %{{customdata[1]}}"
                    "<br>Pairs: %{customdata[2]}<br>Success: %{customdata[3]:.1%}<extra></extra>"
                ),
            }
        )
    return {
        "traces": traces,
        "layout": {
            "title": f"{title}: speed versus accuracy",
            "paper_bgcolor": "#fff",
            "plot_bgcolor": "#fff",
            "font": {"color": "#222"},
            "colorway": ["#2563eb", "#f97316", "#16a34a", "#dc2626"],
            "hovermode": "closest",
            "xaxis": {
                "title": "Average time [s]",
                "type": "log",
                "zeroline": False,
                "showline": True,
                "linecolor": "#9ca3af",
                "gridcolor": "#e5e7eb",
            },
            "yaxis": {
                "title": metric_label,
                "range": [0, 1],
                "tickformat": ".1f",
                "zeroline": True,
                "zerolinecolor": "#9ca3af",
                "showline": True,
                "linecolor": "#9ca3af",
                "gridcolor": "#e5e7eb",
            },
            "legend": {"title": {"text": "Robust method"}, "orientation": "h", "y": -0.24},
            "margin": {"l": 85, "r": 30, "t": 65, "b": 110},
        },
    }


def main(summary_path: str, output_dir: str) -> None:
    summary = json.loads(Path(summary_path).read_text())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = summary["groups"]
    dataset_rows = summary.get("dataset_groups", [])
    history = update_history(output, dataset_rows)
    by_problem: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in dataset_rows:
        if row["suite"] != "public-api":
            aggregate_scene = f"all selected pairs ({row['scene_count']})"
            by_problem[(row["suite"], row["estimator"], aggregate_scene)].append(row)
    plot_sections = []
    plot_figures = []
    for (suite, estimator, scene), problem_rows in sorted(by_problem.items()):
        chart_id = re.sub(r"[^A-Za-z0-9_-]+", "-", f"chart-{suite}-{estimator}-{scene}")
        plot_figures.append((chart_id, make_plot(problem_rows, f"{suite_label(suite)} / {estimator} / {scene}", suite)))
        plot_sections.append(
            "<section><div class=label>"
            f"<h2>{html.escape(suite_label(suite))}</h2>"
            f"<p>{html.escape(estimator.replace('_', ' '))} / {html.escape(scene)}</p>"
            "<p>Up and left is better.</p></div>"
            f"<div id={html.escape(chart_id)} class=chart aria-label='{html.escape(estimator)} {html.escape(scene)} trade-off plot'></div></section>"
        )
    table_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(suite_label(row['suite']))}</td>"
        f"<td>{html.escape(row['estimator'])}</td>"
        f"<td>{html.escape(mode_label(row['scoring_mode']))}</td>"
        f"<td>{html.escape(row['profile'])}</td>"
        f"<td>{html.escape(row['scene'])}</td>"
        f"<td>{percentage(row[quality_metric(row['suite'])[0]])}</td>"
        f"<td>{row['success_rate']:.1%}</td>"
        f"<td>{row['median_normalized_model_error']:.3f}</td>"
        f"<td>{row['median_runtime_ms']:.3f}</td>"
        f"<td>{row['median_iterations']:.0f}</td>"
        "</tr>"
        for row in rows
        if row["suite"] != "public-api"
    )
    history_json = json.dumps(history["runs"])
    plots_json = json.dumps(dict(plot_figures))
    (output / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Inlier Benchmarks</title>"
        "<script src=https://cdn.plot.ly/plotly-2.35.2.min.js></script>"
        "<style>body{background:#222;color:#eee;font-family:system-ui;margin:0 auto;max-width:1800px;padding:2rem}"
        "header{border-bottom:1px solid #555;margin-bottom:2rem}.primer{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1.5rem;border-bottom:1px solid #555;padding:0 0 2rem}"
        ".primer h2{margin:0 0 .4rem}.primer p{color:#c9e5b2;line-height:1.45}section{display:grid;grid-template-columns:260px minmax(0,1fr);gap:2rem;border-bottom:1px solid #555;padding:2rem 0}"
        ".label{background:#385020;border-left:4px solid #9fc46c;padding:1.25rem;height:max-content}.label h2{text-transform:capitalize;margin:0}.label p{color:#c9e5b2}"
        ".chart{background:#fff;min-height:510px;min-width:0;border:1px solid #4b5563}table{border-collapse:collapse;width:100%;margin-top:2rem}"
        "td,th{border:1px solid #666;padding:.4rem;text-align:right}td:first-child,td:nth-child(2),td:nth-child(3),td:nth-child(4),td:nth-child(5){text-align:left}"
        ".tabs{display:flex;gap:.5rem;margin:1rem 0 2rem;border-bottom:1px solid #555}.tabs button{appearance:none;background:transparent;border:0;border-bottom:3px solid transparent;color:#c9e5b2;cursor:pointer;font:inherit;padding:.75rem 1rem}.tabs button[aria-selected=true]{border-color:#9fc46c;color:#fff}.panel[hidden]{display:none}.history-toolbar{display:flex;align-items:center;gap:.75rem;margin:1.25rem 0}.history-toolbar select{background:#fff;border:1px solid #888;color:#222;font:inherit;padding:.35rem}.history-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:1.5rem}.history-chart{min-height:430px;background:#fff}"
        "@media(max-width:800px){section{grid-template-columns:1fr}.history-grid{grid-template-columns:1fr}}</style>"
        "<header><h1>Inlier Benchmarks</h1><p>Robust-estimation speed versus accuracy. Up and left is better.</p></header>"
        "<nav class=tabs role=tablist aria-label='Benchmark views'><button role=tab aria-selected=true aria-controls=quality-panel data-tab=quality-panel>Current results</button><button role=tab aria-selected=false aria-controls=history-panel data-tab=history-panel>Historical regression</button></nav>"
        "<main id=quality-panel class=panel><div class=primer><div><h2>Reading the plots</h2><p>PhotoTourism follows the SuperRANSAC convention: pose AUC@10 degrees on the y-axis and average estimation time in seconds on a logarithmic x-axis. Homography validation uses transfer AUC@3 pixels against its ground-truth homography. Standard-error bars show trial variability.</p>"
        "<p>Triangle, circle, and square markers denote fast, balanced, and thorough iteration budgets. The public dashboard contains only full runs over all selected pairs; smoke runs remain CI artifacts.</p></div>"
        "<div><h2>Scoring modes</h2><p><b>RANSAC</b> ranks hypotheses by inlier count. <b>MSAC</b> uses a truncated squared-residual cost. <b>MAGSAC</b> marginalizes uncertainty in the noise scale.</p>"
        "<p><b><a href=https://openaccess.thecvf.com/content_CVPR_2020/html/Barath_MAGSAC_a_Fast_Reliable_and_Accurate_Robust_Estimator_CVPR_2020_paper.html>MAGSAC++</a></b> is the sigma-consensus++ scoring variant: it uses a robust loss marginalized over the noise scale. This implementation evaluates that loss through a precomputed integral lookup table.</p></div></div>"
        f"{''.join(plot_sections)}<h2>PhotoTourism Pair Diagnostics</h2><table><thead><tr>"
        "<th>Dataset</th><th>Estimator</th><th>Mode</th><th>Profile</th><th>Scene</th>"
        "<th>Quality AUC</th><th>Success</th><th>Median model error</th><th>Median ms</th><th>Median iterations</th>"
        f"</tr></thead><tbody>{table_rows}</tbody></table></main>"
        "<main id=history-panel class=panel hidden><div class=primer><div><h2>Comparable Full Runs</h2><p>Only scheduled and manually requested full runs are retained. Smoke runs are excluded because they use a different pair and seed budget. The x-axis uses the tested inlier commit; click a point to open that revision.</p></div><div><h2>Uncertainty</h2><p>AUC error bars are deterministic bootstrap standard errors over all pair-seed trials. Runtime error bars are standard errors over those trials.</p></div></div><div class=history-toolbar><label for=history-profile>Iteration budget</label><select id=history-profile><option value=balanced selected>Balanced</option><option value=fast>Fast</option><option value=thorough>Thorough</option></select></div><div class=history-grid><div id=history-auc class=history-chart></div><div id=history-runtime class=history-chart></div></div></main>"
        "<script>const benchmarkPlots="
        f"{plots_json};"
        "Object.entries(benchmarkPlots).forEach(([id,figure])=>Plotly.newPlot(id,figure.traces,figure.layout,{responsive:true,displaylogo:false}));"
        "const benchmarkHistory="
        f"{history_json};"
        "const tabs=document.querySelectorAll('[role=tab]');tabs.forEach(tab=>tab.addEventListener('click',()=>{tabs.forEach(other=>{const selected=other===tab;other.setAttribute('aria-selected',selected);document.getElementById(other.dataset.tab).hidden=!selected});if(tab.dataset.tab==='history-panel')renderHistory()}));"
        "let historyRendered=false;const historyProfile=document.getElementById('history-profile');historyProfile.addEventListener('change',()=>{historyRendered=false;renderHistory()});const commitUrl=(repository,revision)=>`https://github.com/${repository}/commit/${revision}`;function renderHistory(){if(historyRendered)return;historyRendered=true;const revisions=[...new Set(benchmarkHistory.map(run=>run.revision.slice(0,7)))];const traces=(metric,errorMetric)=>{const series=new Map();for(const run of benchmarkHistory){for(const point of run.groups){if(point.profile!==historyProfile.value)continue;const key=[point.estimator,point.scoring_mode].join(' / ');if(!series.has(key))series.set(key,{x:[],y:[],customdata:[],error_y:{type:'data',array:[],visible:true},mode:'lines+markers',name:key,hovertemplate:'%{fullData.name}<br>Commit: %{x}<br>%{y:.4f}<br>Run: %{customdata[1]}<br>Click to open commit<extra></extra>'});const trace=series.get(key);trace.x.push(run.revision.slice(0,7));trace.y.push(point[metric]);trace.customdata.push([run.revision,run.run_id,run.repository||'soraxas/inlier-benchmarks']);trace.error_y.array.push(point[errorMetric]||0)}}return [...series.values()]};const layout=(title,yaxis)=>({title,font:{color:'#222'},paper_bgcolor:'#fff',plot_bgcolor:'#fff',xaxis:{title:'Inlier revision',type:'category',categoryorder:'array',categoryarray:revisions},yaxis:{title:yaxis,zeroline:false,gridcolor:'#e5e7eb'},legend:{orientation:'h'},margin:{l:70,r:20,t:55,b:70}});for(const [id,metric,errorMetric,title,yaxis] of [['history-auc','auc_pose_10','auc_pose_10_se','PhotoTourism pose AUC @ 10 degrees','Pose AUC @ 10 degrees'],['history-runtime','mean_runtime_ms','runtime_se_ms','PhotoTourism average estimation time','Average time [ms]']]){const chart=document.getElementById(id);Plotly.react(chart,traces(metric,errorMetric),layout(title,yaxis),{responsive:true,displaylogo:false});chart.removeAllListeners('plotly_click');chart.on('plotly_click',event=>{const [revision,,repository]=event.points[0].customdata;if(/^[0-9a-f]{7,40}$/.test(revision))window.open(commitUrl(repository,revision),'_blank','noopener')})}}</script>"
    )
    (output / "latest.json").write_text(json.dumps(summary, indent=2) + "\n")
    markdown = ["## Benchmark Smoke Summary", "", "| Dataset / estimator | Mode | Scene | Success | Model error | Iterations |", "|---|---|---|---:|---:|---:|"]
    for row in rows:
        markdown.append(f"| {suite_label(row['suite'])} / {row['estimator']} | {mode_label(row['scoring_mode'])} | {row['scene']} | {row['success_rate']:.1%} | {row['median_normalized_model_error']:.3f} | {row['median_iterations']:.0f} |")
    (output / "summary.md").write_text("\n".join(markdown) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])

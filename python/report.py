#!/usr/bin/env python3
"""Create ANN-benchmarks-style quality-versus-throughput plots and dashboard."""

import html
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROFILE_ORDER = {"fast": 0, "balanced": 1, "thorough": 2}
PROFILE_MARKERS = {"fast": "^", "balanced": "o", "thorough": "s"}


def suite_label(suite: str) -> str:
    return {"public-api": "Synthetic public API", "phototourism-val": "PhotoTourism validation"}.get(
        suite, suite
    )


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
        "revision": os.environ.get("GITHUB_SHA", "local"),
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


def make_plot(rows: list[dict], title: str, path: Path, quality_field: str) -> None:
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_mode[row["scoring_mode"]].append(row)
    for mode, points in sorted(by_mode.items()):
        points.sort(key=lambda row: PROFILE_ORDER.get(row["profile"], 99))
        if quality_field == "auc_pose_10":
            x_values = [point["mean_runtime_ms"] / 1_000.0 for point in points]
            y_values = [point["auc_pose_10"] for point in points]
        else:
            x_values = [point["success_rate"] for point in points]
            y_values = [1_000.0 / max(point["mean_runtime_ms"], 1e-9) for point in points]
        (line,) = axis.plot(
            x_values,
            y_values,
            linewidth=2,
            label=mode_label(mode),
        )
        for point, x_value, y_value in zip(points, x_values, y_values):
            axis.scatter(
                x_value,
                y_value,
                marker=PROFILE_MARKERS.get(point["profile"], "o"),
                color=line.get_color(),
                s=55,
            )
    axis.set_title(f"{title}: speed versus accuracy")
    if quality_field == "auc_pose_10":
        axis.set_xlabel("Average time [s]")
        axis.set_ylabel("Pose AUC @ 10 degrees")
        axis.set_xscale("log")
        axis.set_ylim(-0.02, 1.02)
    else:
        axis.set_xlabel("Accuracy: success rate")
        axis.set_ylabel("Speed: average estimates per second")
        axis.set_yscale("log")
        axis.set_xlim(-0.02, 1.02)
    axis.grid(True, which="both", alpha=0.3)
    method_legend = axis.legend(title="Robust method", loc="upper right")
    axis.add_artist(method_legend)
    axis.legend(
        handles=[
            Line2D([], [], color="#ddd", marker=marker, linestyle="None", label=profile.title())
            for profile, marker in PROFILE_MARKERS.items()
        ],
        title="Iteration budget",
        loc="lower left",
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(summary_path: str, output_dir: str) -> None:
    summary = json.loads(Path(summary_path).read_text())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = summary["groups"]
    dataset_rows = summary.get("dataset_groups", [])
    history = update_history(output, dataset_rows)
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    by_problem: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in dataset_rows:
        if row["suite"] != "public-api":
            aggregate_scene = f"all selected pairs ({row['scene_count']})"
            by_problem[(row["suite"], row["estimator"], aggregate_scene)].append(row)
    plot_sections = []
    for (suite, estimator, scene), problem_rows in sorted(by_problem.items()):
        filename = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{suite}-{estimator}-{scene}") + ".png"
        quality_field = "auc_pose_10" if suite == "phototourism-val" else "success_rate"
        make_plot(
            problem_rows,
            f"{suite_label(suite)} / {estimator} / {scene}",
            plots / filename,
            quality_field,
        )
        plot_sections.append(
            "<section><div class=label>"
            f"<h2>{html.escape(suite_label(suite))}</h2>"
            f"<p>{html.escape(estimator.replace('_', ' '))} / {html.escape(scene)}</p>"
            "<p>Up and left is better.</p></div>"
            f"<img src=plots/{html.escape(filename)} alt='{html.escape(estimator)} {html.escape(scene)} trade-off plot'></section>"
        )
    table_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(suite_label(row['suite']))}</td>"
        f"<td>{html.escape(row['estimator'])}</td>"
        f"<td>{html.escape(mode_label(row['scoring_mode']))}</td>"
        f"<td>{html.escape(row['profile'])}</td>"
        f"<td>{html.escape(row['scene'])}</td>"
        f"<td>{percentage(row['auc_pose_10'])}</td>"
        f"<td>{row['success_rate']:.1%}</td>"
        f"<td>{row['median_normalized_model_error']:.3f}</td>"
        f"<td>{row['median_runtime_ms']:.3f}</td>"
        f"<td>{row['median_iterations']:.0f}</td>"
        "</tr>"
        for row in rows
        if row["suite"] != "public-api"
    )
    history_json = json.dumps(history["runs"])
    (output / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Inlier Benchmarks</title>"
        "<script src=https://cdn.plot.ly/plotly-2.35.2.min.js></script>"
        "<style>body{background:#222;color:#eee;font-family:system-ui;margin:0 auto;max-width:1600px;padding:2rem}"
        "header{border-bottom:1px solid #555;margin-bottom:2rem}.primer{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1.5rem;border-bottom:1px solid #555;padding:0 0 2rem}"
        ".primer h2{margin:0 0 .4rem}.primer p{color:#c9e5b2;line-height:1.45}section{display:grid;grid-template-columns:260px minmax(0,1fr);gap:2rem;border-bottom:1px solid #555;padding:2rem 0}"
        ".label{background:#385020;padding:1rem;height:max-content}.label h2{text-transform:capitalize;margin:0}.label p{color:#c9e5b2}"
        "img{background:#fff;max-width:100%;width:100%}table{border-collapse:collapse;width:100%;margin-top:2rem}"
        "td,th{border:1px solid #666;padding:.4rem;text-align:right}td:first-child,td:nth-child(2),td:nth-child(3),td:nth-child(4),td:nth-child(5){text-align:left}"
        ".tabs{display:flex;gap:.5rem;margin:1rem 0 2rem;border-bottom:1px solid #555}.tabs button{appearance:none;background:transparent;border:0;border-bottom:3px solid transparent;color:#c9e5b2;cursor:pointer;font:inherit;padding:.75rem 1rem}.tabs button[aria-selected=true]{border-color:#9fc46c;color:#fff}.panel[hidden]{display:none}.history-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:1.5rem}.history-chart{min-height:430px;background:#fff}"
        "@media(max-width:800px){section{grid-template-columns:1fr}.history-grid{grid-template-columns:1fr}}</style>"
        "<header><h1>Inlier Benchmarks</h1><p>Robust-estimation speed versus accuracy. Up and left is better.</p></header>"
        "<nav class=tabs role=tablist aria-label='Benchmark views'><button role=tab aria-selected=true aria-controls=quality-panel data-tab=quality-panel>Current results</button><button role=tab aria-selected=false aria-controls=history-panel data-tab=history-panel>Historical regression</button></nav>"
        "<main id=quality-panel class=panel><div class=primer><div><h2>Reading the plots</h2><p>PhotoTourism follows the SuperRANSAC convention: pose AUC@10 degrees on the y-axis and average estimation time in seconds on a logarithmic x-axis.</p>"
        "<p>Triangle, circle, and square markers denote fast, balanced, and thorough iteration budgets. Pull requests and pushes publish balanced-only smoke points; scheduled or manually requested full runs provide the curve.</p></div>"
        "<div><h2>Scoring modes</h2><p><b>RANSAC</b> ranks hypotheses by inlier count. <b>MSAC</b> uses a truncated squared-residual cost. <b>MAGSAC</b> marginalizes uncertainty in the noise scale.</p>"
        "<p><b><a href=https://openaccess.thecvf.com/content_CVPR_2020/html/Barath_MAGSAC_a_Fast_Reliable_and_Accurate_Robust_Estimator_CVPR_2020_paper.html>MAGSAC++</a></b> is the sigma-consensus++ scoring variant: it uses a robust loss marginalized over the noise scale. This implementation evaluates that loss through a precomputed integral lookup table.</p></div></div>"
        f"{''.join(plot_sections)}<h2>PhotoTourism Pair Diagnostics</h2><table><thead><tr>"
        "<th>Dataset</th><th>Estimator</th><th>Mode</th><th>Profile</th><th>Scene</th>"
        "<th>AUC@10°</th><th>Success</th><th>Median model error</th><th>Median ms</th><th>Median iterations</th>"
        f"</tr></thead><tbody>{table_rows}</tbody></table></main>"
        "<main id=history-panel class=panel hidden><div class=primer><div><h2>Comparable Full Runs</h2><p>Only scheduled and manually requested full runs are retained. Smoke runs are excluded because they use a different pair and seed budget.</p></div><div><h2>Uncertainty</h2><p>AUC error bars are deterministic bootstrap standard errors over all pair-seed trials. Runtime error bars are standard errors over those trials.</p></div></div><div class=history-grid><div id=history-auc class=history-chart></div><div id=history-runtime class=history-chart></div></div></main>"
        "<script>const benchmarkHistory="
        f"{history_json};"
        "const tabs=document.querySelectorAll('[role=tab]');tabs.forEach(tab=>tab.addEventListener('click',()=>{tabs.forEach(other=>{const selected=other===tab;other.setAttribute('aria-selected',selected);document.getElementById(other.dataset.tab).hidden=!selected});if(tab.dataset.tab==='history-panel')renderHistory()}));"
        "let historyRendered=false;function renderHistory(){if(historyRendered)return;historyRendered=true;const traces=(metric,errorMetric)=>{const series=new Map();for(const run of benchmarkHistory){for(const point of run.groups){const key=[point.estimator,point.scoring_mode,point.profile].join(' / ');if(!series.has(key))series.set(key,{x:[],y:[],error_y:{type:'data',array:[],visible:true},mode:'lines+markers',name:key,hovertemplate:'%{fullData.name}<br>%{x}<br>%{y:.4f}<extra></extra>'});const trace=series.get(key);trace.x.push(run.recorded_at);trace.y.push(point[metric]);trace.error_y.array.push(point[errorMetric]||0)}}return [...series.values()]};const layout=(title,yaxis)=>({title,font:{color:'#222'},paper_bgcolor:'#fff',plot_bgcolor:'#fff',xaxis:{title:'Full benchmark run',type:'date'},yaxis:{title:yaxis,zeroline:false},legend:{orientation:'h'},margin:{l:70,r:20,t:55,b:70}});Plotly.newPlot('history-auc',traces('auc_pose_10','auc_pose_10_se'),layout('PhotoTourism pose AUC @ 10 degrees','Pose AUC @ 10 degrees'),{responsive:true});Plotly.newPlot('history-runtime',traces('mean_runtime_ms','runtime_se_ms'),layout('PhotoTourism average estimation time','Average time [ms]'),{responsive:true})}</script>"
    )
    (output / "latest.json").write_text(json.dumps(summary, indent=2) + "\n")
    markdown = ["## Benchmark Smoke Summary", "", "| Dataset / estimator | Mode | Scene | Success | Model error | Iterations |", "|---|---|---|---:|---:|---:|"]
    for row in rows:
        markdown.append(f"| {suite_label(row['suite'])} / {row['estimator']} | {mode_label(row['scoring_mode'])} | {row['scene']} | {row['success_rate']:.1%} | {row['median_normalized_model_error']:.3f} | {row['median_iterations']:.0f} |")
    (output / "summary.md").write_text("\n".join(markdown) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])

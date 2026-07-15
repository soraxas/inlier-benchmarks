#!/usr/bin/env python3
"""Create ANN-benchmarks-style quality-versus-throughput plots and dashboard."""

import html
import json
import re
import sys
from collections import defaultdict
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


def make_plot(rows: list[dict], title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_mode[row["scoring_mode"]].append(row)
    for mode, points in sorted(by_mode.items()):
        points.sort(key=lambda row: PROFILE_ORDER.get(row["profile"], 99))
        x_values = [point["success_rate"] for point in points]
        y_values = [1_000.0 / max(point["median_runtime_ms"], 1e-9) for point in points]
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
    axis.set_xlabel("Accuracy: success rate")
    axis.set_ylabel("Speed: estimates per second")
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
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    by_problem: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["suite"] == "public-api":
            by_problem[(row["suite"], row["estimator"], row["scene"])].append(row)
    for row in dataset_rows:
        if row["suite"] != "public-api":
            aggregate_scene = f"all selected pairs ({row['scene_count']})"
            by_problem[(row["suite"], row["estimator"], aggregate_scene)].append(row)
    plot_sections = []
    for (suite, estimator, scene), problem_rows in sorted(by_problem.items()):
        filename = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{suite}-{estimator}-{scene}") + ".png"
        make_plot(problem_rows, f"{suite_label(suite)} / {estimator} / {scene}", plots / filename)
        plot_sections.append(
            "<section><div class=label>"
            f"<h2>{html.escape(suite_label(suite))}</h2>"
            f"<p>{html.escape(estimator.replace('_', ' '))} / {html.escape(scene)}</p>"
            "<p>Right and up is better.</p></div>"
            f"<img src=plots/{html.escape(filename)} alt='{html.escape(estimator)} {html.escape(scene)} trade-off plot'></section>"
        )
    table_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(suite_label(row['suite']))}</td>"
        f"<td>{html.escape(row['estimator'])}</td>"
        f"<td>{html.escape(mode_label(row['scoring_mode']))}</td>"
        f"<td>{html.escape(row['profile'])}</td>"
        f"<td>{html.escape(row['scene'])}</td>"
        f"<td>{row['success_rate']:.1%}</td>"
        f"<td>{row['median_normalized_model_error']:.3f}</td>"
        f"<td>{row['median_runtime_ms']:.3f}</td>"
        f"<td>{row['median_iterations']:.0f}</td>"
        "</tr>"
        for row in rows
    )
    (output / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Inlier Benchmarks</title>"
        "<style>body{background:#222;color:#eee;font-family:system-ui;margin:0 auto;max-width:1600px;padding:2rem}"
        "header{border-bottom:1px solid #555;margin-bottom:2rem}.primer{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1.5rem;border-bottom:1px solid #555;padding:0 0 2rem}"
        ".primer h2{margin:0 0 .4rem}.primer p{color:#c9e5b2;line-height:1.45}section{display:grid;grid-template-columns:260px minmax(0,1fr);gap:2rem;border-bottom:1px solid #555;padding:2rem 0}"
        ".label{background:#385020;padding:1rem;height:max-content}.label h2{text-transform:capitalize;margin:0}.label p{color:#c9e5b2}"
        "img{background:#fff;max-width:100%;width:100%}table{border-collapse:collapse;width:100%;margin-top:2rem}"
        "td,th{border:1px solid #666;padding:.4rem;text-align:right}td:first-child,td:nth-child(2),td:nth-child(3),td:nth-child(4),td:nth-child(5){text-align:left}"
        "@media(max-width:800px){section{grid-template-columns:1fr}}</style>"
        "<header><h1>Inlier Benchmarks</h1><p>Robust-estimation speed versus accuracy. Up and right is better.</p></header>"
        "<div class=primer><div><h2>Reading the plots</h2><p>The x-axis is success rate: a trial must reach at least 90% precision, 90% recall, and normalized model error no greater than 1. The y-axis is complete estimates per second.</p>"
        "<p>F, B, and T label the fast, balanced, and thorough iteration budgets. Pull requests and pushes publish B-only smoke points; scheduled or manually requested full runs provide the curve.</p></div>"
        "<div><h2>Scoring modes</h2><p><b>RANSAC</b> ranks hypotheses by inlier count. <b>MSAC</b> uses a truncated squared-residual cost. <b>MAGSAC</b> marginalizes uncertainty in the noise scale.</p>"
        "<p><b><a href=https://openaccess.thecvf.com/content_CVPR_2020/html/Barath_MAGSAC_a_Fast_Reliable_and_Accurate_Robust_Estimator_CVPR_2020_paper.html>MAGSAC++</a></b> is the sigma-consensus++ scoring variant: it uses a robust loss marginalized over the noise scale. This implementation evaluates that loss through a precomputed integral lookup table.</p></div></div>"
        f"{''.join(plot_sections)}<h2>Pair Diagnostics</h2><table><thead><tr>"
        "<th>Dataset</th><th>Estimator</th><th>Mode</th><th>Profile</th><th>Scene</th>"
        "<th>Success</th><th>Median model error</th><th>Median ms</th><th>Median iterations</th>"
        f"</tr></thead><tbody>{table_rows}</tbody></table>"
    )
    (output / "latest.json").write_text(json.dumps(summary, indent=2) + "\n")
    markdown = ["## Benchmark Smoke Summary", "", "| Dataset / estimator | Mode | Scene | Success | Model error | Iterations |", "|---|---|---|---:|---:|---:|"]
    for row in rows:
        markdown.append(f"| {suite_label(row['suite'])} / {row['estimator']} | {mode_label(row['scoring_mode'])} | {row['scene']} | {row['success_rate']:.1%} | {row['median_normalized_model_error']:.3f} | {row['median_iterations']:.0f} |")
    (output / "summary.md").write_text("\n".join(markdown) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])

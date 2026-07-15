#!/usr/bin/env python3
"""Create ANN-benchmarks-style quality-versus-throughput plots and dashboard."""

import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


PROFILE_ORDER = {"fast": 0, "balanced": 1, "thorough": 2}


def make_plot(rows: list[dict], title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_mode[row["scoring_mode"]].append(row)
    for mode, points in sorted(by_mode.items()):
        points.sort(key=lambda row: PROFILE_ORDER.get(row["profile"], 99))
        axis.plot(
            [point["success_rate"] for point in points],
            [1_000.0 / max(point["median_runtime_ms"], 1e-9) for point in points],
            marker="o",
            linewidth=2,
            label=mode,
        )
    axis.set_title(f"{title}: quality-throughput trade-off")
    axis.set_xlabel("Success rate")
    axis.set_ylabel("Runs per second")
    axis.set_yscale("log")
    axis.set_xlim(-0.02, 1.02)
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(title="Robust method", loc="best")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(summary_path: str, output_dir: str) -> None:
    summary = json.loads(Path(summary_path).read_text())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = summary["groups"]
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    by_problem: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_problem[(row["estimator"], row["scene"])].append(row)
    plot_sections = []
    for (estimator, scene), problem_rows in sorted(by_problem.items()):
        filename = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{estimator}-{scene}") + ".png"
        make_plot(problem_rows, f"{estimator} / {scene}", plots / filename)
        plot_sections.append(
            "<section><div class=label>"
            f"<h2>{html.escape(estimator.replace('_', ' '))}</h2>"
            f"<p>{html.escape(scene)} scene</p>"
            "<p>Right and up is better.</p></div>"
            f"<img src=plots/{html.escape(filename)} alt='{html.escape(estimator)} {html.escape(scene)} trade-off plot'></section>"
        )
    table_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['estimator'])}</td>"
        f"<td>{html.escape(row['scoring_mode'])}</td>"
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
        "header{border-bottom:1px solid #555;margin-bottom:2rem}section{display:grid;grid-template-columns:260px minmax(0,1fr);gap:2rem;border-bottom:1px solid #555;padding:2rem 0}"
        ".label{background:#385020;padding:1rem;height:max-content}.label h2{text-transform:capitalize;margin:0}.label p{color:#c9e5b2}"
        "img{background:#fff;max-width:100%;width:100%}table{border-collapse:collapse;width:100%;margin-top:2rem}"
        "td,th{border:1px solid #666;padding:.4rem;text-align:right}td:first-child,td:nth-child(2),td:nth-child(3),td:nth-child(4){text-align:left}"
        "@media(max-width:800px){section{grid-template-columns:1fr}}</style>"
        "<header><h1>Inlier Benchmarks</h1><p>Robust-estimation quality versus throughput. Right and up is better.</p></header>"
        f"{''.join(plot_sections)}<h2>Raw Summary</h2><table><thead><tr>"
        "<th>Estimator</th><th>Mode</th><th>Profile</th><th>Scene</th>"
        "<th>Success</th><th>Median model error</th><th>Median ms</th><th>Median iterations</th>"
        f"</tr></thead><tbody>{table_rows}</tbody></table>"
    )
    (output / "latest.json").write_text(json.dumps(summary, indent=2) + "\n")
    markdown = ["## Benchmark Smoke Summary", "", "| Estimator | Mode | Scene | Success | Model error | Iterations |", "|---|---|---|---:|---:|---:|"]
    for row in rows:
        markdown.append(f"| {row['estimator']} | {row['scoring_mode']} | {row['scene']} | {row['success_rate']:.1%} | {row['median_normalized_model_error']:.3f} | {row['median_iterations']:.0f} |")
    (output / "summary.md").write_text("\n".join(markdown) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])

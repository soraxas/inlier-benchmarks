#!/usr/bin/env python3
"""Create a small static dashboard and a GitHub-friendly Markdown table."""

import html
import json
import sys
from pathlib import Path


def main(summary_path: str, output_dir: str) -> None:
    summary = json.loads(Path(summary_path).read_text())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = summary["groups"]
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
        "<style>body{font-family:system-ui;margin:2rem}table{border-collapse:collapse}"
        "td,th{border:1px solid #ccc;padding:.4rem;text-align:right}td:first-child,td:nth-child(2),"
        "td:nth-child(3),td:nth-child(4){text-align:left}</style>"
        "<h1>Inlier Benchmark Results</h1><table><thead><tr>"
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

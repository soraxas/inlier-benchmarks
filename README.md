# Inlier Benchmarks

Comparative quality-versus-cost evaluation for the `inlier` public APIs.

This repository is separate from the crate: `../inlier` is the implementation
under test and `../inlier-data` owns large fixtures and ground truth. The runner
uses deterministic synthetic scenes today and records one JSON object per trial.

```bash
cargo run --release -- --suite suites/public_api.toml --output results/raw.jsonl
uv run --no-project --with numpy --with matplotlib python/aggregate.py results/raw.jsonl results/summary.json
uv run --no-project --with numpy --with matplotlib python/report.py results/summary.json site
```

Use `--smoke --seeds 5` for a PR-sized run. The default suite runs all profiles
and scenes; `--smoke` limits it to the balanced profile and adversarial outlier scene.

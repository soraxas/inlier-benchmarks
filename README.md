# Inlier Benchmarks

Comparative quality-versus-cost evaluation for the `inlier` public APIs.

This repository is separate from the crate: `../inlier` is the implementation
under test and `../inlier-data` owns large fixtures and ground truth. The runner
records one JSON object per trial for deterministic synthetic scenes and real
fixture adapters.

```bash
cargo run --release -- --suite suites/public_api.toml --output results/raw.jsonl
uv run --no-project --with numpy --with matplotlib python/aggregate.py results/raw.jsonl results/summary.json
uv run --no-project --with numpy --with matplotlib python/report.py results/summary.json site
```

Use `--smoke --seeds 5` for a PR-sized run. The default suite runs all profiles
and scenes; `--smoke` limits it to the balanced profile and adversarial outlier scene.

## PhotoTourism smoke benchmark

The real fundamental-matrix smoke test downloads the immutable HDF5 fixture
with `inlier_data`/Pooch, verifies it, extracts it locally, and prepares one
fixed 512-correspondence pair sampled across confidence levels. It does not use source images.

```bash
just prepare-phototourism
just phototourism-smoke
```

CI runs this path in addition to the synthetic matrix and appends the four
robust-scoring trials to the same JSONL report.

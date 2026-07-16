# Inlier Benchmarks

This repository benchmarks a sibling `../inlier` checkout. Large fixtures are
owned by a sibling `../inlier-data` checkout and must be fetched through its
installed `inlier_data` Pooch registry, never copied into this repository.

## Benchmark Scope

- Keep synthetic public-API coverage in `suites/public_api.toml` and the Rust
  runner. It exercises all seven supported estimation APIs and robust scoring
  modes.
- Keep real datasets in small adapter scripts under `python/`. Adapters fetch
  immutable release archives with Pooch, verify their SHA-256, and translate
  source-specific formats into small JSON inputs for Rust. Do not add HDF5 or
  dataset-specific native dependencies to the Rust benchmark runner.
- PhotoTourism is a real epipolar-geometry smoke benchmark. Its adapter uses
  `phototourism-ransac-val-v1.tar.zst`, extracts cached correspondences and
  `Fgt.h5`, ranks tutorial confidence values ascending because lower is better,
  and writes deterministic inputs. The Rust runner uses the resulting order
  with PROSAC for every scoring mode; do not shuffle these correspondences.

## Running PhotoTourism

```bash
just prepare-phototourism
just phototourism-smoke
```

The first command requires `zstd`, `tar`, `h5py`, and a local `../inlier-data`
checkout; Pooch downloads the archive once and verifies it before extraction.
The smoke path selects one pair with 512 confidence-stratified correspondences and runs every
robust scoring mode once. It is an integration guard, not a statistical full
benchmark. Scheduled and manually-dispatched full runs select eight pairs,
balanced across both scenes, and sweep fast, balanced, and thorough budgets.
Manual full runs use three repetitions for dashboard refreshes; the weekly
scheduled run uses 30 repetitions for the long statistical baseline.

For PhotoTourism fundamental-matrix trials, retain camera intrinsics, relative
pose, the estimated fundamental matrix, and selected inliers. Post-process the
timed estimator output with `python/evaluate_phototourism.py`, which mirrors
SuperRANSAC's `F -> E -> recoverPose` evaluation and reports pose `AUC@10°`.
Use that continuous metric for the primary real-data speed/accuracy plot;
success rate remains a CI gate and diagnostic field.

## CI Checkout Layout

The benchmark workflow checks out `inlier` and `inlier-data` into the workspace
and exposes sibling symlinks because the crate uses relative Cargo paths. Keep
this layout when changing CI. The composite action must run the PhotoTourism
adapter before invoking Rust with `--phototourism-input`, then append its JSONL
trials to the synthetic results before aggregation and report generation.

## Result Compatibility

Each trial is JSONL and the dashboard groups results by estimator and scene.
Scene labels can contain `/` for real data, so report filenames must be
sanitized while preserving the original label in displayed output.

The published report uses Plotly in the generated HTML for both current and
historical charts. Keep the primary real-data chart aligned with the
SuperRANSAC convention: runtime on a logarithmic x-axis, pose AUC@10 degrees
on the y-axis, plus standard-error bars and hover details. Do not reintroduce
static Matplotlib images for the dashboard.

Publish GitHub Pages only from scheduled or manually-dispatched full runs.
Smoke runs are intentionally limited to one pair and are CI diagnostics, not
comparable results; they must never overwrite the public dashboard.

## Reference Dataset Roadmap

`TODO.md` records the outstanding work to reproduce the six-dataset
SuperRANSAC evaluation: ScanNet1500, PhotoTourism, LaMAR, 7Scenes, ETH3D, and
KITTI. Before adding a fixture, update the corresponding checklist item with
the artifact, adapter, ground-truth, feature-track, and evaluation status.

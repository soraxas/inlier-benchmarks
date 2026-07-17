# Inlier Benchmarks

Comparative quality-versus-cost evaluation for the `inlier` public APIs.

This repository is separate from the crate: `../inlier` is the implementation
under test and `../inlier-data` owns large fixtures and ground truth. The runner
records one JSON object per trial for deterministic synthetic scenes and real
fixture adapters.

```bash
cargo run --release -- --suite suites/public_api.toml --output results/raw.jsonl
uv run --no-project --with numpy python/aggregate.py results/raw.jsonl results/summary.json
uv run --no-project python/report.py results/summary.json site
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

## Homography smoke benchmark

The real homography smoke test uses the immutable HPatchesSeq and EVD
validation fixture. It retains precomputed correspondences, their tutorial
confidence ordering, and the ground-truth homography, while excluding source
images. Its 3-pixel transfer threshold is applied consistently for ground
truth inlier labels and estimation.

```bash
just prepare-homography
just homography-smoke
```

## Rigid-registration smoke benchmark

The real rigid-registration smoke test uses the deterministic
`rigid_pose_example_points.txt` and `rigid_pose_example_gt.txt` fixtures from
`inlier-data`. It retains an evenly spaced 1,024-correspondence subset and
uses a 0.2-unit residual threshold calibrated to the fixture's sensor noise.

```bash
just prepare-rigid
just rigid-smoke
```

PhotoTourism's primary chart follows the SuperRANSAC convention: it ranks
matches by the tutorial's error-like confidence, applies PROSAC consistently
to every scoring mode, converts the estimated fundamental matrix to an
essential matrix using ground-truth intrinsics, recovers relative pose from the
estimated inliers, and plots pose `AUC@10°` against average estimation time.
The success-rate gate remains in the diagnostic table.

The dashboard uses Plotly directly in the published HTML, rather than static
Matplotlib images. Its primary plots expose trial-standard-error bars and
hover details without adding a Python plotting dependency.

GitHub Pages publishes only scheduled or manually dispatched full runs. Push
and pull-request smoke runs remain CI artifacts, so a single difficult pair
cannot overwrite the comparable dashboard with a zero-AUC point.

Regression history records the checked-out `inlier` implementation SHA, not
the benchmark harness SHA. Its commit-axis points open the tested source
revision.

## Reference-comparison roadmap

The current PhotoTourism fixture is an `inlier` API benchmark, not a numerical
reproduction of SuperRANSAC's aggregate paper result. See [TODO.md](TODO.md)
for the six-dataset, two-feature-track work required for a like-for-like
comparison.

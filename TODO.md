# Reference Comparison Roadmap

The current dashboard measures `inlier` on deterministic synthetic inputs and
the RANSAC Tutorial PhotoTourism validation fixture. It is intentionally not
presented as a reproduction of SuperRANSAC's six-dataset paper aggregate.

For a like-for-like comparison, each dataset below needs an immutable
`inlier-data` release artifact, a Pooch registry entry, a small adapter under
`python/`, and ground-truth pose evaluation. Both correspondence tracks must
be stored or generated reproducibly: SuperPoint + LightGlue and RoMA. Preserve
the original score direction and reproduce the reference pair split,
correspondence limits, thresholds, and metric calculation.

- [ ] **ScanNet1500**: add the 1,500-pair evaluation split, intrinsics and
  relative-pose ground truth; implement fundamental, essential, and homography
  inputs for both feature tracks.
- [ ] **PhotoTourism**: replace the eight-pair tutorial smoke subset with the
  full reference split and reference SuperPoint + LightGlue and RoMA
  correspondences; retain the existing fixture as a small API smoke test.
- [ ] **LaMAR**: package the reference query/database pair manifests and poses;
  add deterministic correspondence artifacts for both feature tracks.
- [ ] **7Scenes**: add the reference image-pair split, calibrated poses, and
  both correspondence tracks.
- [ ] **ETH3D**: add the reference image-pair split, calibration and poses;
  cover the outdoor/indoor partition used by the reference evaluation.
- [ ] **KITTI**: add the reference pair split and camera calibration/relative
  poses, then generate or store both correspondence tracks.

After all six adapters are available, add an aggregate report that weights all
individual image pairs equally, plots pose AUC@10 degrees against mean runtime,
and keeps per-dataset plots available for diagnosis. The existing native
benchmark remains a separate dashboard section because it answers a different
question: performance on the crate's public APIs with supplied
correspondences.

use clap::Parser;
use inlier::{
    MetasacSettings, estimate_absolute_pose, estimate_essential_matrix,
    estimate_fundamental_matrix, estimate_homography, estimate_line, estimate_plane,
    estimate_rigid_transform,
    settings::{SamplerType, ScoringType},
    types::DataMatrix,
};
use nalgebra::{Matrix3, Rotation3, Vector3};
use serde::{Deserialize, Serialize};
use std::{fs, io::Write, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "suites/public_api.toml")]
    suite: PathBuf,
    #[arg(long, default_value = "results/raw.jsonl")]
    output: PathBuf,
    #[arg(long, default_value_t = 30)]
    seeds: usize,
    #[arg(long)]
    smoke: bool,
    #[arg(long)]
    phototourism_input: Option<PathBuf>,
    #[arg(long)]
    homography_input: Option<PathBuf>,
    #[arg(long)]
    rigid_input: Option<PathBuf>,
}

#[derive(Deserialize)]
struct Suite {
    version: u32,
    name: String,
    estimators: Vec<String>,
    scoring_modes: Vec<String>,
    samplers: Vec<String>,
    scenes: Vec<String>,
    profiles: Vec<String>,
}

#[derive(Serialize)]
struct Trial {
    suite: String,
    suite_version: u32,
    estimator: String,
    scoring_mode: String,
    sampler: String,
    profile: String,
    scene: String,
    seed: u64,
    runtime_ms: f64,
    iterations: usize,
    inlier_precision: f64,
    inlier_recall: f64,
    normalized_model_error: f64,
    inlier_classification_error: f64,
    epipolar_matrix: Option<[[f64; 3]; 3]>,
    inlier_indices: Option<Vec<usize>>,
    homography_auc_3: Option<f64>,
    diagnostics: Option<TrialDiagnostics>,
    success: bool,
    failure_reason: Option<String>,
}

/// Execution counters emitted by the estimator, normalized for JSONL output.
#[derive(Clone, Serialize)]
struct TrialDiagnostics {
    sampling_attempts: usize,
    rejected_samples: usize,
    model_estimation_failures: usize,
    candidate_models: usize,
    rejected_models: usize,
    scored_models: usize,
    local_optimization_runs: usize,
    final_optimization_runs: usize,
    inlier_ratio: f64,
}

#[derive(Deserialize)]
struct PhototourismInput {
    schema_version: u32,
    dataset: String,
    threshold: f64,
    pairs: Vec<PhototourismPair>,
}

#[derive(Deserialize)]
struct PhototourismPair {
    scene: String,
    pair: String,
    points1: Vec<[f64; 2]>,
    points2: Vec<[f64; 2]>,
    match_scores: Vec<f64>,
    fundamental: [[f64; 3]; 3],
    essential: [[f64; 3]; 3],
    intrinsics1: [[f64; 3]; 3],
    intrinsics2: [[f64; 3]; 3],
}

#[derive(Deserialize)]
struct HomographyInput {
    schema_version: u32,
    dataset: String,
    threshold: f64,
    pairs: Vec<HomographyPair>,
}

#[derive(Deserialize)]
struct HomographyPair {
    dataset: String,
    pair: String,
    points1: Vec<[f64; 2]>,
    points2: Vec<[f64; 2]>,
    homography: [[f64; 3]; 3],
}

#[derive(Deserialize)]
struct RigidInput {
    schema_version: u32,
    dataset: String,
    threshold: f64,
    pairs: Vec<RigidPair>,
}

#[derive(Deserialize)]
struct RigidPair {
    scene: String,
    points_src: Vec<[f64; 3]>,
    points_tgt: Vec<[f64; 3]>,
    transform: [[f64; 4]; 4],
}

#[derive(Clone, Copy)]
enum Scene {
    Clean,
    Noisy,
    Outliers,
}
impl Scene {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "clean" => Some(Self::Clean),
            "noisy" => Some(Self::Noisy),
            "outliers" => Some(Self::Outliers),
            _ => None,
        }
    }
    fn noise(self) -> f64 {
        if matches!(self, Self::Clean) {
            0.0
        } else {
            0.0001
        }
    }
    fn outlier(self, index: usize) -> bool {
        matches!(self, Self::Outliers) && index.is_multiple_of(4)
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((self.0 >> 11) as f64) / ((1_u64 << 53) as f64)
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
    fn noise(&mut self, scale: f64) -> f64 {
        self.range(-scale, scale)
    }
}

fn settings(
    profile: &str,
    scoring: &str,
    sampler: &str,
    seed: u64,
) -> Result<MetasacSettings, String> {
    let (iterations, confidence) = match profile {
        "fast" => (250, 0.95),
        "balanced" => (1_000, 0.99),
        "thorough" => (5_000, 0.999),
        _ => return Err(format!("unknown profile {profile}")),
    };
    let scoring = match scoring {
        "ransac" => ScoringType::Ransac,
        "msac" => ScoringType::Msac,
        "magsac" => ScoringType::Magsac,
        "magsac_pp" => ScoringType::MagsacPlusPlus,
        _ => return Err(format!("unknown scoring mode {scoring}")),
    };
    let sampler = match sampler {
        "uniform" => SamplerType::Uniform,
        "prosac" => SamplerType::Prosac,
        _ => return Err(format!("unknown sampler {sampler}")),
    };
    Ok(MetasacSettings {
        min_iterations: iterations,
        max_iterations: iterations,
        // Benchmarks charge each failed minimal solve to the configured
        // hypothesis budget. Retrying a pathological sample internally can
        // otherwise multiply a 5,000-iteration profile into a long run.
        max_sampling_attempts: 1,
        confidence,
        rng_seed: Some(seed),
        sampler,
        scoring,
        ..Default::default()
    })
}

fn image(scene: Scene, seed: u64) -> (DataMatrix, DataMatrix, Vec<bool>) {
    let mut rng = Rng(seed);
    let n = 96;
    let r = Rotation3::from_euler_angles(0.03, -0.12, 0.02);
    let t = Vector3::new(0.35, -0.08, 0.12);
    let mut a = DataMatrix::zeros(n, 2);
    let mut b = DataMatrix::zeros(n, 2);
    let mut truth = Vec::with_capacity(n);
    for i in 0..n {
        let p = Vector3::new(
            rng.range(-1.5, 1.5),
            rng.range(-1.0, 1.0),
            rng.range(4.0, 8.0),
        );
        let q = r * p + t;
        let out = scene.outlier(i);
        a.set(i, 0, p.x / p.z + rng.noise(scene.noise()));
        a.set(i, 1, p.y / p.z + rng.noise(scene.noise()));
        b.set(
            i,
            0,
            if out {
                rng.range(-1.0, 1.0)
            } else {
                q.x / q.z + rng.noise(scene.noise())
            },
        );
        b.set(
            i,
            1,
            if out {
                rng.range(-1.0, 1.0)
            } else {
                q.y / q.z + rng.noise(scene.noise())
            },
        );
        truth.push(!out);
    }
    (a, b, truth)
}
fn homography(scene: Scene, seed: u64) -> (DataMatrix, DataMatrix, Vec<bool>) {
    let mut rng = Rng(seed);
    let n = 96;
    let h = Matrix3::new(1.08, -0.06, 12.0, 0.04, 0.94, -8.0, 0.0008, -0.0005, 1.0);
    let mut a = DataMatrix::zeros(n, 2);
    let mut b = DataMatrix::zeros(n, 2);
    let mut truth = Vec::with_capacity(n);
    for i in 0..n {
        let x = rng.range(-100.0, 100.0);
        let y = rng.range(-80.0, 80.0);
        let q = h * Vector3::new(x, y, 1.0);
        let out = scene.outlier(i);
        a.set(i, 0, x);
        a.set(i, 1, y);
        b.set(
            i,
            0,
            if out {
                rng.range(-150.0, 150.0)
            } else {
                q.x / q.z + rng.noise(scene.noise() * 100.0)
            },
        );
        b.set(
            i,
            1,
            if out {
                rng.range(-150.0, 150.0)
            } else {
                q.y / q.z + rng.noise(scene.noise() * 100.0)
            },
        );
        truth.push(!out);
    }
    (a, b, truth)
}
fn line(scene: Scene, seed: u64) -> (DataMatrix, Vec<bool>) {
    let mut rng = Rng(seed);
    let n = 96;
    let mut d = DataMatrix::zeros(n, 2);
    let mut t = Vec::with_capacity(n);
    for i in 0..n {
        let x = rng.range(-20., 20.);
        let out = scene.outlier(i);
        d.set(i, 0, x);
        d.set(
            i,
            1,
            if out {
                rng.range(-30., 30.)
            } else {
                0.7 * x + 1.5 + rng.noise(scene.noise() * 10.)
            },
        );
        t.push(!out);
    }
    (d, t)
}
fn plane(scene: Scene, seed: u64) -> (DataMatrix, Vec<bool>) {
    let mut rng = Rng(seed);
    let n = 96;
    let mut d = DataMatrix::zeros(n, 3);
    let mut t = Vec::with_capacity(n);
    for i in 0..n {
        let x = rng.range(-20., 20.);
        let y = rng.range(-20., 20.);
        let out = scene.outlier(i);
        d.set(i, 0, x);
        d.set(i, 1, y);
        d.set(
            i,
            2,
            if out {
                rng.range(-30., 30.)
            } else {
                0.25 * x - 0.4 * y + 3. + rng.noise(scene.noise() * 10.)
            },
        );
        t.push(!out);
    }
    (d, t)
}
fn rigid(scene: Scene, seed: u64) -> (DataMatrix, DataMatrix, Vec<bool>) {
    let mut rng = Rng(seed);
    let n = 96;
    let r = Rotation3::from_euler_angles(0.15, -0.2, 0.1);
    let tr = Vector3::new(3., -2., 1.);
    let mut a = DataMatrix::zeros(n, 3);
    let mut b = DataMatrix::zeros(n, 3);
    let mut truth = Vec::with_capacity(n);
    for i in 0..n {
        let p = Vector3::new(
            rng.range(-10., 10.),
            rng.range(-10., 10.),
            rng.range(-10., 10.),
        );
        let q = r * p + tr;
        let out = scene.outlier(i);
        for j in 0..3 {
            a.set(i, j, p[j]);
            b.set(
                i,
                j,
                if out {
                    rng.range(-20., 20.)
                } else {
                    q[j] + rng.noise(scene.noise() * 10.)
                },
            );
        }
        truth.push(!out);
    }
    (a, b, truth)
}
fn pose(scene: Scene, seed: u64) -> (DataMatrix, DataMatrix, Vec<bool>) {
    let mut rng = Rng(seed);
    let n = 64;
    let mut w = DataMatrix::zeros(n, 3);
    let mut im = DataMatrix::zeros(n, 2);
    let mut truth = Vec::with_capacity(n);
    for i in 0..n {
        let p = Vector3::new(rng.range(-2., 2.), rng.range(-1.5, 1.5), rng.range(4., 8.));
        let out = scene.outlier(i);
        for j in 0..3 {
            w.set(i, j, p[j]);
        }
        im.set(
            i,
            0,
            if out {
                rng.range(-1., 1.)
            } else {
                p.x / p.z + rng.noise(scene.noise())
            },
        );
        im.set(
            i,
            1,
            if out {
                rng.range(-1., 1.)
            } else {
                p.y / p.z + rng.noise(scene.noise())
            },
        );
        truth.push(!out);
    }
    (w, im, truth)
}

fn classification_score(inliers: &[usize], truth: &[bool]) -> (f64, f64, f64) {
    let mut selected = vec![false; truth.len()];
    for &i in inliers {
        if i < selected.len() {
            selected[i] = true;
        }
    }
    let tp = (0..truth.len())
        .filter(|&i| truth[i] && selected[i])
        .count() as f64;
    let fp = (0..truth.len())
        .filter(|&i| !truth[i] && selected[i])
        .count() as f64;
    let fn_ = (0..truth.len())
        .filter(|&i| truth[i] && !selected[i])
        .count() as f64;
    let p = if tp + fp == 0. { 0. } else { tp / (tp + fp) };
    let r = if tp + fn_ == 0. { 0. } else { tp / (tp + fn_) };
    let f1 = if p + r == 0. {
        0.
    } else {
        2. * p * r / (p + r)
    };
    (p, r, 1. - f1)
}

fn normalized_median_residual<F>(truth: &[bool], threshold: f64, residual: F) -> f64
where
    F: Fn(usize) -> f64,
{
    let mut residuals: Vec<f64> = (0..truth.len())
        .filter(|&index| truth[index])
        .map(residual)
        .filter(|value| value.is_finite())
        .collect();
    if residuals.is_empty() || threshold <= 0.0 {
        return f64::MAX;
    }
    residuals.sort_by(f64::total_cmp);
    residuals[residuals.len() / 2] / threshold
}

struct Outcome {
    inliers: Vec<usize>,
    iterations: usize,
    truth: Vec<bool>,
    normalized_model_error: f64,
    epipolar_matrix: Option<[[f64; 3]; 3]>,
    homography_auc_3: Option<f64>,
    diagnostics: TrialDiagnostics,
}

fn trial_diagnostics(
    diagnostics: &inlier::core::MetaSacDiagnostics,
    inlier_count: usize,
    point_count: usize,
) -> TrialDiagnostics {
    TrialDiagnostics {
        sampling_attempts: diagnostics.sampling_attempts,
        rejected_samples: diagnostics.rejected_samples,
        model_estimation_failures: diagnostics.model_estimation_failures,
        candidate_models: diagnostics.candidate_models,
        rejected_models: diagnostics.rejected_models,
        scored_models: diagnostics.scored_models,
        local_optimization_runs: diagnostics.local_optimization_runs,
        final_optimization_runs: diagnostics.final_optimization_runs,
        inlier_ratio: inlier_count as f64 / point_count.max(1) as f64,
    }
}

fn run(
    estimator: &str,
    scene: Scene,
    settings: MetasacSettings,
    seed: u64,
) -> Result<Outcome, String> {
    match estimator {
        "homography" => {
            let (a, b, t) = homography(scene, seed);
            let r = estimate_homography(&a, &b, 0.5, Some(settings))?;
            let error = normalized_median_residual(&t, 0.5, |index| {
                let source = Vector3::new(a.get(index, 0), a.get(index, 1), 1.0);
                let projected = r.model.h * source;
                if projected.z.abs() < 1e-12 {
                    return f64::MAX;
                }
                let target = Vector3::new(b.get(index, 0), b.get(index, 1), 1.0);
                (Vector3::new(projected.x / projected.z, projected.y / projected.z, 1.0) - target)
                    .norm()
            });
            Ok(Outcome {
                diagnostics: trial_diagnostics(&r.diagnostics, r.inliers.len(), t.len()),
                inliers: r.inliers,
                iterations: r.iterations,
                truth: t,
                normalized_model_error: error,
                epipolar_matrix: None,
                homography_auc_3: None,
            })
        }
        "fundamental" => {
            let (a, b, t) = image(scene, seed);
            let r = estimate_fundamental_matrix(&a, &b, 0.01, Some(settings))?;
            let error = normalized_median_residual(&t, 0.01, |index| {
                inlier::bundle_adjustment::sampson_error(
                    &r.model.f,
                    &nalgebra::Vector2::new(a.get(index, 0), a.get(index, 1)),
                    &nalgebra::Vector2::new(b.get(index, 0), b.get(index, 1)),
                )
                .abs()
            });
            Ok(Outcome {
                diagnostics: trial_diagnostics(&r.diagnostics, r.inliers.len(), t.len()),
                inliers: r.inliers,
                iterations: r.iterations,
                truth: t,
                normalized_model_error: error,
                epipolar_matrix: None,
                homography_auc_3: None,
            })
        }
        "essential" => {
            let (a, b, t) = image(scene, seed);
            let r = estimate_essential_matrix(&a, &b, 0.01, Some(settings))?;
            let error = normalized_median_residual(&t, 0.01, |index| {
                inlier::bundle_adjustment::sampson_error(
                    &r.model.e,
                    &nalgebra::Vector2::new(a.get(index, 0), a.get(index, 1)),
                    &nalgebra::Vector2::new(b.get(index, 0), b.get(index, 1)),
                )
                .abs()
            });
            Ok(Outcome {
                diagnostics: trial_diagnostics(&r.diagnostics, r.inliers.len(), t.len()),
                inliers: r.inliers,
                iterations: r.iterations,
                truth: t,
                normalized_model_error: error,
                epipolar_matrix: None,
                homography_auc_3: None,
            })
        }
        "absolute_pose" => {
            let (a, b, t) = pose(scene, seed);
            let r = estimate_absolute_pose(&a, &b, 0.01, Some(settings))?;
            let error = normalized_median_residual(&t, 0.01, |index| {
                inlier::bundle_adjustment::reprojection_error(
                    r.model.rotation.to_rotation_matrix().matrix(),
                    &r.model.translation.vector,
                    &nalgebra::Vector2::new(b.get(index, 0), b.get(index, 1)),
                    &Vector3::new(a.get(index, 0), a.get(index, 1), a.get(index, 2)),
                )
            });
            Ok(Outcome {
                diagnostics: trial_diagnostics(&r.diagnostics, r.inliers.len(), t.len()),
                inliers: r.inliers,
                iterations: r.iterations,
                truth: t,
                normalized_model_error: error,
                epipolar_matrix: None,
                homography_auc_3: None,
            })
        }
        "line" => {
            let (a, t) = line(scene, seed);
            let r = estimate_line(&a, 0.05, Some(settings))?;
            let error = normalized_median_residual(&t, 0.05, |index| {
                r.model.distance_to_point(a.get(index, 0), a.get(index, 1))
            });
            Ok(Outcome {
                diagnostics: trial_diagnostics(&r.diagnostics, r.inliers.len(), t.len()),
                inliers: r.inliers,
                iterations: r.iterations,
                truth: t,
                normalized_model_error: error,
                epipolar_matrix: None,
                homography_auc_3: None,
            })
        }
        "plane" => {
            let (a, t) = plane(scene, seed);
            let r = estimate_plane(&a, 0.05, Some(settings))?;
            let error = normalized_median_residual(&t, 0.05, |index| {
                r.model
                    .distance(a.get(index, 0), a.get(index, 1), a.get(index, 2))
            });
            Ok(Outcome {
                diagnostics: trial_diagnostics(&r.diagnostics, r.inliers.len(), t.len()),
                inliers: r.inliers,
                iterations: r.iterations,
                truth: t,
                normalized_model_error: error,
                epipolar_matrix: None,
                homography_auc_3: None,
            })
        }
        "rigid_transform" => {
            let (a, b, t) = rigid(scene, seed);
            let r = estimate_rigid_transform(&a, &b, 0.05, Some(settings))?;
            let error = normalized_median_residual(&t, 0.05, |index| {
                let source =
                    nalgebra::Point3::new(a.get(index, 0), a.get(index, 1), a.get(index, 2));
                let target = Vector3::new(b.get(index, 0), b.get(index, 1), b.get(index, 2));
                (target
                    - (r.model.rotation.transform_point(&source).coords
                        + r.model.translation.vector))
                    .norm()
            });
            Ok(Outcome {
                diagnostics: trial_diagnostics(&r.diagnostics, r.inliers.len(), t.len()),
                inliers: r.inliers,
                iterations: r.iterations,
                truth: t,
                normalized_model_error: error,
                epipolar_matrix: None,
                homography_auc_3: None,
            })
        }
        _ => Err(format!("unknown estimator {estimator}")),
    }
}

fn matrix_from_array(values: [[f64; 3]; 3]) -> Matrix3<f64> {
    Matrix3::new(
        values[0][0],
        values[0][1],
        values[0][2],
        values[1][0],
        values[1][1],
        values[1][2],
        values[2][0],
        values[2][1],
        values[2][2],
    )
}

fn matrix_to_array(matrix: &Matrix3<f64>) -> [[f64; 3]; 3] {
    [
        [matrix[(0, 0)], matrix[(0, 1)], matrix[(0, 2)]],
        [matrix[(1, 0)], matrix[(1, 1)], matrix[(1, 2)]],
        [matrix[(2, 0)], matrix[(2, 1)], matrix[(2, 2)]],
    ]
}

fn homography_transfer_error(homography: &Matrix3<f64>, source: [f64; 2], target: [f64; 2]) -> f64 {
    let projected = homography * Vector3::new(source[0], source[1], 1.0);
    if !projected.z.is_finite() || projected.z.abs() < 1e-12 {
        return f64::MAX;
    }
    let estimate = Vector3::new(projected.x / projected.z, projected.y / projected.z, 1.0);
    (estimate - Vector3::new(target[0], target[1], 1.0)).norm()
}

fn auc_at_threshold(errors: &[f64], threshold: f64) -> f64 {
    if errors.is_empty() || threshold <= 0.0 {
        return 0.0;
    }
    let mut sorted: Vec<f64> = errors
        .iter()
        .copied()
        .filter(|error| error.is_finite() && *error >= 0.0)
        .collect();
    if sorted.is_empty() {
        return 0.0;
    }
    sorted.sort_by(f64::total_cmp);
    let mut previous_error = 0.0;
    let mut previous_recall = 0.0;
    let mut area = 0.0;
    for (index, error) in sorted.iter().enumerate() {
        if *error > threshold {
            break;
        }
        let recall = (index + 1) as f64 / sorted.len() as f64;
        area += (error - previous_error) * (previous_recall + recall) * 0.5;
        previous_error = *error;
        previous_recall = recall;
    }
    area += (threshold - previous_error) * previous_recall;
    area / threshold
}

fn run_homography_fixture(
    pair: &HomographyPair,
    threshold: f64,
    settings: MetasacSettings,
) -> Result<Outcome, String> {
    if pair.points1.len() != pair.points2.len() || pair.points1.len() < 4 {
        return Err(
            "Homography pair must contain matching point arrays with at least 4 rows".into(),
        );
    }
    let mut points1 = DataMatrix::zeros(pair.points1.len(), 2);
    let mut points2 = DataMatrix::zeros(pair.points2.len(), 2);
    for (index, (source, target)) in pair.points1.iter().zip(&pair.points2).enumerate() {
        points1.set(index, 0, source[0]);
        points1.set(index, 1, source[1]);
        points2.set(index, 0, target[0]);
        points2.set(index, 1, target[1]);
    }
    let ground_truth = matrix_from_array(pair.homography);
    if !ground_truth.iter().all(|value| value.is_finite()) {
        return Err("Homography ground truth contains a non-finite value".into());
    }
    let truth: Vec<bool> = pair
        .points1
        .iter()
        .zip(&pair.points2)
        .map(|(source, target)| {
            homography_transfer_error(&ground_truth, *source, *target) <= threshold
        })
        .collect();
    let result = estimate_homography(&points1, &points2, threshold, Some(settings))?;
    let error = normalized_median_residual(&truth, threshold, |index| {
        homography_transfer_error(&result.model.h, pair.points1[index], pair.points2[index])
    });
    let ground_truth_errors: Vec<f64> = pair
        .points1
        .iter()
        .map(|source| {
            let target = ground_truth * Vector3::new(source[0], source[1], 1.0);
            homography_transfer_error(
                &result.model.h,
                *source,
                [target.x / target.z, target.y / target.z],
            )
        })
        .collect();
    Ok(Outcome {
        diagnostics: trial_diagnostics(&result.diagnostics, result.inliers.len(), truth.len()),
        inliers: result.inliers,
        iterations: result.iterations,
        truth,
        normalized_model_error: error,
        epipolar_matrix: None,
        homography_auc_3: Some(auc_at_threshold(&ground_truth_errors, threshold)),
    })
}

fn run_rigid_fixture(
    pair: &RigidPair,
    threshold: f64,
    settings: MetasacSettings,
) -> Result<Outcome, String> {
    if pair.points_src.len() != pair.points_tgt.len() || pair.points_src.len() < 3 {
        return Err(
            "rigid pair must contain matching 3D point arrays with at least three rows".into(),
        );
    }
    let mut source = DataMatrix::zeros(pair.points_src.len(), 3);
    let mut target = DataMatrix::zeros(pair.points_tgt.len(), 3);
    for (index, (src, tgt)) in pair.points_src.iter().zip(&pair.points_tgt).enumerate() {
        for dimension in 0..3 {
            source.set(index, dimension, src[dimension]);
            target.set(index, dimension, tgt[dimension]);
        }
    }
    let rotation = Matrix3::new(
        pair.transform[0][0],
        pair.transform[0][1],
        pair.transform[0][2],
        pair.transform[1][0],
        pair.transform[1][1],
        pair.transform[1][2],
        pair.transform[2][0],
        pair.transform[2][1],
        pair.transform[2][2],
    );
    let translation = Vector3::new(
        pair.transform[0][3],
        pair.transform[1][3],
        pair.transform[2][3],
    );
    let truth: Vec<bool> = (0..pair.points_src.len())
        .map(|index| {
            let src = Vector3::new(
                source.get(index, 0),
                source.get(index, 1),
                source.get(index, 2),
            );
            let tgt = Vector3::new(
                target.get(index, 0),
                target.get(index, 1),
                target.get(index, 2),
            );
            (rotation * src + translation - tgt).norm() <= threshold
        })
        .collect();
    let result = estimate_rigid_transform(&source, &target, threshold, Some(settings))?;
    let error = normalized_median_residual(&truth, threshold, |index| {
        let src = nalgebra::Point3::new(
            source.get(index, 0),
            source.get(index, 1),
            source.get(index, 2),
        );
        let tgt = Vector3::new(
            target.get(index, 0),
            target.get(index, 1),
            target.get(index, 2),
        );
        (result.model.rotation.transform_point(&src).coords + result.model.translation.vector - tgt)
            .norm()
    });
    Ok(Outcome {
        diagnostics: trial_diagnostics(&result.diagnostics, result.inliers.len(), truth.len()),
        inliers: result.inliers,
        iterations: result.iterations,
        truth,
        normalized_model_error: error,
        epipolar_matrix: None,
        homography_auc_3: None,
    })
}

fn run_phototourism(
    estimator: &str,
    pair: &PhototourismPair,
    threshold: f64,
    settings: MetasacSettings,
) -> Result<Outcome, String> {
    if pair.points1.len() != pair.points2.len() || pair.points1.len() < 8 {
        return Err(
            "PhotoTourism pair must contain matching point arrays with at least 8 rows".into(),
        );
    }
    if pair.match_scores.len() != pair.points1.len()
        || pair.match_scores.iter().any(|score| !score.is_finite())
        || pair
            .match_scores
            .windows(2)
            .any(|scores| scores[0] > scores[1])
    {
        return Err("PhotoTourism match scores must be finite and sorted best-first".into());
    }
    let mut points1 = DataMatrix::zeros(pair.points1.len(), 2);
    let mut points2 = DataMatrix::zeros(pair.points2.len(), 2);
    for (index, (source, target)) in pair.points1.iter().zip(&pair.points2).enumerate() {
        points1.set(index, 0, source[0]);
        points1.set(index, 1, source[1]);
        points2.set(index, 0, target[0]);
        points2.set(index, 1, target[1]);
    }
    if estimator == "fundamental" {
        let ground_truth = matrix_from_array(pair.fundamental);
        let truth: Vec<bool> = (0..pair.points1.len())
            .map(|index| {
                inlier::bundle_adjustment::sampson_error(
                    &ground_truth,
                    &nalgebra::Vector2::new(points1.get(index, 0), points1.get(index, 1)),
                    &nalgebra::Vector2::new(points2.get(index, 0), points2.get(index, 1)),
                )
                .abs()
                    <= threshold
            })
            .collect();
        let result = estimate_fundamental_matrix(&points1, &points2, threshold, Some(settings))?;
        let error = normalized_median_residual(&truth, threshold, |index| {
            inlier::bundle_adjustment::sampson_error(
                &result.model.f,
                &nalgebra::Vector2::new(points1.get(index, 0), points1.get(index, 1)),
                &nalgebra::Vector2::new(points2.get(index, 0), points2.get(index, 1)),
            )
            .abs()
        });
        return Ok(Outcome {
            diagnostics: trial_diagnostics(&result.diagnostics, result.inliers.len(), truth.len()),
            inliers: result.inliers,
            iterations: result.iterations,
            truth,
            normalized_model_error: error,
            epipolar_matrix: Some(matrix_to_array(&result.model.f)),
            homography_auc_3: None,
        });
    }

    if estimator != "essential" {
        return Err(format!("unsupported PhotoTourism estimator {estimator}"));
    }
    let intrinsics1 = matrix_from_array(pair.intrinsics1);
    let intrinsics2 = matrix_from_array(pair.intrinsics2);
    let inverse1 = intrinsics1
        .try_inverse()
        .ok_or("PhotoTourism intrinsics1 is singular")?;
    let inverse2 = intrinsics2
        .try_inverse()
        .ok_or("PhotoTourism intrinsics2 is singular")?;
    for index in 0..pair.points1.len() {
        let source = inverse1 * Vector3::new(points1.get(index, 0), points1.get(index, 1), 1.0);
        let target = inverse2 * Vector3::new(points2.get(index, 0), points2.get(index, 1), 1.0);
        points1.set(index, 0, source.x / source.z);
        points1.set(index, 1, source.y / source.z);
        points2.set(index, 0, target.x / target.z);
        points2.set(index, 1, target.y / target.z);
    }
    let normalized_threshold = threshold
        / ((intrinsics1[(0, 0)] + intrinsics1[(1, 1)] + intrinsics2[(0, 0)] + intrinsics2[(1, 1)])
            / 4.0);
    let ground_truth = matrix_from_array(pair.essential);
    let truth: Vec<bool> = (0..pair.points1.len())
        .map(|index| {
            inlier::bundle_adjustment::sampson_error(
                &ground_truth,
                &nalgebra::Vector2::new(points1.get(index, 0), points1.get(index, 1)),
                &nalgebra::Vector2::new(points2.get(index, 0), points2.get(index, 1)),
            )
            .abs()
                <= normalized_threshold
        })
        .collect();
    let result =
        estimate_essential_matrix(&points1, &points2, normalized_threshold, Some(settings))?;
    let error = normalized_median_residual(&truth, normalized_threshold, |index| {
        inlier::bundle_adjustment::sampson_error(
            &result.model.e,
            &nalgebra::Vector2::new(points1.get(index, 0), points1.get(index, 1)),
            &nalgebra::Vector2::new(points2.get(index, 0), points2.get(index, 1)),
        )
        .abs()
    });
    Ok(Outcome {
        diagnostics: trial_diagnostics(&result.diagnostics, result.inliers.len(), truth.len()),
        inliers: result.inliers,
        iterations: result.iterations,
        truth,
        normalized_model_error: error,
        epipolar_matrix: Some(matrix_to_array(&result.model.e)),
        homography_auc_3: None,
    })
}

fn run_phototourism_suite(
    input: &PhototourismInput,
    suite: &Suite,
    args: &Args,
    out: &mut fs::File,
) -> Result<(), Box<dyn std::error::Error>> {
    if input.schema_version != 1 || input.threshold <= 0.0 {
        return Err("unsupported PhotoTourism input schema or threshold".into());
    }
    let profiles: Vec<&String> = if args.smoke {
        suite
            .profiles
            .iter()
            .filter(|profile| profile.as_str() == "balanced")
            .collect()
    } else {
        suite.profiles.iter().collect()
    };
    for estimator in ["fundamental", "essential"] {
        for pair in &input.pairs {
            for mode in &suite.scoring_modes {
                for sampler in &suite.samplers {
                    for profile in &profiles {
                        for index in 0..args.seeds {
                            let seed = 0x5EED_CAFE_D00D_BAAD_u64 ^ (index as u64);
                            let start = Instant::now();
                            let result =
                                settings(profile, mode, sampler, seed).and_then(|settings| {
                                    run_phototourism(estimator, pair, input.threshold, settings)
                                });
                            let runtime_ms = start.elapsed().as_secs_f64() * 1000.;
                            let trial = match result {
                                Ok(outcome) => {
                                    let (precision, recall, classification_error) =
                                        classification_score(&outcome.inliers, &outcome.truth);
                                    Trial {
                                        suite: input.dataset.clone(),
                                        suite_version: input.schema_version,
                                        estimator: estimator.into(),
                                        scoring_mode: mode.clone(),
                                        sampler: sampler.clone(),
                                        profile: (*profile).clone(),
                                        scene: format!("{}/{}", pair.scene, pair.pair),
                                        seed,
                                        runtime_ms,
                                        iterations: outcome.iterations,
                                        inlier_precision: precision,
                                        inlier_recall: recall,
                                        normalized_model_error: outcome.normalized_model_error,
                                        inlier_classification_error: classification_error,
                                        epipolar_matrix: outcome.epipolar_matrix,
                                        inlier_indices: Some(outcome.inliers),
                                        homography_auc_3: outcome.homography_auc_3,
                                        diagnostics: Some(outcome.diagnostics),
                                        success: precision >= 0.9
                                            && recall >= 0.9
                                            && outcome.normalized_model_error <= 1.0,
                                        failure_reason: None,
                                    }
                                }
                                Err(reason) => Trial {
                                    suite: input.dataset.clone(),
                                    suite_version: input.schema_version,
                                    estimator: estimator.into(),
                                    scoring_mode: mode.clone(),
                                    sampler: sampler.clone(),
                                    profile: (*profile).clone(),
                                    scene: format!("{}/{}", pair.scene, pair.pair),
                                    seed,
                                    runtime_ms,
                                    iterations: 0,
                                    inlier_precision: 0.,
                                    inlier_recall: 0.,
                                    normalized_model_error: f64::MAX,
                                    inlier_classification_error: 1.,
                                    epipolar_matrix: None,
                                    inlier_indices: None,
                                    homography_auc_3: None,
                                    diagnostics: None,
                                    success: false,
                                    failure_reason: Some(reason),
                                },
                            };
                            writeln!(out, "{}", serde_json::to_string(&trial)?)?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn run_homography_suite(
    input: &HomographyInput,
    suite: &Suite,
    args: &Args,
    out: &mut fs::File,
) -> Result<(), Box<dyn std::error::Error>> {
    if input.schema_version != 1 || input.threshold <= 0.0 {
        return Err("unsupported homography input schema or threshold".into());
    }
    let profiles: Vec<&String> = if args.smoke {
        suite
            .profiles
            .iter()
            .filter(|profile| profile.as_str() == "balanced")
            .collect()
    } else {
        suite.profiles.iter().collect()
    };
    for pair in &input.pairs {
        for mode in &suite.scoring_modes {
            for sampler in &suite.samplers {
                for profile in &profiles {
                    for index in 0..args.seeds {
                        let seed = 0x5EED_CAFE_D00D_BAAD_u64 ^ (index as u64);
                        let start = Instant::now();
                        let result = settings(profile, mode, sampler, seed).and_then(|settings| {
                            run_homography_fixture(pair, input.threshold, settings)
                        });
                        let runtime_ms = start.elapsed().as_secs_f64() * 1000.;
                        let trial = match result {
                            Ok(outcome) => {
                                let (precision, recall, classification_error) =
                                    classification_score(&outcome.inliers, &outcome.truth);
                                Trial {
                                    suite: input.dataset.clone(),
                                    suite_version: input.schema_version,
                                    estimator: "homography".into(),
                                    scoring_mode: mode.clone(),
                                    sampler: sampler.clone(),
                                    profile: (*profile).clone(),
                                    scene: format!("{}/{}", pair.dataset, pair.pair),
                                    seed,
                                    runtime_ms,
                                    iterations: outcome.iterations,
                                    inlier_precision: precision,
                                    inlier_recall: recall,
                                    normalized_model_error: outcome.normalized_model_error,
                                    inlier_classification_error: classification_error,
                                    epipolar_matrix: None,
                                    inlier_indices: Some(outcome.inliers),
                                    homography_auc_3: outcome.homography_auc_3,
                                    diagnostics: Some(outcome.diagnostics),
                                    success: precision >= 0.9
                                        && recall >= 0.9
                                        && outcome.normalized_model_error <= 1.0,
                                    failure_reason: None,
                                }
                            }
                            Err(reason) => Trial {
                                suite: input.dataset.clone(),
                                suite_version: input.schema_version,
                                estimator: "homography".into(),
                                scoring_mode: mode.clone(),
                                sampler: sampler.clone(),
                                profile: (*profile).clone(),
                                scene: format!("{}/{}", pair.dataset, pair.pair),
                                seed,
                                runtime_ms,
                                iterations: 0,
                                inlier_precision: 0.,
                                inlier_recall: 0.,
                                normalized_model_error: f64::MAX,
                                inlier_classification_error: 1.,
                                epipolar_matrix: None,
                                inlier_indices: None,
                                homography_auc_3: None,
                                diagnostics: None,
                                success: false,
                                failure_reason: Some(reason),
                            },
                        };
                        writeln!(out, "{}", serde_json::to_string(&trial)?)?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn run_rigid_suite(
    input: &RigidInput,
    suite: &Suite,
    args: &Args,
    out: &mut fs::File,
) -> Result<(), Box<dyn std::error::Error>> {
    if input.schema_version != 1 || input.threshold <= 0.0 {
        return Err("unsupported rigid input schema or threshold".into());
    }
    let profiles: Vec<&String> = if args.smoke {
        suite
            .profiles
            .iter()
            .filter(|profile| profile.as_str() == "balanced")
            .collect()
    } else {
        suite.profiles.iter().collect()
    };
    for pair in &input.pairs {
        for mode in &suite.scoring_modes {
            for sampler in &suite.samplers {
                for profile in &profiles {
                    for index in 0..args.seeds {
                        let seed = 0x5EED_CAFE_D00D_BAAD_u64 ^ (index as u64);
                        let start = Instant::now();
                        let result = settings(profile, mode, sampler, seed).and_then(|settings| {
                            run_rigid_fixture(pair, input.threshold, settings)
                        });
                        let runtime_ms = start.elapsed().as_secs_f64() * 1000.;
                        let trial = match result {
                            Ok(outcome) => {
                                let (precision, recall, classification_error) =
                                    classification_score(&outcome.inliers, &outcome.truth);
                                Trial {
                                    suite: input.dataset.clone(),
                                    suite_version: input.schema_version,
                                    estimator: "rigid_transform".into(),
                                    scoring_mode: mode.clone(),
                                    sampler: sampler.clone(),
                                    profile: (*profile).clone(),
                                    scene: pair.scene.clone(),
                                    seed,
                                    runtime_ms,
                                    iterations: outcome.iterations,
                                    inlier_precision: precision,
                                    inlier_recall: recall,
                                    normalized_model_error: outcome.normalized_model_error,
                                    inlier_classification_error: classification_error,
                                    epipolar_matrix: None,
                                    inlier_indices: Some(outcome.inliers),
                                    homography_auc_3: None,
                                    diagnostics: Some(outcome.diagnostics),
                                    success: precision >= 0.9
                                        && recall >= 0.9
                                        && outcome.normalized_model_error <= 1.0,
                                    failure_reason: None,
                                }
                            }
                            Err(reason) => Trial {
                                suite: input.dataset.clone(),
                                suite_version: input.schema_version,
                                estimator: "rigid_transform".into(),
                                scoring_mode: mode.clone(),
                                sampler: sampler.clone(),
                                profile: (*profile).clone(),
                                scene: pair.scene.clone(),
                                seed,
                                runtime_ms,
                                iterations: 0,
                                inlier_precision: 0.,
                                inlier_recall: 0.,
                                normalized_model_error: f64::MAX,
                                inlier_classification_error: 1.,
                                epipolar_matrix: None,
                                inlier_indices: None,
                                homography_auc_3: None,
                                diagnostics: None,
                                success: false,
                                failure_reason: Some(reason),
                            },
                        };
                        writeln!(out, "{}", serde_json::to_string(&trial)?)?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let suite: Suite = toml::from_str(&fs::read_to_string(&args.suite)?)?;
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut out = fs::File::create(&args.output)?;
    if let Some(path) = &args.phototourism_input {
        let input: PhototourismInput = serde_json::from_str(&fs::read_to_string(path)?)?;
        run_phototourism_suite(&input, &suite, &args, &mut out)?;
        return Ok(());
    }
    if let Some(path) = &args.homography_input {
        let input: HomographyInput = serde_json::from_str(&fs::read_to_string(path)?)?;
        run_homography_suite(&input, &suite, &args, &mut out)?;
        return Ok(());
    }
    if let Some(path) = &args.rigid_input {
        let input: RigidInput = serde_json::from_str(&fs::read_to_string(path)?)?;
        run_rigid_suite(&input, &suite, &args, &mut out)?;
        return Ok(());
    }
    let profiles: Vec<&String> = if args.smoke {
        suite
            .profiles
            .iter()
            .filter(|p| p.as_str() == "balanced")
            .collect()
    } else {
        suite.profiles.iter().collect()
    };
    for estimator in &suite.estimators {
        for mode in &suite.scoring_modes {
            for sampler in &suite.samplers {
                for scene_name in suite
                    .scenes
                    .iter()
                    .filter(|scene| !args.smoke || scene.as_str() == "outliers")
                {
                    let scene = Scene::parse(scene_name).ok_or("unknown scene")?;
                    for profile in &profiles {
                        for index in 0..args.seeds {
                            let seed = 0x5EED_CAFE_D00D_BAAD_u64 ^ (index as u64);
                            let start = Instant::now();
                            let result = settings(profile, mode, sampler, seed)
                                .and_then(|s| run(estimator, scene, s, seed));
                            let runtime_ms = start.elapsed().as_secs_f64() * 1000.;
                            let trial = match result {
                                Ok(outcome) => {
                                    let (p, r, classification_error) =
                                        classification_score(&outcome.inliers, &outcome.truth);
                                    let success = p >= 0.9
                                        && r >= 0.9
                                        && outcome.normalized_model_error <= 1.0;
                                    Trial {
                                        suite: suite.name.clone(),
                                        suite_version: suite.version,
                                        estimator: estimator.clone(),
                                        scoring_mode: mode.clone(),
                                        sampler: sampler.clone(),
                                        profile: (*profile).clone(),
                                        scene: scene_name.clone(),
                                        seed,
                                        runtime_ms,
                                        iterations: outcome.iterations,
                                        inlier_precision: p,
                                        inlier_recall: r,
                                        normalized_model_error: outcome.normalized_model_error,
                                        inlier_classification_error: classification_error,
                                        epipolar_matrix: None,
                                        inlier_indices: None,
                                        homography_auc_3: outcome.homography_auc_3,
                                        diagnostics: Some(outcome.diagnostics),
                                        success,
                                        failure_reason: None,
                                    }
                                }
                                Err(reason) => Trial {
                                    suite: suite.name.clone(),
                                    suite_version: suite.version,
                                    estimator: estimator.clone(),
                                    scoring_mode: mode.clone(),
                                    sampler: sampler.clone(),
                                    profile: (*profile).clone(),
                                    scene: scene_name.clone(),
                                    seed,
                                    runtime_ms,
                                    iterations: 0,
                                    inlier_precision: 0.,
                                    inlier_recall: 0.,
                                    normalized_model_error: f64::MAX,
                                    inlier_classification_error: 1.,
                                    epipolar_matrix: None,
                                    inlier_indices: None,
                                    homography_auc_3: None,
                                    diagnostics: None,
                                    success: false,
                                    failure_reason: Some(reason),
                                },
                            };
                            writeln!(out, "{}", serde_json::to_string(&trial)?)?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{classification_score, normalized_median_residual, settings};
    use inlier::settings::SamplerType;

    #[test]
    fn classification_score_requires_precision_and_recall() {
        let truth = [true, true, true, false];
        let (precision, recall, error) = classification_score(&[0, 1, 2], &truth);
        assert_eq!(precision, 1.0);
        assert_eq!(recall, 1.0);
        assert_eq!(error, 0.0);
        assert_eq!(error, 0.0);

        let (_, recall, _) = classification_score(&[0], &truth);
        assert!(recall < 0.9);
    }

    #[test]
    fn model_error_is_median_inlier_residual_normalized_by_threshold() {
        let truth = [true, false, true, true];
        let error = normalized_median_residual(&truth, 0.5, |index| [0.1, 100.0, 0.2, 0.4][index]);
        assert!((error - 0.4).abs() < 1e-12);
    }

    #[test]
    fn benchmark_settings_select_the_requested_sampler() {
        assert_eq!(
            settings("fast", "ransac", "uniform", 1).unwrap().sampler,
            SamplerType::Uniform
        );
        assert_eq!(
            settings("fast", "ransac", "prosac", 1).unwrap().sampler,
            SamplerType::Prosac
        );
        assert!(settings("fast", "ransac", "unknown", 1).is_err());
    }
}

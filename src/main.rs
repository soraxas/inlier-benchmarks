use clap::Parser;
use inlier::{
    MetasacSettings, estimate_absolute_pose, estimate_essential_matrix,
    estimate_fundamental_matrix, estimate_homography, estimate_line, estimate_plane,
    estimate_rigid_transform, settings::ScoringType, types::DataMatrix,
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
}

#[derive(Deserialize)]
struct Suite {
    version: u32,
    name: String,
    estimators: Vec<String>,
    scoring_modes: Vec<String>,
    scenes: Vec<String>,
    profiles: Vec<String>,
}

#[derive(Serialize)]
struct Trial {
    suite: String,
    suite_version: u32,
    estimator: String,
    scoring_mode: String,
    profile: String,
    scene: String,
    seed: u64,
    runtime_ms: f64,
    iterations: usize,
    inlier_precision: f64,
    inlier_recall: f64,
    normalized_error: f64,
    success: bool,
    failure_reason: Option<String>,
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

fn settings(profile: &str, scoring: &str, seed: u64) -> Result<MetasacSettings, String> {
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
    Ok(MetasacSettings {
        min_iterations: iterations,
        max_iterations: iterations,
        confidence,
        rng_seed: Some(seed),
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

fn score(inliers: &[usize], truth: &[bool]) -> (f64, f64, f64, bool) {
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
    (p, r, 1. - f1, p >= 0.9 && r >= 0.9)
}

fn run(
    estimator: &str,
    scene: Scene,
    settings: MetasacSettings,
    seed: u64,
) -> Result<(Vec<usize>, usize, Vec<bool>), String> {
    match estimator {
        "homography" => {
            let (a, b, t) = homography(scene, seed);
            let r = estimate_homography(&a, &b, 0.5, Some(settings))?;
            Ok((r.inliers, r.iterations, t))
        }
        "fundamental" => {
            let (a, b, t) = image(scene, seed);
            let r = estimate_fundamental_matrix(&a, &b, 0.01, Some(settings))?;
            Ok((r.inliers, r.iterations, t))
        }
        "essential" => {
            let (a, b, t) = image(scene, seed);
            let r = estimate_essential_matrix(&a, &b, 0.01, Some(settings))?;
            Ok((r.inliers, r.iterations, t))
        }
        "absolute_pose" => {
            let (a, b, t) = pose(scene, seed);
            let r = estimate_absolute_pose(&a, &b, 0.01, Some(settings))?;
            Ok((r.inliers, r.iterations, t))
        }
        "line" => {
            let (a, t) = line(scene, seed);
            let r = estimate_line(&a, 0.05, Some(settings))?;
            Ok((r.inliers, r.iterations, t))
        }
        "plane" => {
            let (a, t) = plane(scene, seed);
            let r = estimate_plane(&a, 0.05, Some(settings))?;
            Ok((r.inliers, r.iterations, t))
        }
        "rigid_transform" => {
            let (a, b, t) = rigid(scene, seed);
            let r = estimate_rigid_transform(&a, &b, 0.05, Some(settings))?;
            Ok((r.inliers, r.iterations, t))
        }
        _ => Err(format!("unknown estimator {estimator}")),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let suite: Suite = toml::from_str(&fs::read_to_string(&args.suite)?)?;
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut out = fs::File::create(args.output)?;
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
                        let result = settings(profile, mode, seed)
                            .and_then(|s| run(estimator, scene, s, seed));
                        let runtime_ms = start.elapsed().as_secs_f64() * 1000.;
                        let trial = match result {
                            Ok((inliers, iterations, truth)) => {
                                let (p, r, e, success) = score(&inliers, &truth);
                                Trial {
                                    suite: suite.name.clone(),
                                    suite_version: suite.version,
                                    estimator: estimator.clone(),
                                    scoring_mode: mode.clone(),
                                    profile: (*profile).clone(),
                                    scene: scene_name.clone(),
                                    seed,
                                    runtime_ms,
                                    iterations,
                                    inlier_precision: p,
                                    inlier_recall: r,
                                    normalized_error: e,
                                    success,
                                    failure_reason: None,
                                }
                            }
                            Err(reason) => Trial {
                                suite: suite.name.clone(),
                                suite_version: suite.version,
                                estimator: estimator.clone(),
                                scoring_mode: mode.clone(),
                                profile: (*profile).clone(),
                                scene: scene_name.clone(),
                                seed,
                                runtime_ms,
                                iterations: 0,
                                inlier_precision: 0.,
                                inlier_recall: 0.,
                                normalized_error: 1.,
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

#[cfg(test)]
mod tests {
    use super::score;

    #[test]
    fn classification_score_requires_precision_and_recall() {
        let truth = [true, true, true, false];
        let (precision, recall, error, success) = score(&[0, 1, 2], &truth);
        assert_eq!(precision, 1.0);
        assert_eq!(recall, 1.0);
        assert_eq!(error, 0.0);
        assert!(success);

        let (_, recall, _, success) = score(&[0], &truth);
        assert!(recall < 0.9);
        assert!(!success);
    }
}

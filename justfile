suite := "suites/public_api.toml"

prepare-phototourism output="results/phototourism-input.json":
  uv run --no-project --with-editable ../inlier-data --with h5py python/prepare_phototourism.py --output {{output}}

phototourism-smoke input="results/phototourism-input.json":
  cargo run --release -- --suite {{suite}} --smoke --seeds 1 --phototourism-input {{input}} --output results/phototourism.jsonl
  uv run --no-project --with numpy --with opencv-python-headless python/evaluate_phototourism.py --input {{input}} --results results/phototourism.jsonl

prepare-homography output="results/homography-input.json":
  uv run --no-project --with-editable ../inlier-data --with h5py python/prepare_homography.py --output {{output}}

homography-smoke input="results/homography-input.json":
  cargo run --release -- --suite {{suite}} --smoke --seeds 1 --homography-input {{input}} --output results/homography.jsonl

prepare-epos-pnp output="results/epos-pnp-input.json":
  uv run --no-project --with-editable ../inlier-data --with numpy python/prepare_epos_pnp.py --output {{output}}

epos-pnp-smoke input="results/epos-pnp-input.json":
  cargo run --release -- --suite {{suite}} --smoke --seeds 1 --absolute-pose-input {{input}} --output results/epos-pnp.jsonl
  uv run --no-project --with numpy python/evaluate_absolute_pose.py --input {{input}} --results results/epos-pnp.jsonl

prepare-rigid output="results/rigid-input.json":
  uv run --no-project --with-editable ../inlier-data --with numpy python/prepare_rigid.py --output {{output}}

rigid-smoke input="results/rigid-input.json":
  cargo run --release -- --suite {{suite}} --smoke --seeds 1 --rigid-input {{input}} --output results/rigid.jsonl

smoke:
  cargo run --release -- --suite {{suite}} --smoke --seeds 5 --output results/raw.jsonl
  uv run --no-project --with numpy python/aggregate.py results/raw.jsonl results/summary.json
  uv run --no-project python/regress.py results/summary.json baseline/summary.json
  uv run --no-project python/report.py results/summary.json site

full:
  cargo run --release -- --suite {{suite}} --seeds 30 --output results/raw.jsonl
  uv run --no-project --with numpy python/aggregate.py results/raw.jsonl results/summary.json
  uv run --no-project python/report.py results/summary.json site

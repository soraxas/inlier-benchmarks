suite := "suites/public_api.toml"

prepare-phototourism output="results/phototourism-input.json":
  uv run --no-project --with ../inlier-data --with h5py python/prepare_phototourism.py --output {{output}}

phototourism-smoke input="results/phototourism-input.json":
  cargo run --release -- --suite {{suite}} --smoke --seeds 1 --phototourism-input {{input}} --output results/phototourism.jsonl
  uv run --no-project --with numpy --with opencv-python-headless python/evaluate_phototourism.py --input {{input}} --results results/phototourism.jsonl

smoke:
  cargo run --release -- --suite {{suite}} --smoke --seeds 5 --output results/raw.jsonl
  uv run --no-project --with numpy --with matplotlib python/aggregate.py results/raw.jsonl results/summary.json
  uv run --no-project python/regress.py results/summary.json baseline/summary.json
  uv run --no-project --with numpy --with matplotlib python/report.py results/summary.json site

full:
  cargo run --release -- --suite {{suite}} --seeds 30 --output results/raw.jsonl
  uv run --no-project --with numpy --with matplotlib python/aggregate.py results/raw.jsonl results/summary.json
  uv run --no-project --with numpy --with matplotlib python/report.py results/summary.json site

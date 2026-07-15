suite := "suites/public_api.toml"

smoke:
  cargo run --release -- --suite {{suite}} --smoke --seeds 5 --output results/raw.jsonl
  uv run --no-project --with numpy --with matplotlib python/aggregate.py results/raw.jsonl results/summary.json
  uv run --no-project python/regress.py results/summary.json baseline/summary.json
  uv run --no-project --with numpy --with matplotlib python/report.py results/summary.json site

full:
  cargo run --release -- --suite {{suite}} --seeds 30 --output results/raw.jsonl
  uv run --no-project --with numpy --with matplotlib python/aggregate.py results/raw.jsonl results/summary.json
  uv run --no-project --with numpy --with matplotlib python/report.py results/summary.json site


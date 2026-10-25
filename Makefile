PY ?= python
GPU ?=            # e.g. GPU="--backend xgb --device cuda"

.PHONY: install test smoke eda run report all

install:
	$(PY) -m pip install -q -r requirements.txt

test:
	$(PY) -m pytest -q tests

smoke:            # end-to-end on synthetic data, ~30 s, no download
	$(PY) scripts/run_experiments.py --synthetic --out results_smoke/synthetic

eda:
	$(PY) scripts/eda.py --n-users 5000 --out results/eda

run:              # AutoFE search on 1000 users, transfer to 5000 users
	$(PY) scripts/run_experiments.py --n-users 1000 $(GPU) --out results/u1000
	$(PY) scripts/run_experiments.py --n-users 5000 --autofe-from results/u1000 $(GPU) --out results/u5000
	$(PY) scripts/bootstrap.py results/u1000
	$(PY) scripts/bootstrap.py results/u5000

report:
	$(PY) scripts/make_report.py results

all: install test eda run report

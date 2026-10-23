PY ?= python
USERS ?=

.PHONY: install test smoke run scale report all

install:
	$(PY) -m pip install -q -r requirements.txt

test:
	$(PY) -m pytest -q tests

smoke:            # end-to-end on synthetic data, ~30 s, no download
	$(PY) scripts/run_experiments.py --synthetic --out results/synthetic

run:              # main experiment: all 10k users of Yambda-50M
	$(PY) scripts/run_experiments.py --out results/u10000

scale:            # scalability: 1k / 3k users (full histories)
	$(PY) scripts/run_experiments.py --n-users 1000 --out results/u1000
	$(PY) scripts/run_experiments.py --n-users 3000 --out results/u3000

report:
	$(PY) scripts/make_report.py results

all: install test run scale report

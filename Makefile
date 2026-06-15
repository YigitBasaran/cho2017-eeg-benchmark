# Cho2017 Motor Imagery Benchmark - convenience targets.
# On Windows without `make`, run the underlying `python scripts/...` commands
# shown in the README directly.

PY ?= python

.PHONY: install download validate-data prepare-data splits test clean-cache status all-data

install:
	$(PY) -m pip install -e .

download:
	$(PY) scripts/download_dataset.py

validate-data:
	$(PY) scripts/validate_dataset.py

prepare-data:
	$(PY) scripts/prepare_dataset.py

splits:
	$(PY) scripts/create_splits.py

all-data: download validate-data prepare-data splits

test:
	$(PY) -m pytest -q

clean-cache:
	$(PY) -c "import glob, os; [os.remove(p) for p in glob.glob('data/processed/common/*.npz')]; print('processed cache cleared')"

status:
	$(PY) scripts/print_project_status.py

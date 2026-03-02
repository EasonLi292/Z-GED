PYTHON ?= .venv/bin/python

.PHONY: setup setup-dev doctor train eval-pz generate-pz test test-unit

setup:
	./scripts/setup_venv.sh runtime

setup-dev:
	./scripts/setup_venv.sh dev

doctor:
	$(PYTHON) -c "import torch, torch_geometric, numpy, scipy; print('ok')"

train:
	$(PYTHON) scripts/training/train.py

eval-pz:
	$(PYTHON) scripts/eval/eval_pz.py

generate-pz:
	$(PYTHON) scripts/generation/generate_from_specs.py --pole-real -6283 --pole-imag 0 --num-samples 3

test:
	$(PYTHON) tests/run_tests.py --suite all

test-unit:
	$(PYTHON) tests/run_tests.py --suite unit

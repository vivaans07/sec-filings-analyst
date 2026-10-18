PY ?= python

test:
	$(PY) -m pytest -q tests

fetch:
	$(PY) -m secqa.cli fetch

build:
	$(PY) -m secqa.cli build

eval:
	$(PY) -m secqa.cli eval

eval-dense:
	$(PY) -m secqa.cli eval --dense

all: fetch build eval

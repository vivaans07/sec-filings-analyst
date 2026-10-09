PY ?= python

test:
	$(PY) -m pytest -q tests

fetch:
	$(PY) -m secqa.cli fetch

build:
	$(PY) -m secqa.cli build

eval:
	$(PY) -m secqa.cli eval

all: fetch build eval

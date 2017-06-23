PYTHON ?= python
CYTHON ?= cython

all: inplace

clean:
	$(PYTHON) setup.py clean
	rm -rf dist

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

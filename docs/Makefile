# Minimal Sphinx Makefile

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
PYTHON        ?= python
SOURCEDIR      = source
BUILDDIR       = build

.PHONY: help install-docs html clean

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

install-docs:
	@$(PYTHON) -m pip install --disable-pip-version-check -r requirements.txt

html: install-docs
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

help:
	@cat Makefile

.EXPORT_ALL_VARIABLES:

# create an .env file to override the default settings
-include .env
export $(shell sed 's/=.*//' .env)


.PHONY: build docs

# ----------------
# default settings
# ----------------
# user
LOCAL_USER:=$(shell whoami)
LOCAL_USER_ID:=$(shell id -u)
# project
PROJECT_NAME:=chess-rl
EXAMPLE_DIR:=./examples
# python
PYTHON?=python
PYTHON_EXEC?=python -m
PYTHONVERSION?=3.10.10
PYTEST?=pytest
SYSTEM=$(shell python -c "import sys; print(sys.platform)")
# poetry
PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
POETRY=poetry


# -----------
# install project's dependencies
# -----------
# dev dependencies
install-init:
	$(PYTHON_EXEC) pip install --upgrade pip
	$(PYTHON_EXEC) pip install --upgrade poetry

install: install-init
	$(PYTHON_EXEC) poetry install --no-cache


# -----------
# testing
# -----------
pytest:
	$(PYTHON_EXEC) pytest -x --testmon --nbmake --overwrite "$(EXAMPLE_DIR)"
	$(PYTHON_EXEC) mypy test
	$(PYTHON_EXEC) pytest -x --testmon --pylint --cov-fail-under 98

test: pytest


# -----------
# git
# -----------
# Locally delete branches that have been merged
git-clean:
	bash scripts/git-clean.sh

# squash all commits before rebasing, see https://stackoverflow.com/questions/25356810/git-how-to-squash-all-commits-on-branch
git-squash:
	git reset $(git merge-base main $(git branch --show-current))
	git add -A
	git commit -m "squashed commit"

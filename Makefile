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
PROJECT_NAME:=shark-chess
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
# docker
DOCKER?=docker
DOCKERFILE?=Dockerfile
MEMORY=8g
SHM=4g
CPUS=1
DOCKER_COMMON_FLAGS=--cpus=$(CPUS) --memory=$(MEMORY) --shm-size=$(SHM) --network=host --volume $(PWD):/workdir -e LOCAL_USER_ID -e LOCAL_USER
REGISTRY=registry.gitlab.com/svnv-svsv-jm/chess-rl
IMAGE=$(PROJECT_NAME)
IMAGE_PYTHON=/venv/bin/python


# -----------
# install project's dependencies
# -----------
# dev dependencies
install-init:
	$(PYTHON_EXEC) pip install --upgrade pip
	$(PYTHON_EXEC) pip install --upgrade poetry
	$(PYTHON_EXEC) poetry self update

install: install-init
	$(PYTHON_EXEC) poetry install --no-cache


# -----------
# testing
# -----------
mypy:
	$(PYTHON_EXEC) mypy tests

pytest:
	$(PYTHON_EXEC) pytest -x --testmon --pylint --cov-fail-under 95

pytest-nbmake:
	$(PYTHON_EXEC) pytest -x --testmon --nbmake --overwrite "$(EXAMPLE_DIR)"

test: mypy pytest pytest-nbmake

tests: test

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


# -----------
# docker
# -----------
build: TAG=latest
build:
	$(DOCKER) build -t $(REGISTRY)/$(IMAGE):$(TAG) --network=host --build-arg PROJECT_NAME=$(PROJECT_NAME) -f $(DOCKERFILE) .

buildx: TAG=latest
buildx: PLATFORM=linux/amd64
buildx: LOAD_PUSH=--load
buildx:
	$(DOCKER) buildx build --platform $(PLATFORM) $(LOAD_PUSH) --network=host -t $(REGISTRY)/$(IMAGE):$(TAG) --build-arg PROJECT_NAME=$(PROJECT_NAME) -f $(DOCKERFILE) .

buildx-multi: PLATFORM=linux/amd64,linux/arm64
buildx-multi: LOAD_PUSH=--push
buildx-multi: build

push:
	echo $(CI_JOB_TOKEN) | $(DOCKER) login -u $(LOCAL_USER) $(REGISTRY) --password-stdin
	$(DOCKER) image push $(REGISTRY)/$(PROJECT_NAME):latest

bash: CONTAINER_NAME=bash
bash:
	$(DOCKER) run --rm -it $(DOCKER_COMMON_FLAGS) \
		--name $(PROJECT_NAME)-$(CONTAINER_NAME) \
		-t $(REGISTRY)/$(PROJECT_NAME) \
		bash

up: FLAGS=
up: docker-compose.yml
	@echo "DOCKER_BUILDKIT=${DOCKER_BUILDKIT}"
	@echo "COMPOSE_DOCKER_CLI_BUILD=${COMPOSE_DOCKER_CLI_BUILD}"
	docker-compose -p $(PROJECT_NAME) up -d $(FLAGS)

up-force: FLAGS=--build --force-recreate
up-force: up

down: docker-compose.yml
	$(DOCKER)-compose -p $(PROJECT_NAME) down --volumes


# -----------------------------------------
# experiments
# -----------------------------------------
run:
	supervisord -c supervisord.conf

# [DO NOT CALL THIS COMMAND]
exp-base: CONFIG=
exp-base: RANDOM=$(shell bash -c 'echo $$RANDOM')
exp-base: CONTAINER_NAME=exp-$(CONFIG)-$(RANDOM)
exp-base: SCRIPT=experiments/main.py
exp-base: PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
exp-base: DOCKER_FLAGS=
exp-base: OVERRIDE=
exp-base: NOW=$(shell date '+%Y-%m-%d_%H:%M:%S')
exp-base: CMD=$(IMAGE_PYTHON) -u /workdir/$(SCRIPT) --config-name $(CONFIG) $(OVERRIDE)
exp-base:
	$(DOCKER) run --rm -it $(DOCKER_FLAGS) $(DOCKER_COMMON_FLAGS) \
		$(GPU_FLAGS) \
		--name $(PROJECT_NAME)-$(CONTAINER_NAME) \
		-t $(REGISTRY)/$(PROJECT_NAME) \
		$(CMD)

exp: CONFIG=main.yaml
exp: exp-base

# exp-gpu: GPU_FLAGS=--runtime=nvidia --gpus all
exp-gpu: GPU_FLAGS=--gpus all
exp-gpu: exp-base
PYTHON = python3
PIP = pip
DOCKER_BASE_IMAGE = docker.io/ocrd/core-cuda-torch:v3.3.0
DOCKER_TAG = ocrd/froc

install:
	pip install .

install-dev:
	pip install -e .

deps:
	pip install -r requirements.txt

docker:
	docker build --progress=plain \
	--build-arg DOCKER_BASE_IMAGE=$(DOCKER_BASE_IMAGE) \
	--build-arg VCS_REF=$$(git rev-parse --short HEAD) \
	--build-arg BUILD_DATE=$$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
	-t $(DOCKER_TAG) .

# assets
.PHONY: always-update

repo/assets: always-update
	git submodule sync "$@"
	git submodule update --init "$@"

# tests

deps-test:
	$(PIP) install -r requirements-test.txt

assets-clean:
	rm -rf tests/assets

assets: assets-clean tests/assets

tests/assets: repo/assets
	mkdir -p $@
	cp -r $</data/* $@

test: tests/assets deps-test
	$(PYTHON) -m pytest tests --durations=0 --continue-on-collection-errors $(PYTEST_ARGS)

coverage: deps-test
	coverage erase
	$(MAKE) test PYTHON="coverage run"
	coverage combine
	coverage report -m


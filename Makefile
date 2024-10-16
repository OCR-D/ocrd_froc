DOCKER_BASE_IMAGE = docker.io/ocrd/core-cuda-torch:v2.70.0
DOCKER_TAG = ocrd/froc

install:
	pip install .

install-dev:
	pip install -e .

docker:
	docker build \
	--build-arg DOCKER_BASE_IMAGE=$(DOCKER_BASE_IMAGE) \
	--build-arg VCS_REF=$$(git rev-parse --short HEAD) \
	--build-arg BUILD_DATE=$$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
	-t $(DOCKER_TAG) .

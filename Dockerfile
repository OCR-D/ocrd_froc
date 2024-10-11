ARG DOCKER_BASE_IMAGE
FROM $DOCKER_BASE_IMAGE
ARG VCS_REF
ARG BUILD_DATE
LABEL \
    maintainer="https://ocr-d.de/kontakt" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vcs-url="https://github.com/ocr-d/ocrd_froc" \
    org.label-schema.build-date=$BUILD_DATE

WORKDIR /build/ocrd_froc
COPY pyproject.toml .
COPY ocrd_froc/ocrd-tool.json .
COPY ocrd_froc ./ocrd_froc
COPY requirements.txt .
COPY README.md .
COPY Makefile .
RUN make install \
	&& rm -rf /build/ocrd_froc

WORKDIR /data
VOLUME ["/data"]

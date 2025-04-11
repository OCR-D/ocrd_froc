ARG DOCKER_BASE_IMAGE
FROM $DOCKER_BASE_IMAGE
ARG VCS_REF
ARG BUILD_DATE
LABEL \
    maintainer="https://ocr-d.de/en/contact" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vcs-url="https://github.com/OCR-D/ocrd_froc" \
    org.label-schema.build-date=$BUILD_DATE \
    org.opencontainers.image.vendor="DFG-Funded Initiative for Optical Character Recognition Development" \
    org.opencontainers.image.title="ocrd_froc" \
    org.opencontainers.image.description="Perform font classification and text recognition (in one step) on historic documents." \
    org.opencontainers.image.source="https://github.com/OCR-D/ocrd_froc" \
    org.opencontainers.image.documentation="https://github.com/OCR-D/ocrd_froc/blob/${VCS_REF}/README.md" \
    org.opencontainers.image.revision=$VCS_REF \
    org.opencontainers.image.created=$BUILD_DATE \
    org.opencontainers.image.base.name=ocrd/core-cuda-torch

ENV PYTHONIOENCODING utf8
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV XDG_DATA_HOME /usr/local/share
ENV XDG_CONFIG_HOME /usr/local/share/ocrd-resources

WORKDIR /build/ocrd_froc
COPY . .
RUN make install \
	&& rm -rf /build/ocrd_froc

WORKDIR /data
VOLUME /data

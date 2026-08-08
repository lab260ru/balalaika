# syntax=docker/dockerfile:1.7

ARG CUDA_BASE=docker.io/nvidia/cuda:12.8.1-base-ubuntu24.04@sha256:133c78a0575303be34164d0b90137a042172bdf60696af01a3c424ab402d86e2
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7@sha256:240fb85ab0f263ef12f492d8476aa3a2e4e1e333f7d67fbdd923d00a506a516a

FROM ${UV_IMAGE} AS uv

FROM ${CUDA_BASE} AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy

# The OCI base is digest-pinned; Ubuntu patch revisions intentionally follow
# the current security repository during the pilot build.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv
COPY requirements_dev.cuda128.txt /tmp/requirements.txt

# ruaccent declares the CPU onnxruntime distribution. Reinstall the pinned GPU
# wheel last so it owns the shared `onnxruntime` module files.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/balalaika/.venv --python /usr/bin/python3.12 \
    && uv pip install \
        --python /opt/balalaika/.venv/bin/python \
        --requirements /tmp/requirements.txt \
    && uv pip install \
        --python /opt/balalaika/.venv/bin/python \
        --reinstall \
        --no-deps \
        onnxruntime-gpu==1.26.0

FROM ${CUDA_BASE} AS runtime

ARG BALALAIKA_UID=10001
ARG BALALAIKA_GID=10001

ENV DEBIAN_FRONTEND=noninteractive

# See the builder-stage rationale for unpinned Ubuntu security revisions.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        coreutils \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsndfile1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        python3.12 \
        tini \
        util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${BALALAIKA_GID}" balalaika \
    && useradd \
        --uid "${BALALAIKA_UID}" \
        --gid "${BALALAIKA_GID}" \
        --create-home \
        --shell /bin/bash \
        balalaika

COPY --from=builder --chown=balalaika:balalaika \
    /opt/balalaika/.venv /opt/balalaika/.venv
COPY --chown=balalaika:balalaika . /opt/balalaika/app

RUN mkdir -p \
        /cache/balalaika \
        /cache/home \
        /cache/huggingface \
        /cache/matplotlib \
        /cache/numba \
        /cache/ruaccent \
        /cache/torch \
        /cache/trt \
        /cache/xdg \
        /data \
        /logs \
        /models \
        /output \
        /work/runtime \
    && chown -R balalaika:balalaika \
        /cache /data /logs /models /output /work \
    && ln -s /models /opt/balalaika/app/models \
    && ln -s /cache/balalaika /opt/balalaika/app/cache \
    && chmod 0755 \
        /opt/balalaika/app/docker/entrypoint.sh \
        /opt/balalaika/app/docker/run_gpu0.sh \
        /opt/balalaika/app/docker/build.sh

# DL3064 treats TOKENIZERS_PARALLELISM as a secret-like name. It is a boolean
# library setting; no credentials are stored in the image environment.
# hadolint ignore=DL3064
ENV VIRTUAL_ENV=/opt/balalaika/.venv \
    PATH=/opt/balalaika/.venv/bin:${PATH} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    HOME=/cache/home \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES=0 \
    BALALAIKA_DISABLE_CUDF=1 \
    BALALAIKA_NODE_PROFILE=/cache/balalaika/node_profile.json \
    BALALAIKA_RUACCENT_WORKDIR=/cache/ruaccent \
    HF_HOME=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub \
    XDG_CACHE_HOME=/cache/xdg \
    TORCH_HOME=/cache/torch \
    NUMBA_CACHE_DIR=/cache/numba \
    MPLCONFIGDIR=/cache/matplotlib \
    LD_LIBRARY_PATH=/opt/balalaika/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/balalaika/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/balalaika/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/opt/balalaika/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/opt/balalaika/.venv/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:/opt/balalaika/.venv/lib/python3.12/site-packages/nvidia/cufft/lib:/opt/balalaika/.venv/lib/python3.12/site-packages/nvidia/cufile/lib:/opt/balalaika/.venv/lib/python3.12/site-packages/nvidia/curand/lib:/opt/balalaika/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:/opt/balalaika/.venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/opt/balalaika/.venv/lib/python3.12/site-packages/nvidia/nvtx/lib:/opt/balalaika/.venv/lib/python3.12/site-packages/nvidia/cusolver/lib:/opt/balalaika/.venv/lib/python3.12/site-packages/nvidia/cusparse/lib:/opt/balalaika/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:/opt/balalaika/.venv/lib/python3.12/site-packages/tensorrt_libs:${LD_LIBRARY_PATH}

WORKDIR /opt/balalaika/app
# The account is created above with the build-argument UID/GID.
# hadolint ignore=DL3066
USER balalaika
STOPSIGNAL SIGINT

ENTRYPOINT ["/usr/bin/tini", "-g", "--", "/opt/balalaika/app/docker/entrypoint.sh"]
CMD ["smoke"]

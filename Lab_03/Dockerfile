# python:3.10 because tfx 1.16.0 declares requires_python >=3.9,<3.11.
# bookworm (glibc 2.36) is enough because every native dependency at this pin
# ships manylinux_2_17 wheels -- see requirements.txt for why that matters.
FROM python:3.10-slim-bookworm

# libgomp1 is the OpenMP runtime that the TensorFlow and Arrow wheels link
# against at import time; it is not in the slim base. curl is only used by the
# compose healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

# Installing with uv rather than pip is not a preference here, it is a
# requirement. tfx depends on apache-beam[gcp], and pip's backtracking resolver
# does not terminate on that graph in any practical time -- a plain
# `pip install tfx==1.16.0` was still resolving after 24 hours. uv resolves the
# same set in seconds.
RUN pip install --no-cache-dir uv==0.5.11

# The lock is a complete, already-resolved set, so this install performs no
# dependency resolution at all. Copied on its own so the dependency layer is
# cached independently of the notebook and module files.
COPY requirements.lock.txt /tmp/requirements.lock.txt

# pyfarmhash, pulled in by tensorflow-data-validation, publishes no wheel -- it
# is an sdist with a C++ extension and compiles on install. A lock pins which
# version is used, not whether a prebuilt wheel exists, so the toolchain has to
# be present. Installed and purged inside one RUN: layers are append-only, so
# splitting them would leave the compiler in the image permanently.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && uv pip install --system --no-cache -r /tmp/requirements.lock.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8888

# Transform imports breast_cancer_constants from inside the module file that
# gets passed as module_file, and Beam may run it from a different working
# directory than the notebook's.
ENV PYTHONPATH=/workspace

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--ServerApp.root_dir=/workspace", \
     "--IdentityProvider.token=tfx"]

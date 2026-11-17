# LIEP reproduction image: Julia 1.12.7 with the pinned environment (PhasorNetworks.jl 903cf06), FashionMNIST, and the
# figure toolchain (Python, Plotly, kaleido, Chromium, Liberation fonts). amd64 and arm64.
# Build from the repo root:  docker build -f docker/Dockerfile -t liep .
# GPU runs need the NVIDIA container toolkit on the host (docker run --gpus all); CUDA.jl brings its own runtime.
FROM debian:bookworm-slim

ARG JULIA_VERSION=1.12.7
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git make python3 python3-venv chromium fonts-liberation fonts-dejavu-core poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN arch=$(uname -m) && case $arch in x86_64) d=x64;; aarch64) d=aarch64;; *) echo "unsupported $arch"; exit 1;; esac \
    && curl -fsSL https://julialang-s3.julialang.org/bin/linux/$d/${JULIA_VERSION%.*}/julia-${JULIA_VERSION}-linux-$arch.tar.gz \
       | tar -xz -C /opt && ln -s /opt/julia-${JULIA_VERSION}/bin/julia /usr/local/bin/julia

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt
ENV PATH=/opt/venv/bin:$PATH BROWSER_PATH=/usr/bin/chromium

# The Julia environment: instantiate and precompile into a depot baked into the image.
ENV JULIA_DEPOT_PATH=/opt/julia-depot JULIA_PKG_PRECOMPILE_AUTO=0
# CUDA.jl picks its runtime from the driver it finds at precompile time, and an image is built without one, so the
# runtime is pinned here, in the depot's default environment (it then holds with a host checkout mounted over
# /opt/liep). 13.4 is what the paper's GB10 runs used; pass --build-arg CUDA_RUNTIME=12.x for drivers older than CUDA 13.
ARG CUDA_RUNTIME=13.4
RUN e=/opt/julia-depot/environments/v${JULIA_VERSION%.*} && mkdir -p $e \
    && printf '[extras]\nCUDA_Runtime_jll = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"\n' > $e/Project.toml \
    && printf '[CUDA_Runtime_jll]\nversion = "%s"\n' "$CUDA_RUNTIME" > $e/LocalPreferences.toml
WORKDIR /opt/liep
COPY Project.toml Manifest.toml ./
COPY deps/PhasorNetworks.jl deps/PhasorNetworks.jl
# cuDNN/LuxCUDA cannot precompile without a driver (they dlopen libcuda.so.1); they compile on the first GPU run, into
# the user's depot. Everything else, and the CUDA artifacts themselves, are baked in here.
RUN julia --project=. -e 'using Pkg; Pkg.instantiate(); try Pkg.precompile() catch; @warn "GPU packages compile on first use" end'
# FashionMNIST (the library downloads it into a scratch space on first use); kept at /opt/data for read-only images.
RUN julia --project=. -e 'using PhasorNetworks; PhasorNetworks.fashion_mnist_data(:train); PhasorNetworks.fashion_mnist_data(:test)' \
    && mkdir -p /opt/data && cp -r /opt/julia-depot/scratchspaces/*/fashion_mnist /opt/data/ && chmod -R a+rX /opt/julia-depot /opt/data

COPY . /opt/liep
COPY docker/entrypoint.sh /usr/local/bin/liep-entry
ENV JULIA_DEPOT_PATH=/root/.julia:/opt/julia-depot
ENTRYPOINT ["/usr/local/bin/liep-entry"]
CMD ["make", "figures"]

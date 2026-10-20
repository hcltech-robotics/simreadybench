# syntax=docker/dockerfile:1@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
# check=skip=FromPlatformFlagConstDisallowed

# Harbor authoring tools run on Linux amd64.
FROM --platform=linux/amd64 ubuntu:24.04@sha256:a2d43bffc46670d7108bb3c0468c67c5557c4c92e83e1e34cbff1322ad1b94dc AS nodebase
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN export QEMU_CPU=Nehalem && apt-get update && apt-get install -y --no-install-recommends ca-certificates curl xz-utils && rm -rf /var/lib/apt/lists/*
RUN export QEMU_CPU=Nehalem && mkdir -p /opt/simready /root/.nvm/versions/node/v22.23.3 && \
    curl --fail --show-error --location --retry 3 https://nodejs.org/dist/v22.23.3/node-v22.23.3-linux-x64.tar.xz -o /tmp/node.tar.xz && \
    echo "df450af89261115ef9f9e3830c3eeb2cc9213b63c720b1af623cb5dcbe2e02de  /tmp/node.tar.xz" | sha256sum --check && \
    tar -xJf /tmp/node.tar.xz --strip-components=1 -C /root/.nvm/versions/node/v22.23.3 && \
    curl --fail --show-error --location --retry 3 https://codeload.github.com/nvm-sh/nvm/tar.gz/refs/tags/v0.40.2 -o /tmp/nvm.tar.gz && \
    echo "da642dbff51d9afc729d67f8aa327decaf16e8b4efee792f0be2da817bd6760e  /tmp/nvm.tar.gz" | sha256sum --check && \
    tar -xzf /tmp/nvm.tar.gz --strip-components=1 -C /root/.nvm && \
    mkdir -p /root/.nvm/alias && echo 22.23.3 > /root/.nvm/alias/default && \
    rm /tmp/node.tar.xz /tmp/nvm.tar.gz
ENV NVM_DIR=/root/.nvm \
    PATH=/opt/actor-python/bin:/root/.nvm/versions/node/v22.23.3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

FROM --platform=$BUILDPLATFORM ubuntu:24.04@sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac AS npmlocks
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl xz-utils python3.12 python3.12-venv && rm -rf /var/lib/apt/lists/*
ARG BUILDARCH
RUN mkdir -p /opt/node && \
    case "$BUILDARCH" in \
      arm64) arch=arm64; hash=a44aeb94849a299b22df10b9e622ec2f605c2183501bc40590705131de7c740f;; \
      amd64) arch=x64; hash=df450af89261115ef9f9e3830c3eeb2cc9213b63c720b1af623cb5dcbe2e02de;; \
      *) echo "Unsupported builder architecture" >&2; exit 2;; \
    esac && \
    curl --fail --show-error --location --retry 3 "https://nodejs.org/dist/v22.23.3/node-v22.23.3-linux-$arch.tar.xz" -o /tmp/node.tar.xz && \
    echo "$hash  /tmp/node.tar.xz" | sha256sum --check && \
    tar -xJf /tmp/node.tar.xz --strip-components=1 -C /opt/node && rm /tmp/node.tar.xz
ENV PATH=/opt/node/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
COPY pi/package.json /opt/simready/pi/package.json
COPY bridge/package.json /tmp/harbor-pi-mcp-adapter/package.json

FROM npmlocks AS dependencies
COPY locks/pi/package-lock.json /opt/simready/pi/package-lock.json
COPY locks/bridge/package-lock.json /tmp/harbor-pi-mcp-adapter/package-lock.json
RUN npm ci --prefix /opt/simready/pi --cpu=x64 --os=linux --ignore-scripts --no-audit --no-fund && \
    npm ci --prefix /tmp/harbor-pi-mcp-adapter --cpu=x64 --os=linux --ignore-scripts --no-audit --no-fund
COPY opencode/package.json opencode/package-lock.json /opt/simready/opencode/
RUN npm ci --prefix /opt/simready/opencode --force --omit=optional --ignore-scripts --no-audit --no-fund
COPY python-requirements.txt /opt/simready/python-requirements.txt
RUN python3.12 -m venv /opt/wheel-installer && \
    /opt/wheel-installer/bin/python -m pip install --no-cache-dir --target /opt/python-site \
    --platform manylinux_2_17_x86_64 --platform manylinux_2_27_x86_64 --platform manylinux_2_28_x86_64 --python-version 3.12 --implementation cp --abi cp312 --abi none \
    --only-binary=:all: --require-hashes -r /opt/simready/python-requirements.txt

FROM nodebase AS actor
RUN export QEMU_CPU=Nehalem && apt-get update && apt-get install -y --no-install-recommends \
    blender python3.12 python3.12-venv git ripgrep && rm -rf /var/lib/apt/lists/*
COPY --from=dependencies /opt/simready/opencode /opt/simready/opencode
COPY --from=dependencies /opt/simready/pi /opt/simready/pi
COPY --from=dependencies /tmp/harbor-pi-mcp-adapter /tmp/harbor-pi-mcp-adapter
COPY --from=dependencies /opt/python-site /usr/local/lib/python3.12/dist-packages
COPY python-requirements.txt /opt/simready/python-requirements.txt
RUN export QEMU_CPU=Nehalem && \
    ln -s /opt/simready/pi/node_modules/.bin/pi /root/.nvm/versions/node/v22.23.3/bin/pi && \
    ln -s /usr/bin/python3.12 /usr/local/bin/python && \
    mkdir -p /output /task /installed-agent && \
    test -z "$(dpkg --audit)"
ENV PI_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /output
LABEL org.opencontainers.image.title="SimReadyBench actor" \
      org.opencontainers.image.version="1.0.0" simready.harness.pi="0.85.1" simready.harness.mcp="2.34.0"
CMD ["/bin/bash"]

# Production layout: preserve stock Pi commands as a non-root actor.
FROM actor AS runtime
USER root
RUN export QEMU_CPU=Nehalem && mv /root/.nvm /opt/nvm && \
    usermod -l actor -d /home/actor ubuntu && groupmod -n actor ubuntu && \
    mkdir -p /home/actor /opt/simready/runtime/simready_bench/engine && \
    touch /opt/simready/runtime/simready_bench/__init__.py /opt/simready/runtime/simready_bench/engine/__init__.py && \
    chown -R 0:0 /opt/nvm && chmod -R go-w /opt/nvm && \
    ln -s /opt/nvm /home/actor/.nvm && ln -s /opt/nvm /root/.nvm && \
    ln -s /opt/nvm/versions/node/v22.23.3/bin/node /usr/local/bin/node && \
    ln -s /opt/nvm/versions/node/v22.23.3/bin/npm /usr/local/bin/npm && \
    ln -s /opt/nvm/versions/node/v22.23.3/bin/npx /usr/local/bin/npx && \
    ln -s /opt/simready/pi/node_modules/.bin/pi /usr/local/bin/pi && \
    ln -s /opt/simready/opencode/node_modules/opencode-linux-x64/bin/opencode /usr/local/bin/opencode && \
    chown 1000:1000 /home/actor /output /installed-agent
COPY actor_mcp.py /opt/simready/runtime/simready_bench/engine/actor_mcp.py
COPY actor_relay.py /opt/simready/runtime/simready_bench/engine/actor_relay.py
COPY pi_runtime.py /opt/simready/runtime/simready_bench/engine/pi_runtime.py
ENV HOME=/home/actor NVM_DIR=/opt/nvm \
    PYTHONPATH=/opt/simready/runtime \
    PATH=/opt/nvm/versions/node/v22.23.3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
USER 1000:1000
WORKDIR /output
LABEL org.opencontainers.image.version="1.0.0"
ENTRYPOINT []
CMD ["/bin/bash"]

# Dedicated verifier: stdlib client only, no reference or actor harness.
FROM --platform=linux/amd64 ubuntu:24.04@sha256:a2d43bffc46670d7108bb3c0468c67c5557c4c92e83e1e34cbff1322ad1b94dc AS verifier
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN export QEMU_CPU=Nehalem && apt-get update && apt-get install -y --no-install-recommends python3.12 && \
    rm -rf /var/lib/apt/lists/* && ln -s /usr/bin/python3.12 /usr/local/bin/python3 && \
    mkdir -p /tests /output /logs/verifier /opt/simready/runtime/simready_bench/engine && \
    touch /opt/simready/runtime/simready_bench/__init__.py /opt/simready/runtime/simready_bench/engine/__init__.py && \
    test -z "$(dpkg --audit)"
COPY verifier.py /opt/simready/runtime/simready_bench/engine/verifier.py
COPY nvidia_profile.json /opt/simready/runtime/simready_bench/engine/nvidia_profile.json
COPY test.sh /tests/test.sh
RUN chmod 755 /tests/test.sh
ENV PYTHONPATH=/opt/simready/runtime PYTHONDONTWRITEBYTECODE=1
WORKDIR /output
ENTRYPOINT []
CMD ["/bin/bash"]


FROM python:3.12-slim-trixie@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704 AS core
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PXR_WORK_THREAD_LIMIT=2
WORKDIR /opt/simready-build
COPY pyproject.toml ./
COPY README.md LICENSE ./
COPY src ./src
RUN PYTHONPATH=src python -m simready_eval.dependencies --role actor --output /tmp/actor.requirements.txt && \
    python -m pip install --no-cache-dir --require-hashes --no-deps -r /tmp/actor.requirements.txt && \
    python -m pip install --no-cache-dir --no-deps . && \
    useradd --uid 1000 --create-home agent && \
    mkdir -p /output/submission /task && chown -R 1000:1000 /output
WORKDIR /output
CMD ["sleep", "infinity"]

FROM core AS formal
RUN python -m simready_eval.dependencies --role feedback --output /tmp/feedback.requirements.txt && \
    python -m pip install --no-cache-dir --require-hashes --no-deps -r /tmp/feedback.requirements.txt

FROM formal AS verifier
ARG BLENDER_PACKAGE_VERSION=4.3.2+dfsg-2
RUN apt-get update && \
    apt-get install -y --no-install-recommends "blender=${BLENDER_PACKAGE_VERSION}" && \
    rm -rf /var/lib/apt/lists/* && \
    blender --version | head -1 | grep -Fx 'Blender 4.3.2'

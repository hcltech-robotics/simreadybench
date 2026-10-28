FROM python:3.12-slim-trixie@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PXR_WORK_THREAD_LIMIT=2
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git && rm -rf /var/lib/apt/lists/*
COPY simreadybench_eval-2.2.0-py3-none-any.whl /opt/wheels/simreadybench_eval-2.2.0-py3-none-any.whl
COPY actor.requirements.txt /opt/locks/actor.requirements.txt
RUN python -m pip install --no-cache-dir --require-hashes --no-deps -r /opt/locks/actor.requirements.txt && python -m pip install --no-cache-dir --no-deps '/opt/wheels/simreadybench_eval-2.2.0-py3-none-any.whl'
COPY actor-assets.json /opt/setup/actor-assets.json
COPY actor-runtime.json /opt/setup/actor-runtime.json
RUN mkdir -p /opt/setup/frozen-public && chmod 700 /opt/setup/frozen-public
COPY observation.png /opt/setup/frozen-public/observation.png
COPY source-public-instruction.txt /opt/setup/frozen-public/instruction.md
RUN python -m simready_eval.fetch --manifest /opt/setup/actor-assets.json --destination /opt/setup/frozen-public --verify-only && rm -rf /opt/setup/frozen-public
COPY observation.png instruction.md /task/
RUN python -m simready_eval.fetch --manifest /opt/setup/actor-runtime.json --destination /task --verify-only && chmod -R a+rX /task && chmod -R a-w /task
RUN useradd --create-home --uid 1000 agent && mkdir -p /output/submission && chown -R 1000:1000 /output /home/agent
ENV HOME=/home/agent
WORKDIR /output
USER 1000
CMD ["sh", "-c", "sleep infinity"]

# Exact pinned CAD skill; Harbor registers this directory before Codex starts.
COPY --chown=1000:1000 authoring-skills/ /opt/simready-authoring-skills/

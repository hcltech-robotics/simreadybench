FROM python:3.12-slim-trixie@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PXR_WORK_THREAD_LIMIT=2
ARG BLENDER_PACKAGE_VERSION=4.3.2+dfsg-2
RUN apt-get update && apt-get install -y --no-install-recommends blender=$BLENDER_PACKAGE_VERSION && blender --background --version | head -n 1 | grep -F 'Blender 4.3.2' && rm -rf /var/lib/apt/lists/*
COPY simreadybench_eval-2.2.0-py3-none-any.whl /opt/wheels/simreadybench_eval-2.2.0-py3-none-any.whl
COPY verifier.requirements.txt /opt/locks/verifier.requirements.txt
RUN python -m pip install --no-cache-dir --require-hashes --no-deps -r /opt/locks/verifier.requirements.txt && python -m pip install --no-cache-dir --no-deps '/opt/wheels/simreadybench_eval-2.2.0-py3-none-any.whl'
COPY verifier-assets.json /tests/verifier-assets.json
COPY evaluator.json /tests/evaluator.json
COPY test.sh /tests/test.sh
RUN python -m simready_eval.fetch --manifest /tests/verifier-assets.json --destination /fixtures --cache /tmp/verifier-cache && chmod -R a+rX /fixtures /tests && chmod -R a-w /fixtures /tests && chmod a+x /tests/test.sh && rm -rf /tmp/verifier-cache
WORKDIR /output
CMD ["sh", "-c", "sleep infinity"]

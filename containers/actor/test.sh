#!/bin/sh
set -eu
exec python3 -m simready_bench.engine.verifier --socket /run/simready-verifier/service.sock --output-dir /logs/verifier

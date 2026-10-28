#!/bin/sh
set -eu
exec python -m simready_eval.diagnostics evaluate --submission /output/submission --fixtures /fixtures --config /tests/evaluator.json --output /logs/verifier

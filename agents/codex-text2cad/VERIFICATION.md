# Local verification — 2026-10-29

No model inference, hosted trial, credential creation, push or publication was used.

- Pinned runtime installed from the hash-locked requirements into a fresh venv.
  `cadgen doctor` and the real headless STEP/STL/GLB generation/reimport check passed.
- Node 24.19.0 was downloaded and checksum checked. `npm ci` installed the locked
  toolchain; the explicitly selected binary reported `codex-cli 0.164.0`.
- `uv sync --frozen --no-dev` passed for the hosted project. Wheel build passed;
  archive inspection confirmed all 13 skill files, pins, runtime lock, Node pin,
  npm lock and exporter are included.
- Two per-trial registration checks reused the prepared runtime without changing
  the installation marker and verified all 13 native skill files.
- Actual stdio CAD MCP initialization and tool listing passed: `cad_show`,
  `cad_http`, `cad_export`, `cad_snapshot`, `cad_inspect`.
- The real pinned codex-acp 1.12.0 initialized over ACP without credentials.
  No session/prompt was sent to that process. Prompt interception was separately
  exercised through an echo subprocess with no model.
- The example reconstruction executed a real cadgen model, then exported static
  and articulated USD assets with STEP intermediates. Source/CAD/USD hash chains
  verified. Structural tests checked dimensions in meters, Z up, transformed mesh
  bounds, triangle indices, materials, rigid bodies, mass, collision schemas,
  hinge local frames/axis/limits and USD reopening.
- `14 passed` in the package test suite, including malformed input rejection,
  altered-final-asset detection, factory selection, instruction injection,
  baked-install lifecycle, variant identity and preserved parent usage metrics.

Environment-specific supporting artifacts (not part of the package):
`/workspace/codex-text2cad-validation/` contains tests.log, build.log, uv-sync.log,
bake-and-wheel.json, mcp.json, and static/hinge USD/STEP/generation receipts.
`/workspace/codex-text2cad-install.log` records installation/preflight.

The successful local command was:

```sh
PYTHONPATH=agents/codex-text2cad:/workspace/simreadybench/.venv/lib/python3.12/site-packages \
CAD_AGENT_TEST_PREFIX=/workspace/codex-text2cad-runtime \
/workspace/codex-text2cad-runtime/bin/cad-python -m pytest -q agents/codex-text2cad/tests
```

These checks establish local functionality and structural articulation, not a
physics simulation, benchmark score or observed model compliance. Native commands
and generation receipts must be audited together on a future authorized run.
Hosted repository/access validation is pending approved publication; the source
configuration deliberately retains a commit placeholder. Original benchmark
tasks, verifier, runner policy and existing pilot runs are unchanged.

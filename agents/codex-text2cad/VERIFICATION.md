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
After explicit publication approval, implementation commit
`15716cc553a94d3e5c75a91c71cff2b9ec41c309` was pushed on the isolated
`feat/codex-text2cad-agent` branch. The hosted source example now pins that
immutable implementation commit. Hosted service validation still requires the
existing authenticated Hub session; no hosted trial has been submitted. Original
benchmark tasks, verifier, runner policy and existing pilot runs are unchanged.

Anonymous downloads from the immutable GitHub revision matched all 26 pinned
skill/runtime/code files, including all 13 skill files. The published manifest
and resolved hosted agent example passed the installed Harbor 0.24.0
`AcpSourceManifest` and `HostedAgentConfig` schemas. This is public-source and
local-schema validation; an authenticated hosted service dry run remains pending.

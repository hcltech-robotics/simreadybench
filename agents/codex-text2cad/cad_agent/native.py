"""Harbor 0.24.0 installed-agent variant; benchmark tasks remain untouched."""

import importlib.metadata
import hashlib
import json
from pathlib import Path

from harbor.agents.installed.codex import Codex

from . import AGENT_ID, VERSION
from .workflow import PACKAGE, bundle_identity, render_instruction

REMOTE = "/opt/codex-text2cad"


class CodexText2Cad(Codex):
    @staticmethod
    def name():
        return AGENT_ID

    def __init__(self, *args, **kwargs):
        if importlib.metadata.version("harbor") != "0.24.0":
            raise RuntimeError("This adapter requires Harbor 0.24.0")
        kwargs.setdefault("version", "0.164.0")
        if kwargs["version"] != "0.164.0":
            raise ValueError("This agent pins Codex 0.164.0")
        super().__init__(*args, **kwargs)
        # Construct these every trial: baked image reuse can skip install().
        servers = self._base_config.setdefault("mcp_servers", {})
        if "cad" in servers:
            raise ValueError("cad MCP configuration belongs to the CAD agent")
        servers["cad"] = {
            "command": f"{REMOTE}/bin/cad-python",
            "args": ["-m", "cad_agent.prepare", "mcp", "--prefix", REMOTE],
            "startup_timeout_sec": 300,
            "required": True,
        }

    async def install(self, environment):
        await super().install(environment)
        await self.exec_as_root(environment, f"mkdir -p {REMOTE}")
        await environment.upload_dir(PACKAGE, f"{REMOTE}/cad_agent")
        await self.exec_as_root(environment,
            f"python3 -m cad_agent.prepare install --prefix {REMOTE}",
            env={"PYTHONPATH": REMOTE}, timeout_sec=1200)

    def render_instruction(self, instruction):
        rendered = render_instruction(super().render_instruction(instruction), REMOTE)
        logs = self.logs_dir / "cad"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "workflow-applied.json").write_text(json.dumps({
            "skill_commit": bundle_identity()["skill_commit"],
            "instruction_sha256": hashlib.sha256(rendered.encode()).hexdigest()}, indent=2) + "\n")
        return rendered

    async def run(self, instruction, environment, context):
        # Never rely on constructor state that existed only at image-bake time.
        await self.exec_as_agent(environment,
            f"{REMOTE}/bin/cad-python -m cad_agent.prepare trial --prefix {REMOTE}",
            timeout_sec=120)
        try:
            await super().run(instruction, environment, context)
        finally:
            # Audit failure must not mask the original run failure or erase logs.
            try:
                await self.exec_as_agent(environment,
                    f"{REMOTE}/bin/cad-python -m cad_agent.prepare verify --prefix {REMOTE}")
                await environment.download_dir("/logs/agent/cad", self.logs_dir / "cad")
                await environment.download_dir("/output/authoring/cad-evidence", self.logs_dir / "cad" / "generation")
            except Exception:
                self.logger.exception("CAD evidence collection incomplete")

    def convert_trajectory(self, logs_dir: Path):
        trajectory = super().convert_trajectory(logs_dir)
        if trajectory:
            trajectory.agent.name = AGENT_ID
            trajectory.agent.extra = {**(trajectory.agent.extra or {}),
                "variant_version": VERSION, "base_agent": "codex",
                "cad_bundle": bundle_identity()}
        return trajectory

    def populate_context_post_run(self, context):
        super().populate_context_post_run(context)
        use = self.logs_dir / "cad/use.json"
        context.metadata = {**(context.metadata or {}), "cad_agent": {
            "id": AGENT_ID, "version": VERSION, "bundle": bundle_identity(),
            "use": json.loads(use.read_text()) if use.exists() else {"status": "unverified"}}}

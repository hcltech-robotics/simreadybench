"""Public task files and capture timing shared by Harbor services."""

from simready_bench.engine.native_output import EXTENSION_FILENAME

PUBLIC_FILES = frozenset(
    {
        "instruction.md",
        "thumbnail.png",
        "contract.md",
        "tool-docs.md",
        "scoring-policy.json",
        EXTENSION_FILENAME,
    }
)

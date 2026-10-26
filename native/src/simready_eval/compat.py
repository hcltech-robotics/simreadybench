"""Scoped compatibility for the pinned NVIDIA material connection reader."""

from contextlib import contextmanager
from functools import wraps
import hashlib
import importlib
import inspect
from pathlib import Path
import threading

ADAPTER_ID = "simready-nvidia-material-surface-attribute-v1"
RULE_MODULE = (
    "simready.foundation.tier_core.capabilities.visualization.materials.validation"
)
RULE_SOURCE_SHA256 = "9715c4e0d70b09110d1c5efe788065befc8ecce4eb3802f4cbd854fd7762c25b"
_LOCK = threading.RLock()


def compatibility_identity():
    """Bind the adapter implementation and the unchanged upstream implementation."""
    return {
        "id": ADAPTER_ID,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "upstream_module": RULE_MODULE,
        "upstream_source_sha256": RULE_SOURCE_SHA256,
        "methods": ["_connected_surface_terminals", "_terminal_shader"],
        "scope": "authored_relationship_at_builtin_material_surface_attribute",
    }


def relationship_at_material_surface(prim, terminal="outputs:surface"):
    """Identify the precise authored spec that cannot provide an attribute connection.

    GetAttribute is truthy for this malformed property because Material defines an
    attribute fallback. Its strongest authored Sdf spec still is a relationship,
    so querying attribute connections raises an OpenUSD verification error.
    """
    if terminal != "outputs:surface":
        return False
    from pxr import Sdf

    definition = prim.GetPrimDefinition().GetPropertyDefinition(terminal)
    if not definition or not definition.IsAttribute():
        return False
    attribute = prim.GetAttribute(terminal)
    specs = attribute.GetPropertyStack() if attribute else []
    return bool(specs) and isinstance(specs[0], Sdf.RelationshipSpec)


@contextmanager
def material_surface_connections():
    """Let native requirements judge a malformed surface as having no connection.

    Only the two upstream connection readers are guarded. Native requirement
    methods emit their own findings, and other exceptions reach the SDK normally.
    The installed module, the stage and submitted bytes are never changed.
    """
    with _LOCK:
        module = importlib.import_module(RULE_MODULE)
        source = Path(inspect.getfile(module))
        if hashlib.sha256(source.read_bytes()).hexdigest() != RULE_SOURCE_SHA256:
            raise RuntimeError("Pinned NVIDIA material compatibility source changed")
        checker = module.VisualMaterialsCapabilityChecker
        connected = checker._connected_surface_terminals
        shader = checker._terminal_shader
        if any(
            Path(inspect.getfile(method)) != source
            for method in (connected, shader)
        ):
            raise RuntimeError("Pinned NVIDIA material connection reader changed")

        @wraps(connected)
        def guarded_connections(self, prim):
            if not relationship_at_material_surface(prim):
                return connected(self, prim)
            result = []
            for name in self.surface_terminals:
                if name == "outputs:surface":
                    continue
                attribute = prim.GetAttribute(name)
                if attribute and attribute.IsAuthored() and attribute.GetConnections():
                    result.append((name, attribute))
            return result

        @wraps(shader)
        def guarded_shader(self, prim, terminal):
            if relationship_at_material_surface(prim, terminal):
                return None
            return shader(self, prim, terminal)

        checker._connected_surface_terminals = guarded_connections
        checker._terminal_shader = guarded_shader
        try:
            yield compatibility_identity()
        finally:
            checker._connected_surface_terminals = connected
            checker._terminal_shader = shader

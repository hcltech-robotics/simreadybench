"""Loopback inference admission shared by local benchmark processes."""

from contextlib import contextmanager
from pathlib import Path
import secrets

from .daytona_route_throttle import requires_route_throttle, route_identity
from .route_throttle import RouteThrottleServer


@contextmanager
def local_route_throttles(resolved):
    policy = resolved["operator_policy"]
    routes = {}
    for condition in resolved["conditions"]:
        model = condition["model"]
        manifest = {"models": [model]}
        if requires_route_throttle(policy=policy, manifest=manifest):
            routes[(model["provider_ref"], model["model_id"])] = route_identity(policy, manifest)
    if not routes:
        yield {}
        return
    directory = Path(policy["runtime"]["authority_state_dir"]).absolute() / "route-throttles"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.resolve() != directory:
        raise ValueError("The local route governor requires a regular authority directory")
    token = secrets.token_urlsafe(32)
    # Separate coordinators may expose different loopback ports. SQLite shares
    # their route permits and cooldowns through the persistent authority store.
    with RouteThrottleServer(directory / "local.sqlite3", token, route_keys=set(routes.values())) as server:
        yield {reference: {"url": server.url, "token": token, "route_key": route}
               for reference, route in routes.items()}

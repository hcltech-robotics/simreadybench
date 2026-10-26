"""Read-only inspection of malformed USD material terminal properties."""

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


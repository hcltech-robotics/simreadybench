"""Dependency scanning regressions; the USD boundary is stubbed in these CPU unit tests."""

from types import ModuleType, SimpleNamespace
import sys
import pytest
from simready_bench.engine.package_checks import inspect_package


@pytest.fixture
def dependency_scan_only(monkeypatch):
    def composition_boundary(*args):
        raise RuntimeError("test stops after dependency scan")

    pxr = ModuleType("pxr")
    pxr.Sdf = SimpleNamespace(
        Layer=SimpleNamespace(OpenAsAnonymous=lambda path: object())
    )
    pxr.Usd = SimpleNamespace(Stage=SimpleNamespace(Open=composition_boundary))
    pxr.UsdUtils = SimpleNamespace(ModifyAssetPaths=lambda layer, callback: None)
    for name in ("UsdGeom", "UsdPhysics", "UsdValidation", "Gf"):
        setattr(pxr, name, SimpleNamespace())
    monkeypatch.setitem(sys.modules, "pxr", pxr)


def test_local_mdl_resource_passes_the_offline_dependency_scan(
    tmp_path, dependency_scan_only
):
    (tmp_path / "asset.usda").write_text("#usda 1.0\n")
    (tmp_path / "texture.png").write_bytes(b"scan-only-resource")
    (tmp_path / "material.mdl").write_text(
        'export material m() = material();\ntexture_2d("texture.png");\n'
    )
    report = inspect_package(tmp_path, entrypoint="asset.usda")
    assert report["infrastructure_errors"] == []
    assert report["package_ok"] is True
    assert report["composition_attempted"] is False
    assert report["findings"] == []


@pytest.mark.parametrize(
    "mdl,code",
    [
        ('texture_2d("/outside/texture.png");', "external_dependency"),
        ('texture_2d("missing.png");', "missing_dependency"),
        ("import .relative::module::*;", "unsupported_mdl_import"),
    ],
)
def test_unsafe_mdl_resource_stops_before_composition(
    tmp_path, dependency_scan_only, mdl, code
):
    (tmp_path / "asset.usda").write_text("#usda 1.0\n")
    (tmp_path / "material.mdl").write_text(mdl)
    report = inspect_package(tmp_path, entrypoint="asset.usda")
    assert report["infrastructure_errors"] == []
    assert report["package_ok"] is False
    assert report["composition_attempted"] is False
    assert any(x["code"] == code for x in report["findings"])

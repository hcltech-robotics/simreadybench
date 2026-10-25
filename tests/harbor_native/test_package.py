"""Package inspection rejects dangerous authored paths before composition."""
import os
import zipfile

import pytest
from pxr import Usd

from simready_eval.package import inspect_package


def asset(root, body='def Xform "Asset" {}'):
    path = root / "asset.usd"
    path.write_text('#usda 1.0\n(defaultPrim = "Asset")\n' + body + '\n')
    return path


def test_local_package_is_scanned_without_composition(tmp_path, monkeypatch):
    asset(tmp_path)
    monkeypatch.setattr(Usd.Stage, "Open", lambda *args: pytest.fail("No composition during preflight"))
    result = inspect_package(tmp_path)
    assert result["package_ok"] is True and result["infrastructure_errors"] == []
    assert result["composition_attempted"] is False
    assert result["inspection_scope"] == "package_dependencies"


@pytest.mark.parametrize("kind", ["missing", "symlink", "hardlink", "fifo", "root_symlink", "archive_escape", "archive_symlink", "archive_special", "size", "count"])
def test_proven_invalid_package_forms(tmp_path, monkeypatch, kind):
    from simready_eval import package
    root = tmp_path
    if kind != "missing":
        asset(root)
    if kind == "symlink":
        (root / "link").symlink_to(root / "asset.usd")
    elif kind == "hardlink":
        os.link(root / "asset.usd", root / "copy.usd")
    elif kind == "fifo":
        os.mkfifo(root / "pipe")
    elif kind == "root_symlink":
        target = root / "real"
        target.mkdir()
        asset(target)
        root = root / "linked"
        root.symlink_to(target, target_is_directory=True)
    elif kind.startswith("archive_"):
        (root / "asset.usd").unlink()
        with zipfile.ZipFile(root / "asset.usdz", "w") as archive:
            if kind == "archive_escape":
                archive.writestr("../escape.usda", "#usda 1.0")
            else:
                info = zipfile.ZipInfo("bad")
                info.external_attr = (0o120777 if kind == "archive_symlink" else 0o010600) << 16
                archive.writestr(info, "target")
    elif kind == "size":
        monkeypatch.setattr(package, "LIMIT_BYTES", 10)
    elif kind == "count":
        monkeypatch.setattr(package, "LIMIT_FILES", 1)
        (root / "directory").mkdir()
    result = inspect_package(root, entrypoint="asset.usdz" if kind.startswith("archive_") else "asset.usd")
    assert result["package_ok"] is False
    assert result["proven_invalid"] is True
    assert result["infrastructure_errors"] == []
    assert result["composition_attempted"] is False


@pytest.mark.parametrize("reference", ["https://example.test/a.usd", "../escape.usd", "/etc/passwd", "missing.usd"])
def test_unsafe_or_missing_authored_dependency_blocks_composition(tmp_path, reference):
    asset(tmp_path, 'def Xform "Asset" (prepend references = @' + reference + '@) {}')
    result = inspect_package(tmp_path)
    assert result["package_ok"] is False and result["proven_invalid"] is True
    assert result["composition_attempted"] is False


def test_unexpected_layer_inspection_error_is_infrastructure(tmp_path, monkeypatch):
    from pxr import Sdf
    asset(tmp_path)
    def broken(*args):
        raise RuntimeError("scanner broke")
    monkeypatch.setattr(Sdf.Layer, "OpenAsAnonymous", broken)
    result = inspect_package(tmp_path)
    assert result["proven_invalid"] is False
    assert result["infrastructure_errors"] == ["package_checker_error:RuntimeError"]


def test_archive_aliasing_member_cannot_overwrite_prior_member(tmp_path):
    with zipfile.ZipFile(tmp_path / "asset.usdz", "w") as archive:
        archive.writestr("dir/asset.usda", '#usda 1.0\ndef Xform "Asset" {}')
        archive.writestr("dir//asset.usda", "replacement")
    report = inspect_package(tmp_path, entrypoint="asset.usdz")
    assert report["proven_invalid"] is True
    assert any(row["code"] == "unsafe_archive_member" for row in report["findings"])


def test_archive_io_failure_is_infrastructure(tmp_path, monkeypatch):
    with zipfile.ZipFile(tmp_path / "asset.usdz", "w") as archive:
        archive.writestr("asset.usda", '#usda 1.0\ndef Xform "Asset" {}')
    def broken(*args, **kwargs):
        raise OSError("scratch storage failure")
    monkeypatch.setattr(zipfile.ZipFile, "open", broken)
    report = inspect_package(tmp_path, entrypoint="asset.usdz")
    assert report["proven_invalid"] is False
    assert report["infrastructure_errors"] == ["package_checker_error:OSError"]


@pytest.mark.parametrize("external", [False, True])
def test_reference_nested_entrypoint_uses_whole_package_boundary(tmp_path, external):
    from PIL import Image
    root = tmp_path / "reference"
    (root / "simready_usd").mkdir(parents=True)
    (root / "Materials").mkdir()
    Image.new("RGB", (2, 2), (120, 30, 20)).save(root / "Materials/texture.png")
    Image.new("RGB", (2, 2)).save(tmp_path / "outside.png")
    dependency = "../../outside.png" if external else "../Materials/texture.png"
    path = root / "simready_usd/sm_nested.usd"
    path.write_text('#usda 1.0\n(defaultPrim = "Asset")\ndef Xform "Asset" {\n'
                    ' custom asset sourceTexture = @' + dependency + '@\n}\n')
    before = path.read_bytes()
    result = inspect_package(root, entrypoint="simready_usd/sm_nested.usd",
                             allow_nested_entrypoint=True)
    assert result["package_ok"] is (not external)
    assert result["proven_invalid"] is external
    assert result["infrastructure_errors"] == []
    assert result["composition_attempted"] is False
    if external:
        assert any(row["code"] == "external_dependency" for row in result["findings"])
    else:
        assert result["inspection_scope"] == "package_dependencies"
    assert path.read_bytes() == before


def test_nested_candidate_entrypoint_stays_forbidden_by_default(tmp_path):
    nested = tmp_path / "simready_usd"
    nested.mkdir()
    asset(nested)
    result = inspect_package(tmp_path, entrypoint="simready_usd/asset.usd")
    assert result["package_ok"] is False and result["proven_invalid"] is True
    assert any(row["code"] == "invalid_entrypoint" for row in result["findings"])


def test_trusted_reference_bytes_are_separate_from_declared_candidate_limit(tmp_path):
    asset(tmp_path)
    with (tmp_path / 'data.bin').open('wb') as stream:
        stream.truncate(1024**3 + 1)
    candidate = inspect_package(tmp_path)
    assert not candidate['package_ok']
    assert any(row['code'] == 'package_limit_exceeded' for row in candidate['findings'])
    reference = inspect_package(tmp_path, trusted_reference=True)
    assert reference['package_ok'] and not reference['infrastructure_errors']
    assert reference['metrics']['package_bytes'] > 1024**3


def test_one_gib_candidate_boundary_is_inclusive_without_composing_sparse_data(tmp_path):
    entry = asset(tmp_path)
    data = tmp_path / 'payload.bin'
    with data.open('wb') as stream:
        stream.truncate(1024**3 - entry.stat().st_size)
    result = inspect_package(tmp_path)
    assert result['package_ok'] and result['metrics']['package_bytes'] == 1024**3
    with data.open('r+b') as stream:
        stream.truncate(data.stat().st_size + 1)
    result = inspect_package(tmp_path)
    assert result['proven_invalid']
    assert any(row['code'] == 'package_limit_exceeded' for row in result['findings'])


@pytest.mark.parametrize('required_bytes', [503323398, 452060328, 411827612, 364169524, 289041323])
def test_retained_reference_closure_byte_sizes_fit_candidate_allowance(tmp_path, required_bytes):
    # Asset-owner inventories are exact byte counts. Sparse fixtures test the
    # byte admission rule only, not source semantics or runtime qualification.
    entry = asset(tmp_path)
    with (tmp_path / 'closure-data.bin').open('wb') as stream:
        stream.truncate(required_bytes - entry.stat().st_size)
    result = inspect_package(tmp_path)
    assert result['package_ok'] and result['metrics']['package_bytes'] == required_bytes


def test_usdz_expansion_cannot_spend_tar_header_allowance(tmp_path, monkeypatch):
    from simready_eval import package
    monkeypatch.setattr(package, 'LIMIT_BYTES', 4096)
    with zipfile.ZipFile(tmp_path / 'asset.usdz', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('asset.usda', '#usda 1.0\ndef Xform "Asset" {}')
        archive.writestr('large.bin', b'\x00' * 8192)
    assert (tmp_path / 'asset.usdz').stat().st_size < 4096
    result = inspect_package(tmp_path, entrypoint='asset.usdz')
    assert result['proven_invalid']
    assert any(row['code'] == 'expanded_package_limit_exceeded' for row in result['findings'])

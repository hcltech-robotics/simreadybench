"""Reference staging includes declared resources without neighbouring files."""

import pytest

from simready_bench.asset_inputs import dependency_files
from simready_bench.benchmark import stage_assets, stage_skills


def stage(root, entries, *, name="output", assets_root=None):
    output = root / name
    manifest = {"assets": entries}
    stage_assets(
        manifest,
        base_dir=root,
        destination=output / "assets",
        output_dir=output,
        assets_root=assets_root,
        package_inputs=True,
    )
    return manifest, output


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_dependency_selection_ignores_siblings_and_follows_layers_materials_and_udim(
    tmp_path,
):
    root = tmp_path / "case"
    write(
        root / "asset.usda",
        """#usda 1.0
(
    subLayers = [@layers/base.usda@]
)
def Xform "Asset" {
    asset direct = @textures/direct.png@
    asset tiles = @textures/tile.<UDIM>.png@
    asset mdl = @materials/material.mdl@
    asset materialx = @materials/material.mtlx@
}
""",
    )
    write(
        root / "layers/base.usda",
        '#usda 1.0\ndef Cube "Part" {\n    asset texture = @../textures/layer.png@\n}\n',
    )
    write(
        root / "materials/material.mdl",
        """mdl 1.7;
import ::df::*;
texture_2d("../textures/mdl.png");
// texture_2d("../.env");
/* texture_2d("../missing.png"); */
""",
    )
    write(
        root / "materials/material.mtlx",
        """<materialx>
    <input name="file" type="filename" value="../textures/materialx.png"/>
    <include href="nested.mtlx"/>
    <input name="unset" type="filename" value=""/>
</materialx>""",
    )
    write(
        root / "materials/nested.mtlx",
        '<materialx><image file="../textures/nested.png"/></materialx>',
    )
    textures = [
        "direct.png",
        "layer.png",
        "mdl.png",
        "materialx.png",
        "nested.png",
        "tile.1001.png",
        "tile.1002.png",
    ]
    for name in textures:
        write(root / "textures" / name, name)
    write(root / ".env", "EXAMPLE_KEY=placeholder")
    write(root / ".git/config", "placeholder repository data")
    write(root / "textures/unrelated.png", "unrelated pixels")
    write(root / "notes.txt", "unrelated notes")
    result, output = stage(tmp_path, ["case/asset.usda"])
    entry = result["assets"][0]
    copied = {
        path.relative_to(output / "assets/packages/asset").as_posix()
        for path in (output / "assets/packages/asset").rglob("*")
        if path.is_file()
    }
    assert copied == {
        "asset.usda",
        "layers/base.usda",
        "materials/material.mdl",
        "materials/material.mtlx",
        "materials/nested.mtlx",
        *["textures/" + name for name in textures],
    }
    assert len(entry["dependencies"]) == len(copied) - 1
    assert (output / "assets" / entry["usd"]).read_bytes() == (
        root / "asset.usda"
    ).read_bytes()


def test_binary_usdc_paths_are_selected_without_stage_composition(
    tmp_path, monkeypatch
):
    from pxr import Sdf, Usd

    root = tmp_path / "binary"
    root.mkdir()
    layer = Sdf.Layer.CreateNew(str(root / "asset.usdc"))
    prim = Sdf.PrimSpec(layer, "Asset", Sdf.SpecifierDef, "Xform")
    attribute = Sdf.AttributeSpec(prim, "texture", Sdf.ValueTypeNames.Asset)
    attribute.default = Sdf.AssetPath("texture.png")
    layer.Save()
    write(root / "texture.png", "reference pixels")
    write(root / "unused.png", "unrelated pixels")
    monkeypatch.setattr(
        Usd.Stage,
        "Open",
        lambda *_args, **_kwargs: pytest.fail("Dependency selection composed a stage"),
    )
    assert {path.name for path in dependency_files(root / "asset.usdc", root)} == {
        "asset.usdc",
        "texture.png",
    }


def test_materialx_filename_prefix_inherits_overrides_and_can_be_cleared(tmp_path):
    write(
        tmp_path / "look.mtlx",
        """<materialx fileprefix="textures/">
      <image name="inherited"><input type="filename" value="albedo.png"/></image>
      <image name="override" fileprefix="other/"><input type="filename" value="albedo.png"/></image>
      <image name="cleared"><input type="filename" fileprefix="" value="clear.png"/></image>
    </materialx>""",
    )
    for name in ("textures/albedo.png", "other/albedo.png", "clear.png", "albedo.png"):
        write(tmp_path / name, name)
    assert {
        path.relative_to(tmp_path).as_posix()
        for path in dependency_files(tmp_path / "look.mtlx", tmp_path)
    } == {
        "look.mtlx",
        "textures/albedo.png",
        "other/albedo.png",
        "clear.png",
    }


def test_materialx_include_href_is_relative_to_its_own_document(tmp_path):
    write(
        tmp_path / "look.mtlx",
        """<materialx xmlns:xi="http://www.w3.org/2001/XInclude">
      <image fileprefix="textures/"><input type="filename" value="albedo.png"/></image>
      <xi:include href="libraries/nodes.mtlx"/>
    </materialx>""",
    )
    write(
        tmp_path / "libraries/nodes.mtlx",
        '<materialx fileprefix="../textures/"><image><input type="filename" value="secondary.png"/></image></materialx>',
    )
    for name in (
        "textures/albedo.png",
        "textures/secondary.png",
        "libraries/secondary.png",
        "albedo.png",
    ):
        write(tmp_path / name, name)
    assert {
        path.relative_to(tmp_path).as_posix()
        for path in dependency_files(tmp_path / "look.mtlx", tmp_path)
    } == {
        "look.mtlx",
        "libraries/nodes.mtlx",
        "textures/albedo.png",
        "textures/secondary.png",
    }


@pytest.mark.parametrize(
    "xml",
    [
        '<materialx fileprefix="textures/"><include href="nodes.mtlx"/></materialx>',
        '<materialx xml:base="textures/"><image><input type="filename" value="albedo.png"/></image></materialx>',
        '<materialx><image><input type="filename" value="albedo.[variant].png"/></image></materialx>',
        '<materialx><image><input type="filename" value="albedo.{frame}.png"/></image></materialx>',
        '<materialx><image><input type="filename" value="albedo.&lt;UVTILE&gt;.png"/></image></materialx>',
    ],
)
def test_unsupported_materialx_resolution_requires_an_explicit_manifest(tmp_path, xml):
    write(tmp_path / "look.mtlx", xml)
    with pytest.raises(ValueError, match="explicit asset manifest"):
        dependency_files(tmp_path / "look.mtlx", tmp_path)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "utf-32"])
def test_materialx_declarations_are_rejected_before_xml_or_network_resolution(
    tmp_path, monkeypatch, encoding
):
    import socket
    import urllib.request
    from simready_bench import asset_inputs

    write(
        tmp_path / "look.mtlx",
        """<!DOCTYPE materialx [<!ENTITY remote SYSTEM "https://example.org/private">]>
    <materialx><image><input type="filename" value="&remote;"/></image></materialx>""",
    )
    material = tmp_path / "look.mtlx"
    material.write_bytes(material.read_text().encode(encoding))

    def forbidden(*_args, **_kwargs):
        pytest.fail("The dependency scan resolved declared XML content")

    monkeypatch.setattr(asset_inputs.ET, "fromstring", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    with pytest.raises(ValueError, match="explicit asset manifest"):
        dependency_files(tmp_path / "look.mtlx", tmp_path)


@pytest.mark.parametrize(
    "expression",
    [
        'texture_2d("textures/albedo.png")',
        'texture_2d(name: "textures/albedo.png")',
        'uniform texture_2d tex = "textures/albedo.png";',
        'export material test(uniform texture_2d tex = texture_2d(name: "textures/albedo.png")) = material();',
    ],
)
def test_mdl_literal_resources_include_named_arguments_and_typed_defaults(
    tmp_path, expression
):
    write(tmp_path / "look.mdl", "mdl 1.7;\n" + expression)
    write(tmp_path / "textures/albedo.png", "selected resource")
    write(tmp_path / "unrelated.png", "unrelated pixels")
    assert {
        path.relative_to(tmp_path).as_posix()
        for path in dependency_files(tmp_path / "look.mdl", tmp_path)
    } == {"look.mdl", "textures/albedo.png"}


@pytest.mark.parametrize(
    "expression",
    [
        "texture_2d(name: filename)",
        'texture_2d("textures/" + filename)',
        "uniform texture_2d tex = DEFAULT_TEXTURE;",
        "uniform texture_2d tex = texture_2d(name: filename);",
    ],
)
def test_unresolved_mdl_resource_expressions_require_an_explicit_manifest(
    tmp_path, expression
):
    write(tmp_path / "look.mdl", "mdl 1.7;\n" + expression)
    with pytest.raises(ValueError, match="explicit asset manifest"):
        dependency_files(tmp_path / "look.mdl", tmp_path)


def test_qualified_custom_mdl_import_requires_an_explicit_file_manifest(tmp_path):
    write(
        tmp_path / "asset.usda",
        '#usda 1.0\ndef Xform "Asset" {\n    asset mdl = @materials/material.mdl@\n}\n',
    )
    write(
        tmp_path / "materials/material.mdl", "mdl 1.7;\nimport ::materials::foo::*;\n"
    )
    write(tmp_path / "materials/foo.mdl", "mdl 1.7;\nimport ::df::*;\n")
    with pytest.raises(ValueError, match="explicit asset manifest"):
        stage(tmp_path, ["asset.usda"])
    assert not (tmp_path / "output").exists()
    result, output = stage(
        tmp_path,
        [
            {
                "id": "reference",
                "usd": "asset.usda",
                "dependencies": ["materials/material.mdl", "materials/foo.mdl"],
            }
        ],
    )
    assert result["assets"][0]["dependencies"] == [
        "materials/foo.mdl",
        "materials/material.mdl",
    ]
    assert (output / "assets/materials/foo.mdl").is_file()


def test_shared_dependencies_use_only_the_selected_assets_root(tmp_path):
    write(
        tmp_path / "case/asset.usda",
        '#usda 1.0\ndef Xform "Asset" {\n    asset file = @../shared/texture.png@\n}\n',
    )
    write(tmp_path / "shared/texture.png", "selected pixels")
    write(tmp_path / "shared/unrelated.png", "unrelated pixels")
    with pytest.raises(ValueError, match="assets-root"):
        stage(tmp_path, ["case/asset.usda"])
    assert not (tmp_path / "output").exists()
    result, output = stage(tmp_path, ["case/asset.usda"], assets_root=tmp_path)
    entry = result["assets"][0]
    assert (output / "assets" / entry["usd"]).is_file()
    assert (
        output / "assets/packages/asset/shared/texture.png"
    ).read_text() == "selected pixels"
    assert not (output / "assets/packages/asset/shared/unrelated.png").exists()


@pytest.mark.parametrize(
    "reference",
    ["../outside.png", "/tmp/outside.png", "https://example.org/private.png", ".env"],
)
def test_unsafe_authored_paths_fail_before_copying_any_asset(tmp_path, reference):
    write(tmp_path / "first/asset.usda", '#usda 1.0\ndef Xform "First" {}\n')
    write(
        tmp_path / "second/asset.usda",
        '#usda 1.0\ndef Xform "Second" {\n    asset file = @' + reference + "@\n}\n",
    )
    write(tmp_path / "second/.env", "EXAMPLE_KEY=placeholder")
    write(tmp_path / "outside.png", "outside package")
    with pytest.raises(ValueError):
        stage(tmp_path, ["first/asset.usda", "second/asset.usda"])
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "dependency",
    [
        ".env",
        ".env.production",
        ".git/config",
        ".venv/pyvenv.cfg",
        "id_ed25519",
        "secret.pem",
        "helper.py",
    ],
)
def test_explicit_manifests_reject_non_asset_private_files(tmp_path, dependency):
    write(tmp_path / "asset.usda", "#usda 1.0\n")
    write(tmp_path / dependency, "placeholder")
    with pytest.raises(ValueError, match="credentials, project metadata or executable"):
        stage(
            tmp_path,
            [{"id": "reference", "usd": "asset.usda", "dependencies": [dependency]}],
        )
    assert not (tmp_path / "output").exists()


def test_explicit_manifest_is_an_exact_selection_without_usd_inspection(
    tmp_path, monkeypatch
):
    from simready_bench import asset_inputs

    write(
        tmp_path / "asset.usdc", "binary fixture not inspected for explicit selection"
    )
    write(tmp_path / "resources/material.mdl", "selected material")
    write(tmp_path / "reference.png", "selected observation")
    write(tmp_path / "unrelated.png", "unrelated pixels")
    monkeypatch.setattr(
        asset_inputs,
        "dependency_files",
        lambda *_: pytest.fail("An explicit manifest was expanded"),
    )
    result, output = stage(
        tmp_path,
        [
            {
                "id": "reference",
                "usd": "asset.usdc",
                "dependencies": ["resources/material.mdl"],
                "source_view": "reference.png",
            }
        ],
    )
    assert {
        path.relative_to(output / "assets").as_posix()
        for path in (output / "assets").rglob("*")
        if path.is_file()
    } == {"asset.usdc", "resources/material.mdl", "reference.png"}
    assert result["assets"][0]["dependencies"] == [
        "reference.png",
        "resources/material.mdl",
    ]


def test_common_basenames_get_stable_package_ids(tmp_path):
    entries = ["case/asset.usda", "drawer/asset.usda"]
    for entry in entries:
        write(tmp_path / entry, "#usda 1.0\n")
    result, _ = stage(tmp_path, entries)
    reverse, _ = stage(tmp_path, list(reversed(entries)), name="reverse")
    assert {entry["id"] for entry in result["assets"]} == {"case-asset", "drawer-asset"}
    assert {entry["id"]: entry["usd"] for entry in result["assets"]} == {
        entry["id"]: entry["usd"] for entry in reverse["assets"]
    }


def test_long_path_ids_are_order_independent(tmp_path):
    prefix = "x" * 70
    entries = [prefix + "a/asset.usda", prefix + "b/asset.usda"]
    for entry in entries:
        write(tmp_path / entry, "#usda 1.0\n")
    result, _ = stage(tmp_path, entries)
    reverse, _ = stage(tmp_path, list(reversed(entries)), name="reverse")
    assert len({entry["id"] for entry in result["assets"]}) == 2
    assert {entry["id"] for entry in result["assets"]} == {
        entry["id"] for entry in reverse["assets"]
    }


def test_selected_symlinks_fail_but_unselected_siblings_are_ignored(tmp_path):
    write(tmp_path / "case/asset.usda", "#usda 1.0\n")
    write(tmp_path / "outside.png", "outside pixels")
    link = tmp_path / "case/texture.png"
    try:
        link.symlink_to(tmp_path / "outside.png")
    except OSError:
        pytest.skip("Creating symbolic links is unavailable")
    stage(tmp_path, ["case/asset.usda"])
    write(
        tmp_path / "case/asset.usda",
        '#usda 1.0\ndef Xform "Asset" {\n    asset file = @texture.png@\n}\n',
    )
    with pytest.raises(ValueError, match="without links"):
        stage(tmp_path, ["case/asset.usda"], name="selected")
    assert not (tmp_path / "selected").exists()


def test_conflicting_explicit_paths_fail_before_copy(tmp_path):
    write(tmp_path / "asset.usda", "#usda 1.0\n")
    with pytest.raises(ValueError, match="own id"):
        stage(
            tmp_path,
            [{"id": "same", "usd": "asset.usda"}, {"id": "same", "usd": "asset.usda"}],
        )
    assert not (tmp_path / "output").exists()


def test_selected_skill_folders_keep_their_complete_contents(tmp_path):
    root = tmp_path / "skill"
    write(root / "SKILL.md", "Use the selected authoring helper")
    write(root / "scripts/helper.py", "print('authoring helper')")
    selected = stage_skills([root], tmp_path / "skills")
    package = tmp_path / "skills" / selected[0]["paths"][0]
    assert (package / "scripts/helper.py").is_file()

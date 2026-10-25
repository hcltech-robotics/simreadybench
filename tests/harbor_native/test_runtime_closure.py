"""Runtime acquisition selects authored dependencies, not the source archive."""
import hashlib

import pytest
from pxr import Sdf, Usd, UsdGeom

from simready_eval import prepare


def package(tmp_path):
    root = tmp_path / 'package'
    (root / 'nested').mkdir(parents=True)
    source = root / 'nested/main.usda'
    stage = Usd.Stage.CreateNew(str(source))
    stage.GetRootLayer().subLayerPaths = ['../shared.usda']
    stage.GetRootLayer().Save()
    shared = Usd.Stage.CreateNew(str(root / 'shared.usda'))
    prim = UsdGeom.Cube.Define(shared, '/Cube').GetPrim()
    prim.CreateAttribute('texture', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('colour.png'))
    shared.GetRootLayer().Save()
    (root / 'colour.png').write_bytes(b'pinned texture bytes')
    (root / '.metadata').mkdir()
    (root / '.metadata/LICENSE.txt').write_text('Source attribution.\n')
    (root / '.metadata/archive-bom.json').write_text('{}')
    (root / 'archive').mkdir()
    (root / 'archive/unused.usda').write_text('An unrelated original archive, not a composed layer.')
    (root / 'archive/unused.usdz').write_bytes(b'unrelated original archive bytes')
    return root, source


def test_inventory_is_read_only_and_keeps_nested_dependencies_and_license(tmp_path):
    root, source = package(tmp_path)
    before = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.rglob('*') if p.is_file()}
    result = prepare.runtime_closure(source, root)
    assert result['schema'] == 'simready-runtime-closure-v1'
    assert result['source_entrypoint'] == 'nested/main.usda'
    assert {r['path'] for r in result['files']} == {
        'nested/main.usda', 'shared.usda', 'colour.png', '.metadata/LICENSE.txt'}
    assert set(result['excluded_files']) == {
        'archive/unused.usda', 'archive/unused.usdz', '.metadata/archive-bom.json'}
    assert all(before[r['path']] == r['sha256'] for r in result['files'])
    assert before == {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in root.rglob('*') if p.is_file()}


def test_referenced_archive_is_retained_and_missing_dependency_is_rejected(tmp_path):
    root, source = package(tmp_path)
    stage = Usd.Stage.Open(str(source))
    prim = stage.DefinePrim('/Reference')
    prim.CreateAttribute('archive', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('../archive/unused.usdz[inner.usda]'))
    stage.GetRootLayer().Save()
    result = prepare.runtime_closure(source, root)
    assert 'archive/unused.usdz' in {row['path'] for row in result['files']}
    (root / 'colour.png').unlink()
    with pytest.raises(ValueError, match='unrecorded_dependency'):
        prepare.runtime_closure(source, root)


def test_public_instruction_retains_nominal_dimensions_without_renderer(tmp_path):
    text = prepare.public_instruction([.25, .4, .15])
    assert 'Nominal dimensions in canonical metres XYZ: [0.25, 0.4, 0.15]' in text
    assert '256 MiB' in text and '10000 files/directories' in text
    assert 'Do not use Blender for authoring' in text


def test_relative_udim_tiles_are_canonical_members_of_the_runtime_closure(tmp_path):
    root, source = package(tmp_path)
    (root / 'tiles').mkdir()
    for number in (1001, 1002):
        (root / f'tiles/colour.{number}.png').write_bytes(b'pinned tile')
    stage = Usd.Stage.Open(str(source))
    stage.DefinePrim('/Tiles').CreateAttribute('texture', Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath('../tiles/colour.<UDIM>.png'))
    stage.GetRootLayer().Save()
    inventory = prepare.runtime_closure(source, root)
    assert {'tiles/colour.1001.png', 'tiles/colour.1002.png'} <= {
        row['path'] for row in inventory['files']}

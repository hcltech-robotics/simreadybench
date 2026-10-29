import json, importlib.metadata as md, math
from pathlib import Path
import build123d as bd
from cadgen import model
ROOT=Path(__file__).resolve().parent
@model
def smoke_bracket():
    shape=bd.Box(40,20,6)-bd.Cylinder(4,6)
    shape.label='headless-smoke-bracket'
    return shape
if __name__=='__main__':
    doc=smoke_bracket()
    doc.export(ROOT/'smoke.step',ROOT/'smoke.stl',ROOT/'smoke.glb')
    imported=bd.import_step(str(ROOT/'smoke.step'))
    expected=40*20*6-math.pi*4**2*6
    assert imported.is_valid
    assert math.isclose(imported.volume,expected,rel_tol=1e-7)
    receipt={'cadgen':md.version('cadgen'),'build123d':md.version('build123d'),'cadquery-ocp-novtk':md.version('cadquery-ocp-novtk'),'document_id':doc.id,'valid':imported.is_valid,'volume_mm3':imported.volume,'expected_volume_mm3':expected,'artifacts':{p.name:p.stat().st_size for p in (ROOT/'smoke.step',ROOT/'smoke.stl',ROOT/'smoke.glb')}}
    (ROOT/'headless-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

"""The offline driver cannot load a different producer or edit protected inputs."""
import importlib.util
from pathlib import Path

import pytest


def driver():
    path=Path(__file__).parents[2]/'native/tools/render_observation.py'
    spec=importlib.util.spec_from_file_location('render_observation_driver',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_driver_rejects_unqualified_producer_before_executing_code(tmp_path):
    module=driver();marker=tmp_path/'executed'
    producer=tmp_path/'render.py'
    producer.write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\n')
    (tmp_path/'render_worker.py').write_text('# not the qualified worker')
    with pytest.raises(ValueError):module.load_producer(producer)
    assert not marker.exists()


def test_driver_does_not_allow_an_output_inside_original_inputs(tmp_path):
    module=driver();source=tmp_path/'source';source.mkdir()
    with pytest.raises(ValueError):module.check_output(source/'output',[source])
    assert not (source/'output').exists()


def test_driver_does_not_overwrite_a_completed_capture(tmp_path):
    module=driver();output=tmp_path/'output';output.mkdir()
    (output/'observation.png').write_bytes(b'prior capture')
    with pytest.raises(ValueError):module.check_output(output,[tmp_path/'source'])
    assert (output/'observation.png').read_bytes()==b'prior capture'

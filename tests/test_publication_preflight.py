"""Reject invalid publication targets before benchmark acquisition or resume."""

import importlib

import pytest


@pytest.mark.parametrize('resume', [False, True])
@pytest.mark.parametrize('repo,run', [('owner/results', 'x' * 97), ('bad', 'run-a')])
def test_invalid_publication_target_precedes_acquisition_and_resume(tmp_path, monkeypatch, resume, repo, run):
    module = importlib.import_module('simready_bench.benchmark')

    def unexpected(*args, **kwargs):
        pytest.fail('Publication validation must precede acquisition and resume')

    monkeypatch.setattr(module, 'reference_manifest', unexpected)
    monkeypatch.setattr('simready_bench.resume.resume_benchmark', unexpected)
    output = tmp_path / 'results'
    with pytest.raises(ValueError, match='dataset ID|run identifier'):
        module.benchmark(model='model', output_dir=output, hf_repo=repo, hf_run=run, resume=resume)
    assert not output.exists()

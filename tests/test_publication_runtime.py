import importlib

import pytest
from simready_bench.publishing import HuggingFacePublisher, PublicationError


@pytest.mark.parametrize("missing", ["matplotlib", "pyarrow.parquet"])
def test_publication_requires_rendering_dependencies_before_access_or_compute(monkeypatch, missing):
    original = importlib.import_module

    def load(name, *args, **kwargs):
        if name == missing:
            raise ModuleNotFoundError(name)
        if name in {"huggingface_hub", "matplotlib", "pyarrow.parquet"}:
            return object()
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", load)
    publisher = object.__new__(HuggingFacePublisher)
    with pytest.raises(PublicationError, match=r"simready-bench\[publish\]") as failure:
        publisher.check_access()
    assert failure.value.retryable is False

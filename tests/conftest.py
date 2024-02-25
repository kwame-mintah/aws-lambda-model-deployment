import pytest


@pytest.fixture(autouse=True)
def set_bucket_name(monkeypatch):
    monkeypatch.setattr(monkeypatch, "MODEL_OUTPUT_BUCKET_NAME", "unit-test")

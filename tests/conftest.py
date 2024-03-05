import pytest

import model_deployment


@pytest.fixture(autouse=True)
def set_bucket_name(monkeypatch):
    monkeypatch.setattr(model_deployment, "MODEL_OUTPUT_BUCKET_NAME", "unit-test")

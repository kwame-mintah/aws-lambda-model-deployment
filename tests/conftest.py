import pytest

import model_deployment


@pytest.fixture(autouse=True)
def set_environment_name(monkeypatch):
    """
    Stub bucket name for unit tests
    :param monkeypatch:
    """
    monkeypatch.setattr(model_deployment, "SERVERLESS_ENVIRONMENT", "local")

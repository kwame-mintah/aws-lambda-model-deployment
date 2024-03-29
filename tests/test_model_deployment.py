import botocore
from botocore.stub import Stubber, ANY

import model_deployment
from example_responses import example_event
from model_deployment import (
    lambda_handler,
    create_sagemaker_model,
    create_endpoint_config,
    create_serverless_endpoint,
)

MODEL_ARN = "arn:aws:sagemaker::012345678901:model/model"
ROLE = "arn:aws:iam::012345678901:role/SageMakerRole"
S3_MODEL_ARN = "s3://bucket-name/training-name/output/model.tar.gz"
ENDPOINT_CONFIG_ARN = (
    "arn:aws:sagemaker::012345678901:endpoint-config/endpoint-config-name"
)
ENDPOINT_ARN = "arn:aws:sagemaker::012345678901:endpoint/endpoint-name"


def test_lambda_handler(monkeypatch):

    def model_created(name, image, model_data_url, execution_role_arn):
        """Stub creating model"""
        return "model_name", "model_arn"

    def endpoint_config_created(name, model_name, variant_name):
        """Stub creating endpoint config"""
        assert model_name == "model_name"
        return "endpoint_config_name", "endpoint_config"

    def endpoint_created(name, endpoint_config_name):
        """Stub endpoint created"""
        assert endpoint_config_name == "endpoint_config_name"
        return "serverless_endpoint"

    monkeypatch.setattr(model_deployment, "create_sagemaker_model", model_created)
    monkeypatch.setattr(
        model_deployment, "create_endpoint_config", endpoint_config_created
    )
    monkeypatch.setattr(
        model_deployment, "create_serverless_endpoint", endpoint_created
    )
    event = example_event()
    result = lambda_handler(event, None)
    assert result["Records"][0]["eventName"] == "ObjectCreated:Put"


def test_create_sagemaker_model(monkeypatch):
    sagemaker_client = botocore.session.get_session().create_client("sagemaker")
    stubber = Stubber(sagemaker_client)
    expected_params = {
        "Containers": [
            {"Environment": ANY, "Image": ANY, "Mode": ANY, "ModelDataUrl": ANY}
        ],
        "ExecutionRoleArn": ANY,
        "ModelName": ANY,
        "Tags": ANY,
    }
    stubber.add_response("create_model", {"ModelArn": MODEL_ARN}, expected_params)

    with stubber:
        model_name, result = create_sagemaker_model(
            name="name",
            image="xgboost",
            model_data_url=S3_MODEL_ARN,
            execution_role_arn=ROLE,
            boto_client=sagemaker_client,
        )
        assert result == "arn:aws:sagemaker::012345678901:model/model"
        assert "name-serverless" in model_name


def test_create_endpoint_config(monkeypatch):
    sagemaker_client = botocore.session.get_session().create_client("sagemaker")
    stubber = Stubber(sagemaker_client)
    expected_params = {
        "EndpointConfigName": ANY,
        "ProductionVariants": [
            {
                "ModelName": ANY,
                "ServerlessConfig": {"MaxConcurrency": 1, "MemorySizeInMB": 4096},
                "VariantName": ANY,
            }
        ],
        "Tags": ANY,
    }
    stubber.add_response(
        "create_endpoint_config",
        {"EndpointConfigArn": ENDPOINT_CONFIG_ARN},
        expected_params,
    )

    with stubber:
        config_name, result = create_endpoint_config(
            name="name",
            model_name="model_name",
            variant_name="variant_name",
            boto_client=sagemaker_client,
        )
        assert (
            result
            == "arn:aws:sagemaker::012345678901:endpoint-config/endpoint-config-name"
        )
        assert "name-serverless" in config_name


def test_create_serverless_endpoint(monkeypatch):
    sagemaker_client = botocore.session.get_session().create_client("sagemaker")
    stubber = Stubber(sagemaker_client)
    expected_params = {"EndpointConfigName": ANY, "EndpointName": ANY, "Tags": ANY}
    stubber.add_response(
        "create_endpoint", {"EndpointArn": ENDPOINT_ARN}, expected_params
    )

    with stubber:
        serverless_endpoint_name = create_serverless_endpoint(
            name="name",
            endpoint_config_name="name-serverless",
            boto_client=sagemaker_client,
        )
        assert (
            serverless_endpoint_name
            == "arn:aws:sagemaker::012345678901:endpoint/endpoint-name"
        )

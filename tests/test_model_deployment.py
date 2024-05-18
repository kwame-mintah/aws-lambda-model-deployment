import botocore
from botocore.stub import Stubber, ANY

import model_deployment
from example_responses import (
    example_event,
    example_queue_url,
    example_send_message,
    example_sagemaker_list_tags,
)
from model_deployment import (
    lambda_handler,
    create_sagemaker_model,
    create_serverless_endpoint_config,
    create_serverless_endpoint,
    trigger_model_evaluation,
    get_training_job_test_data_location,
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
        """
        Stub creating model
        """
        return "model_name", "model_arn"

    def endpoint_config_created(name, model_name, variant_name):
        """
        Stub creating endpoint config
        """
        assert model_name == "model_name"
        return "endpoint_config_name", "endpoint_config"

    def endpoint_created(name, endpoint_config_name):
        """
        Stub endpoint created
        """
        assert endpoint_config_name == "endpoint_config_name"
        return "serverless_endpoint", "serverless_endpoint_name"

    def ssm_value(name):
        """
        Stub parameter store retrieval
        """
        return "value"

    def message_model_evaluation(
        endpoint_name, test_data_s3_bucket_name, test_data_s3_key, queue_name
    ):
        """
        Stub parameter store retrieval
        """
        return None

    monkeypatch.setattr(model_deployment, "create_sagemaker_model", model_created)
    monkeypatch.setattr(
        model_deployment, "create_serverless_endpoint_config", endpoint_config_created
    )
    monkeypatch.setattr(
        model_deployment, "create_serverless_endpoint", endpoint_created
    )
    monkeypatch.setattr(model_deployment, "get_parameter_store_value", ssm_value)
    monkeypatch.setattr(
        model_deployment, "trigger_model_evaluation", message_model_evaluation
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
        config_name, result = create_serverless_endpoint_config(
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
        serverless_endpoint_arn, endpoint_name = create_serverless_endpoint(
            name="name",
            endpoint_config_name="name-serverless",
            boto_client=sagemaker_client,
        )
        assert (
            serverless_endpoint_arn
            == "arn:aws:sagemaker::012345678901:endpoint/endpoint-name"
        )
        assert "name-serverless-ep-" in endpoint_name


def test_get_training_job_test_data_location():
    sagemaker_client = botocore.session.get_session().create_client("sagemaker")
    stubber = Stubber(sagemaker_client)
    expected_params_list_tags = {"ResourceArn": ANY}

    stubber.add_response(
        "list_tags", example_sagemaker_list_tags(), expected_params_list_tags
    )

    with stubber:
        test_data_s3_bucket_name, test_data_s3_key = (
            get_training_job_test_data_location(
                model_output_location="2024-04-22/output/xgboost-2024-04-22-20-51-18-610/output/model.tar.gz",
                boto_client=sagemaker_client,
            )
        )
        assert test_data_s3_bucket_name == "bucket-name"
        assert (
            test_data_s3_key == "automl/2024-04-22/training/testing/test_21_51_18.csv"
        )


def test_trigger_model_evaluation():
    sqs_client = botocore.session.get_session().create_client("sqs")
    stubber = Stubber(sqs_client)
    expected_params_get_queue_url = {"QueueName": ANY}
    expected_params_send_message = {"QueueUrl": ANY, "MessageBody": ANY}

    stubber.add_response(
        "get_queue_url", example_queue_url(), expected_params_get_queue_url
    )
    stubber.add_response(
        "send_message", example_send_message(), expected_params_send_message
    )

    with stubber:
        assert (
            trigger_model_evaluation(
                endpoint_name="endpoint-name",
                test_data_s3_bucket_name="bucket-name",
                test_data_s3_key="s3://bucket-name/test.csv",
                queue_name="queue-name",
                boto_client=sqs_client,
            )
            is None
        )

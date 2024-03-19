import logging
import os
from time import strftime, gmtime
from typing import Any

import boto3
import sagemaker

from models import S3Record

# The AWS region
aws_region = os.environ.get("AWS_REGION", "eu-west-2")

# Configure SageMaker client
sagemaker_client = boto3.client(service_name="sagemaker", region_name=aws_region)

# Configure logging
logger = logging.getLogger("model-deployment")
logger.setLevel(logging.INFO)

# The model output bucket name
MODEL_OUTPUT_BUCKET_NAME = os.environ.get("MODEL_OUTPUT_BUCKET_NAME")

# The model monitoring bucket name
MODEL_MONITORING_BUCKET_NAME = os.environ.get("MODEL_MONITORING_BUCKET_NAME")

# The SageMakerExecutionRole ARN
# TODO: Rather than setting as environment var, retrieve from parameter store
sagemaker_role_arn = os.environ.get("SAGEMAKER_ROLE_ARN")


def lambda_handler(event, context):
    """
    Create serverless inference endpoint for new models.

    :param event: S3 event for `.tar.gz` object added.
    :param context:
    :return: event
    """
    s3_record = S3Record(event)
    logger.info(
        "Received event: %s on bucket: %s for object: %s",
        s3_record.event_name,
        s3_record.bucket_name,
        s3_record.object_key,
    )

    # Create the model using the model artifacts
    model_name, model_arn = create_sagemaker_model(
        name="xgboost",
        image="xgboost",
        model_data_url="s3://{}/{}".format(
            MODEL_OUTPUT_BUCKET_NAME, s3_record.object_key
        ),
        execution_role_arn=sagemaker_role_arn,
    )

    logger.info("Created Model Arn: " + model_arn)

    # Create endpoint configuration
    endpoint_config_name, endpoint_config = create_endpoint_config(
        name="xgboost", model_name=model_name, variant_name="mlops"
    )

    logger.info("Created endpoint config Arn: " + endpoint_config)

    # Create serverless endpoint
    serverless_endpoint = create_serverless_endpoint(
        name="xgboost", endpoint_config_name=endpoint_config_name
    )

    logger.info("Created serverless endpoint Arn: " + serverless_endpoint)

    return event


def create_sagemaker_model(
    name: str,
    image: str,
    model_data_url: str,
    execution_role_arn: str,
    image_version: str = "latest",
    boto_client: Any = sagemaker_client,
) -> tuple[str, str]:
    """
    Creates a model in SageMaker.

    :param name: The name of the new model.
    :param image: Location of inference code image
    :param model_data_url: Location of model artifacts, enter the Amazon S3 URI to your ML model.
    :param image_version: The framework or algorithm version.
    :param execution_role_arn: The IAM role that SageMaker can assume to access model artifacts
    :param boto_client: Client representing Amazon SageMaker Service
    :return: model name and model arn
    """

    # Create unique model name
    model_name = name + "-serverless-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    logger.info("Model name: %s", model_name)

    # Specify the algorithm container
    # https://sagemaker.readthedocs.io/en/stable/api/utility/image_uris.html#sagemaker.image_uris.retrieve
    # TODO: Get the image from the S3 file path
    container = sagemaker.image_uris.retrieve(image, aws_region, image_version)

    # The environment variables to set in the Docker container
    container_env_vars = {"SAGEMAKER_CONTAINER_LOG_LEVEL": "20"}

    # Create model in SageMaker, using model training output
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker/client/create_model.html
    create_model_response = boto_client.create_model(
        ModelName=model_name,
        Containers=[
            {
                "Image": container,
                "Mode": "SingleModel",
                "ModelDataUrl": model_data_url,
                "Environment": container_env_vars,
            }
        ],
        ExecutionRoleArn=execution_role_arn,
        Tags=[
            {"Key": "Project", "Value": "MLOps"},
            {"Key": "Region", "Value": aws_region},
        ],
    )

    return model_name, create_model_response["ModelArn"]


def create_endpoint_config(
    name: str,
    model_name: str,
    variant_name: str,
    memory_size_in_mb: int = 4096,
    max_concurrency: int = 1,
    boto_client: Any = sagemaker_client,
) -> tuple[str, str]:
    """
    Creates an endpoint configuration that SageMaker hosting services uses to deploy models.

    :param name: The name of the endpoint configuration.
    :param model_name: The name of the model to host.
    :param variant_name: The name of the production variant.
    :param memory_size_in_mb: The memory size of the serverless endpoint.
    :param max_concurrency: The maximum number of concurrent invocations.
    :param boto_client: Client representing Amazon SageMaker Service
    :return:
    """

    # Name for the endpoint configuration created
    endpoint_config_name = (
        name + "-serverless-epc-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    )

    # Specify either Input, Output, or both
    # Example - Use list comprehension to capture both Input and Output
    capture_modes = ["Input", "Output"]

    # Create endpoint config in SageMaker
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker/client/create_endpoint_config.html#SageMaker.Client.create_endpoint_config
    # https://sagemaker.readthedocs.io/en/stable/api/inference/model_monitor.html#sagemaker.model_monitor.data_capture_config.DataCaptureConfig
    endpoint_config_response = boto_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": variant_name,
                "ModelName": model_name,
                "ServerlessConfig": {
                    "MemorySizeInMB": memory_size_in_mb,
                    "MaxConcurrency": max_concurrency,
                },
            },
        ],
        DataCaptureConfig={
            # Whether data should be captured or not.
            "EnableCapture": True,
            # Sampling percentage. Choose an integer value between 0 and 100
            "InitialSamplingPercentage": 20,
            # The S3 URI containing the captured data
            "DestinationS3Uri": "s3://{}/{}/".format(
                MODEL_MONITORING_BUCKET_NAME, model_name
            ),
            "CaptureOptions": [
                {"CaptureMode": capture_mode} for capture_mode in capture_modes
            ],
        },
        Tags=[
            {"Key": "Project", "Value": "MLOps"},
            {"Key": "Region", "Value": aws_region},
        ],
    )

    return endpoint_config_name, endpoint_config_response["EndpointConfigArn"]


def create_serverless_endpoint(
    name: str, endpoint_config_name: str, boto_client: Any = sagemaker_client
) -> str:
    """
    Creates an endpoint using the endpoint configuration specified.

    :param name: The name of the endpoint must be unique within an AWS Region.
    :param endpoint_config_name: The name of an endpoint configuration.
    :param boto_client: Client representing Amazon SageMaker Service
    :return:
    """

    # The endpoint name
    endpoint_name = name + "-serverless-ep-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    # Create serverless endpoint
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker/client/create_endpoint.html#SageMaker.Client.create_endpoint
    created_endpoint_response = boto_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name,
        Tags=[
            {"Key": "Project", "Value": "MLOps"},
            {"Key": "Region", "Value": aws_region},
        ],
    )

    return created_endpoint_response["EndpointArn"]

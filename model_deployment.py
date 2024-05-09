import json
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

# The environment the lambda is currently deployed in
SERVERLESS_ENVIRONMENT = os.environ.get("SERVERLESS_ENVIRONMENT")

ssm_model_monitoring_bucket_name = (
    "mlops-eu-west-2-%s-model-monitoring" % SERVERLESS_ENVIRONMENT
)

ssm_model_queue_name = "mlops-eu-west-2-%s-model-monitoring" % SERVERLESS_ENVIRONMENT

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
        model_data_url="s3://{}/{}".format(s3_record.bucket_name, s3_record.object_key),
        execution_role_arn=sagemaker_role_arn,
    )

    logger.info("Created Model Arn: " + model_arn)

    # Get the S3 bucket for collecting model monitoring outputs
    monitoring_bucket_name = get_parameter_store_value(
        name=ssm_model_monitoring_bucket_name
    )

    # Create endpoint configuration
    endpoint_config_name, endpoint_config = create_endpoint_config(
        name="xgboost",
        model_name=model_name,
        variant_name="mlops",
        monitoring_bucket_name=monitoring_bucket_name,
    )

    logger.info("Created endpoint config Arn: " + endpoint_config)

    # Create serverless endpoint
    serverless_endpoint, serverless_endpoint_name = create_serverless_endpoint(
        name="xgboost", endpoint_config_name=endpoint_config_name
    )

    logger.info("Created serverless endpoint Arn: " + serverless_endpoint)

    # Send message to model evaluation queue
    trigger_model_evaluation(
        endpoint_name=serverless_endpoint_name,
        test_data_s3_bucket_name="",
        test_data_s3_key="",
        queue_name="",
    )

    logger.info("Message sent to model-evaluation for prediction(s)")

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
    :param image: Location of inference code image.
    :param model_data_url: Location of model artifacts, enter the Amazon S3 URI to your ML model.
    :param image_version: The framework or algorithm version.
    :param execution_role_arn: The IAM role that SageMaker can assume to access model artifacts.
    :param boto_client: Client representing Amazon SageMaker Service.
    :return: model name and model arn.
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
    monitoring_bucket_name: str,
    memory_size_in_mb: int = 4096,
    max_concurrency: int = 1,
    boto_client: Any = sagemaker_client,
) -> tuple[str, str]:
    """
    Creates an endpoint configuration that SageMaker hosting services uses to deploy models.

    :param name: The name of the endpoint configuration.
    :param model_name: The name of the model to host.
    :param variant_name: The name of the production variant.
    :param monitoring_bucket_name: The Amazon S3 location used to capture the data.
    :param memory_size_in_mb: The memory size of the serverless endpoint.
    :param max_concurrency: The maximum number of concurrent invocations.
    :param boto_client: Client representing Amazon SageMaker Service.
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
                monitoring_bucket_name, model_name
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
) -> tuple[str, str]:
    """
    Creates an endpoint using the endpoint configuration specified.

    :param name: The name of the endpoint must be unique within an AWS Region.
    :param endpoint_config_name: The name of an endpoint configuration.
    :param boto_client: Client representing Amazon SageMaker Service.
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

    return created_endpoint_response["EndpointArn"], endpoint_name


def get_training_job_test_data_location(
    model_output_location: str, boto_client: Any = sagemaker_client
) -> tuple[str, str]:
    """

    :param model_output_location:
    :param boto_client:
    """
    # Training jobs name must be unique within an Amazon Web Services Region
    # in an Amazon Web Services account. Because no `TrainingJobName` is provided.
    # A unique name is generated, due to `Estimator` config `base_job_name` not being provided resulting in
    # the estimator to  generate a default job name, based on the training image name and current timestamp:
    # https://github.com/kwame-mintah/aws-lambda-model-training/blob/c38e2ab32cda11ef2cf66f7d7289799be1bcad35/model_training.py#L229
    # So, we are able to replace the text to determine the training job name.
    regex_training_job_algorithm = re.sub(
        r"\d{4}-\d{2}-\d{2}/[A-Za-z0-9]+/", "", model_output_location
    ).removesuffix("/output/model.tar.gz")
    # Get a list of all the tags on the training job.
    tags = boto_client.list_tags(
        ResourceArn=f"arn:aws:sagemaker:eu-west-2:827284457226:training-job/{regex_training_job_algorithm}"
    )["Tags"]

    for tag in tags:
        if "Testing" in tag["Key"]:
            test_data_s3_key = str(tag["Value"])
            # To determine the S3 Bucket name, remove unnecessary object path found in `test_data_s3_key`
            test_data_s3_bucket_name = re.sub(
                pattern=r"/[A-Za-z]+/(\d+(-\d+)+)/[A-Za-z]+/[A-Za-z]+/([A-Za-z0-9]+(_[A-Za-z0-9]+)+)\.[A-Za-z]+",
                repl="",
                string=test_data_s3_key.removeprefix("s3://"),
            )
            logger.info(
                "Found Testing tag(s), will set test data key as: %s and test data bucket name: %s",
                test_data_s3_key,
                test_data_s3_bucket_name,
            )
            return test_data_s3_bucket_name, test_data_s3_key
    logger.warning(
        "Unable to find relevant tag(s) to determine test data location, empty values provided."
    )
    return "", ""


def trigger_model_evaluation(
    endpoint_name: str,
    test_data_s3_bucket_name: str,
    test_data_s3_key: str,
    queue_name: str,
    boto_client: Any = boto3.client("sqs", region_name=aws_region),
) -> None:
    """
    Send a message to model evaluation lambda, to invoke the endpoint via predictions.

    :param endpoint_name: Name of the Amazon SageMaker endpoint to which requests are sent.
    :param test_data_s3_bucket_name: The S3 bucket containing the test data.
    :param test_data_s3_key: The S3 object full path to the test data csv.
    :param queue_name: The URL of the Amazon SQS queue to which a message is sent.
    :param boto_client: Boto3 client for sqs.
    """
    queue_url = boto_client.get_queue_url(QueueName=queue_name)["QueueUrl"]
    message_body = {
        "endpointName": endpoint_name,
        "testDataS3BucketName": test_data_s3_bucket_name,
        "testDataS3Key": test_data_s3_key,
    }
    boto_client.send_message(QueueUrl=queue_url, MessageBody=json.dumps(message_body))


def get_parameter_store_value(
    name: str, client: Any = boto3.client("ssm", region_name=aws_region)
) -> str:
    """
    Get a parameter store value from AWS.

    :param name: The name or Amazon Resource Name (ARN) of the parameter that you want to query
    :param client: boto3 client configured to use ssm
    :return: value
    """
    logger.info("Retrieving %s from parameter store", name)
    return client.get_parameter(Name=name, WithDecryption=True)["Parameter"]["Value"]

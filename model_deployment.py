import logging
import os

from time import strftime, gmtime

import boto3
import sagemaker

from models import S3Record

# The AWS region
aws_region = os.environ.get("AWS_REGION", "eu-west-2")

# Configure SageMaker client and runtime
sagemaker_client = boto3.client(service_name="sagemaker", region_name=aws_region)

# Configure logging
logger = logging.getLogger("model-deployment")
logger.setLevel(logging.INFO)

# The model output bucket name
MODEL_OUTPUT_BUCKET_NAME = os.environ.get("MODEL_OUTPUT_BUCKET_NAME")

# The SageMakerExecutionRole ARN
# TODO: Rather than setting as environment var, retrieve from parameter store
sagemaker_role_arn = os.environ.get("SAGEMAKER_ROLE_ARN")


def lambda_handler(event, context):
    s3_record = S3Record(event)
    logger.info(
        "Received event: %s on bucket: %s for object: %s",
        s3_record.event_name,
        s3_record.bucket_name,
        s3_record.object_key,
    )
    # Create the model using the model artifacts
    model = create_sagemaker_model(
        name="xgboost",
        image="xgboost",
        model_data_url="s3://{}/{}".format(
            MODEL_OUTPUT_BUCKET_NAME, s3_record.object_key
        ),
        execution_role_arn=sagemaker_role_arn,
    )

    logger.info("Created Model Arn: " + model)

    # TODO: Endpoint configuration creation ...

    # TODO: Serverless endpoint creation ...

    return event


def create_sagemaker_model(
    name: str,
    image: str,
    model_data_url: str,
    execution_role_arn: str,
    image_version: str = "latest",
) -> str:
    """
    Creates a model in SageMaker.

    :param name: The name of the new model.
    :param image: Location of inference code image
    :param model_data_url: Location of model artifacts, enter the Amazon S3 URI to your ML model.
    :param image_version: The framework or algorithm version.
    :param execution_role_arn: The IAM role that SageMaker can assume to access model artifacts
    :return:
    """

    # Create unique model name
    model_name = name + "-serverless-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    logger.info("Model name: %s", model_name)

    # Specify the algorithm container
    # https://sagemaker.readthedocs.io/en/stable/api/utility/image_uris.html#sagemaker.image_uris.retrieve
    # TODO: Get the image from the S3 file path
    container = sagemaker.image_uris.retrieve(image, aws_region, image_version)

    # The environment variables to set in the Docker container.
    container_env_vars = {"SAGEMAKER_CONTAINER_LOG_LEVEL": "20"}

    # Create model in SageMaker, using model training output
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker/client/create_model.html
    create_model_response = sagemaker_client.create_model(
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

    return create_model_response["ModelArn"]

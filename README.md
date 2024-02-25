# AWS Lambda Model Deployment

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3121/)
[![🚧 Bump version](https://github.com/kwame-mintah/aws-lambda-model-deployment/actions/workflows/bump-repository-version.yml/badge.svg)](https://github.com/kwame-mintah/aws-lambda-model-deployment/actions/workflows/bump-repository-version.yml)
[![🚀 Push Docker image to AWS ECR](https://github.com/kwame-mintah/aws-lambda-model-deployment/actions/workflows/push-docker-image-to-aws-ecr.yml/badge.svg)](https://github.com/kwame-mintah/aws-lambda-model-deployment/actions/workflows/push-docker-image-to-aws-ecr.yml)
[![🧹 Run linter](https://github.com/kwame-mintah/aws-lambda-model-deployment/actions/workflows/run-python-linter.yml/badge.svg)](https://github.com/kwame-mintah/aws-lambda-model-deployment/actions/workflows/run-python-linter.yml)

A lambda to deploy a model via serverless after training has completed. The model artifacts will be used to create a new model,
endpoint configuration and serverless endpoint configuration within SageMaker.

This repository does not create other necessarily resources within AWS, this is created via Terraform found here [terraform-aws-machine-learning-pipeline](https://github.com/kwame-mintah/terraform-aws-machine-learning-pipeline).
For more details on the entire flow and how this lambda is deployed, see [aws-automlops-serverless-deployment](https://github.com/kwame-mintah/aws-automlops-serverless-deployment).

## Development

### Dependencies

- [Python](https://www.python.org/downloads/release/python-3121/)
- [Docker for Desktop](https://www.docker.com/products/docker-desktop/)
- [Amazon Web Services](https://aws.amazon.com/?nc2=h_lg)

## Usage

1. Build the docker image locally:

   ```commandline
   docker build --no-cache -t model_deployment:local .
   ```

2. Run the docker image built:

   ```commandline
   docker run --platform linux/amd64 -p 9000:8080 model_deployment:local
   ```

3. Send an event to the lambda via curl:
   ```commandline
   curl "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{<REPLACE_WITH_JSON_BELOW>}'
   ```
   ```json
   {
     "Records": [
       {
         "eventVersion": "2.0",
         "eventSource": "aws:s3",
         "awsRegion": "us-east-1",
         "eventTime": "1970-01-01T00:00:00.000Z",
         "eventName": "ObjectCreated:Put",
         "userIdentity": { "principalId": "EXAMPLE" },
         "requestParameters": { "sourceIPAddress": "127.0.0.1" },
         "responseElements": {
           "x-amz-request-id": "EXAMPLE123456789",
           "x-amz-id-2": "EXAMPLE123/5678abcdefghijklambdaisawesome/mnopqrstuvwxyzABCDEFGH"
         },
         "s3": {
           "s3SchemaVersion": "1.0",
           "configurationId": "testConfigRule",
           "bucket": {
             "name": "example-bucket",
             "ownerIdentity": { "principalId": "EXAMPLE" },
             "arn": "arn:aws:s3:::example-bucket"
           },
           "object": {
             "key": "2024-02-23/output/xgboost-2024-02-23-18-04-06-024/output/model.tar.gz",
             "size": 1024,
             "eTag": "0123456789abcdef0123456789abcdef",
             "sequencer": "0A1B2C3D4E5F678901"
           }
         }
       }
     ]
   }
   ```

## GitHub Action (CI/CD)

The GitHub Action "🚀 Push Docker image to AWS ECR" will checkout the repository and push a docker image to the chosen AWS ECR using
[configure-aws-credentials](https://github.com/aws-actions/configure-aws-credentials/tree/v4.0.1/) action. The following repository secrets need to be set:

| Secret             | Description                  |
| ------------------ | ---------------------------- |
| AWS_REGION         | The AWS Region.              |
| AWS_ACCOUNT_ID     | The AWS account ID.          |
| AWS_ECR_REPOSITORY | The AWS ECR repository name. |

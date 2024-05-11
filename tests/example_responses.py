def example_event():
    """
    Example event received when a new model has been created.
    :return:
    """
    return {
        "Records": [
            {
                "eventVersion": "2.0",
                "eventSource": "aws:s3",
                "awsRegion": "us-east-1",
                "eventTime": "1970-01-01T00:00:00.000Z",
                "eventName": "ObjectCreated:Put",
                "userIdentity": {"principalId": "EXAMPLE"},
                "requestParameters": {"sourceIPAddress": "127.0.0.1"},
                "responseElements": {
                    "x-amz-request-id": "EXAMPLE123456789",
                    "x-amz-id-2": "EXAMPLE123/5678abcdefghijklambdaisawesome/mnopqrstuvwxyzABCDEFGH",
                },
                "s3": {
                    "s3SchemaVersion": "1.0",
                    "configurationId": "testConfigRule",
                    "bucket": {
                        "name": "example-bucket",
                        "ownerIdentity": {"principalId": "EXAMPLE"},
                        "arn": "arn:aws:s3:::example-bucket",
                    },
                    "object": {
                        "key": "2024-02-23/output/xgboost-2024-02-23-18-04-06-024/output/model.tar.gz",
                        "size": 1024,
                        "eTag": "0123456789abcdef0123456789abcdef",
                        "sequencer": "0A1B2C3D4E5F678901",
                    },
                },
            }
        ]
    }


def example_send_message():
    """
    Example response after a message has been sent to a SQS.
    :return:
    """
    return {
        "MD5OfMessageBody": "string",
        "MD5OfMessageAttributes": "string",
        "MD5OfMessageSystemAttributes": "string",
        "MessageId": "string",
        "SequenceNumber": "string",
    }


def example_queue_url():
    """
    Example response after getting a queue url, using the queue name.
    :return:
    """
    return {"QueueUrl": "string"}


def example_sagemaker_list_tags():
    """
    Example response when getting list of tags on a SageMaker resource.
    :return:
    """
    return {
        "Tags": [
            {"Key": "string", "Value": "string"},
            {"Key": "string", "Value": "string"},
            {
                "Key": "Testing",
                "Value": "s3://bucket-name/automl/2024-04-22/training/testing/test_21_51_18.csv",
            },
        ],
        "NextToken": "string",
    }

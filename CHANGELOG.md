## 0.8.1 (2024-05-18)

### Refactor

- **ssm_model_monitoring_bucket_name**: remove reference to monitoring bucket

## 0.8.0 (2024-05-18)

### Feat

- **create_serverless_endpoint_config**: remove datacaptureconfig from endpoint config

## 0.7.2 (2024-05-14)

### Refactor

- **ssm_model_evaluation_queue_name**: use correct paramater store name

## 0.7.1 (2024-05-11)

### Refactor

- **get_training_job_test_data_location**: object key is not a full s3 uri path

## 0.7.0 (2024-05-11)

### Feat

- **sagemaker_role_arn**: get sagemaker arn from parameter store

## 0.6.0 (2024-05-09)

### Feat

- **get_training_job_test_data_location**: interpolate training job and test data location
- **get_parameter_store_value**: add function to get parameter store values

### Refactor

- **trigger_model_evaluation**: interpolate test data created during training split
- **lambda_handler**: no longer use env for bucket names

## 0.5.0 (2024-04-16)

### Feat

- **trigger_model_evaluation**: send message to model evaluation queue for predictions

## 0.4.0 (2024-04-07)

### Feat

- **pre-commit-config**: update black and commitizen versions

## 0.3.0 (2024-03-19)

### Feat

- **model_deployment**: enable model monitoring for endpoint configs

## 0.2.2 (2024-03-05)

### Refactor

- **model_deployment**: include boto client as arg for sagemaker calls

## 0.2.1 (2024-02-27)

### Refactor

- **create_serverless_endpoint**: add hypen inbetween name for endpoint created

## 0.2.0 (2024-02-26)

### Feat

- **create_serverless_endpoint**: new function added to create serverless endpoint
- **create_endpoint_config**: new function added to create endpoint confiuration in sagemaker

## 0.1.0 (2024-02-25)

### Feat

- **model_deployment**: create sagemaker model from training artifacts

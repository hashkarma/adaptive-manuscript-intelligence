# EC2 IAM Role for Manuscript Intelligence

Use an **EC2 instance role**. Do not place IAM-user access keys on the instance.

## Bedrock Mantle

The platform's Provider C uses the Amazon Bedrock Mantle OpenAI-compatible
endpoint and `aws-bedrock-token-generator` short-term tokens.

For the professor demo, the simplest AWS-managed inference policy for the EC2
role is:

`arn:aws:iam::aws:policy/AmazonBedrockMantleInferenceAccess`

AWS documents this policy as granting the Mantle inference operations,
including `bedrock-mantle:CallWithBearerToken` and inference access.

Relevant AWS documentation:

- https://docs.aws.amazon.com/bedrock/latest/userguide/inference.html
- https://docs.aws.amazon.com/aws-managed-policy/latest/reference/AmazonBedrockMantleInferenceAccess.html
- https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys-generate.html

The application uses the normal AWS credential-provider chain. On EC2, the
instance role is therefore used automatically; no long-lived access key is
required.

## Stage-6 asset bucket

If Stage-6 assets are stored in S3, add access to the specific bucket/prefix.
A minimal pattern is:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadManuscriptRuntimeAssets",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::YOUR-BUCKET/manuscript-intelligence/*"
    }
  ]
}
```

If deployment tooling needs to list the prefix, add `s3:ListBucket` on the
bucket with an appropriate prefix condition.

## Region

The validated Provider-C configuration uses `us-east-1`.

Set:

```bash
export AWS_REGION=us-east-1
export AWS_DEFAULT_REGION=us-east-1
```

## Security

- Do not commit AWS credentials.
- Do not copy the local `~/.aws` directory to EC2.
- Prefer an EC2 instance profile.
- Restrict the S3 resource ARN to the research-assets prefix.
- Move toward a narrower custom Bedrock policy after the professor demo.

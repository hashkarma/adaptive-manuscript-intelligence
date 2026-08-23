# AWS Deployment

AWS deployment must use the same repository and FastAPI application used
locally.

Target application entrypoint:

`backend.main:app`

EC2 should obtain source code from GitHub, create Linux virtual environments
locally, install dependencies, obtain required model/knowledge assets through
the approved asset strategy, and run the same application.

Do not commit or copy Mac virtual environments to EC2.
Do not place long-lived IAM-user credentials in this repository.
Use an EC2 IAM role for AWS permissions.

# AWS deployment: one Fargate task

The cheapest honest way to run the service on AWS for a demo, then delete it. A single ECS Fargate task in the
default VPC with a public IP runs three containers, which share `localhost`:

| Container | Image | Job |
|---|---|---|
| `db` | `pgvector/pgvector:pg17` | Postgres. Data lives only as long as the task; ingestion rebuilds it in minutes. |
| `ingest` | this repo | chunks and embeds the docs into `db`, then exits |
| `api` | this repo | starts after `ingest` succeeds; generates answers with Amazon Bedrock (Nova Micro) |

Only the address in `allowed_cidr` can reach port 8000. The task role may call `bedrock:InvokeModel` and nothing
else.

## What is deliberately not here

Each of these bills by the hour whether it is used or not, so none of them is created:

* a load balancer (~$18/month);
* a NAT gateway (~$35/month);
* an EKS control plane (~$73/month);
* an RDS instance.

App Runner would have been simpler, but it is closed to new customers since 2026-04-30.

## Cost (eu-north-1, approximate list prices; check the AWS pricing pages before applying)

| Item | Rate | Per hour | Per month if left on (730 h) |
|---|---|---|---|
| Fargate, 1 vCPU | ~$0.0405 per vCPU-hour | $0.040 | ~$30 |
| Fargate, 2 GB memory | ~$0.0044 per GB-hour | $0.009 | ~$6.5 |
| Public IPv4 address | $0.005 per hour | $0.005 | ~$3.6 |
| ECR storage, image ~1.5 GB | $0.10 per GB-month | – | ~$0.15 |
| CloudWatch Logs, 3-day retention | $0.57 per GB ingested | – | cents |
| Bedrock Nova Micro | ~$0.035 / $0.14 per 1M input/output tokens | – | ~$0.01 per 100 questions² |
| **Total** | | **~$0.055** | **~$40** |

² About 1,900 input and 45 output tokens per question, measured in the eval runs.

A demo session of two hours costs about **$0.11 plus Bedrock tokens**. The cost only becomes real if the stack is
left running, so **destroy it the same day**. Before applying, set an AWS Budget alert (e.g. $5) on the account.

## Deploy

Prerequisites: the AWS CLI logged in, Docker, Terraform ≥ 1.6, and Bedrock model access for Nova Micro enabled in
the console if the account requires it (eu-north-1).

```bash
cd infra/aws
terraform init
terraform apply -var "allowed_cidr=$(curl -s https://checkip.amazonaws.com)/32"

# build and push the image to the ECR repository Terraform created
REPO=$(terraform output -raw ecr_repository_url)
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin "${REPO%%/*}"
docker build -t "$REPO:latest" ../..
docker push "$REPO:latest"

# start the task now that the image exists, then find its public IP
aws ecs update-service --region eu-north-1 --cluster rag-evals-service --service rag-evals-service --force-new-deployment
terraform output -raw find_public_ip | sh
curl -s http://<ip>:8000/health
```

Startup takes a few minutes, because ingestion embeds the docs on 1 vCPU before the API starts.

## Pause or destroy

* **Pause compute only:** `terraform apply -var desired_count=0 -var allowed_cidr=...`. This stops the Fargate
  and public-IP charges but keeps the ECR image and IAM roles.
* **Delete everything:** run the command below.

  ```bash
  terraform destroy -var "allowed_cidr=0.0.0.0/32"
  ```

  * `force_delete` on the ECR repository means pushed images are deleted too.
  * The log group is managed by Terraform, so it is removed as well.
  * Afterwards, check the console for anything tagged `project = rag-evals-service`; there should be nothing left.

## What happened when it was applied (2026-11-21)

* **The free plan blocks Bedrock.** Every call returned "Operation not allowed", so the run used
  `-var llm_backend=extractive`.
* **The free-plan project is pinned to eu-north-1,** hence the default region.
* **The first tasks were OOM-killed** while embedding with batches of 64. Batches of 16 fit in 2 GB.
* **Startup took 36 minutes,** spent on embedding 6,363 chunks on 1 vCPU.
* **Docker on Windows could not store the ECR login** in the Windows credential manager. The push used a temporary
  `DOCKER_CONFIG` with no `credsStore`.
* **Destroy removed all 10 resources.**

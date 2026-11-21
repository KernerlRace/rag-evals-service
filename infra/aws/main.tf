# The cheapest honest AWS deployment of this service: ONE Fargate task in the default VPC with a public IP.
#
#   containers in the task (they share localhost):
#     db      pgvector/pgvector:pg17   - data lives only as long as the task (ingestion rebuilds it in minutes)
#     ingest  this repo's image        - chunks + embeds the docs into db, then exits
#     api     this repo's image        - starts after ingest succeeds; generation through Amazon Bedrock
#
# Deliberately absent, because each one bills per hour whether used or not: load balancer (~$18/month),
# NAT gateway (~$35/month), EKS control plane (~$73/month), RDS instance. App Runner is closed to new
# customers since 2026-04-30, so it is not an option for a new account.

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { project = "rag-evals-service", managed_by = "terraform" }
  }
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_caller_identity" "me" {}

resource "aws_ecr_repository" "rag" {
  name                 = "rag-evals-service"
  image_tag_mutability = "MUTABLE"
  force_delete         = true # so `terraform destroy` also removes pushed images
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_cloudwatch_log_group" "rag" {
  name              = "/ecs/rag-evals-service"
  retention_in_days = 3
}

# --- IAM: the execution role pulls images and writes logs; the task role is what the app itself may do. ---

data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "rag-ecs-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "task" {
  name               = "rag-ecs-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

# Least privilege: the app may only invoke models, nothing else in the account.
data "aws_iam_policy_document" "bedrock" {
  statement {
    actions = ["bedrock:InvokeModel"]
    resources = [
      "arn:aws:bedrock:*::foundation-model/*",
      "arn:aws:bedrock:*:${data.aws_caller_identity.me.account_id}:inference-profile/*",
    ]
  }
}

resource "aws_iam_role_policy" "bedrock" {
  name   = "invoke-bedrock"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.bedrock.json
}

# --- Network: only my IP can reach port 8000. ---

resource "aws_security_group" "rag" {
  name        = "rag-api"
  description = "rag-evals-service API, reachable from one IP"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "API from my IP"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }
  egress {
    description = "image pulls, Bedrock"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# --- ECS ---

resource "aws_ecs_cluster" "rag" {
  name = "rag-evals-service"
}

locals {
  image        = "${aws_ecr_repository.rag.repository_url}:${var.image_tag}"
  database_url = "postgresql://rag:rag@localhost:5432/rag" # only reachable inside the task
  log = {
    logDriver = "awslogs"
    options = {
      awslogs-group         = aws_cloudwatch_log_group.rag.name
      awslogs-region        = var.region
      awslogs-stream-prefix = "rag"
    }
  }
}

resource "aws_ecs_task_definition" "rag" {
  family                   = "rag-evals-service"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  container_definitions = jsonencode([
    {
      name      = "db"
      image     = "pgvector/pgvector:pg17"
      essential = true
      environment = [
        { name = "POSTGRES_USER", value = "rag" },
        { name = "POSTGRES_PASSWORD", value = "rag" },
        { name = "POSTGRES_DB", value = "rag" },
      ]
      healthCheck = {
        command  = ["CMD-SHELL", "pg_isready -U rag"]
        interval = 5
        retries  = 10
      }
      logConfiguration = local.log
    },
    {
      name             = "ingest"
      image            = local.image
      essential        = false # it is supposed to exit
      command          = ["python", "-m", "ragsvc.ingest", "--sizes", "256", "--postgres"]
      environment      = [{ name = "DATABASE_URL", value = local.database_url }]
      dependsOn        = [{ containerName = "db", condition = "HEALTHY" }]
      logConfiguration = local.log
    },
    {
      name         = "api"
      image        = local.image
      essential    = true
      portMappings = [{ containerPort = 8000, protocol = "tcp" }]
      environment = [
        { name = "RAG_STORE", value = "postgres" },
        { name = "DATABASE_URL", value = local.database_url },
        { name = "RAG_RETRIEVER", value = var.retriever },
        { name = "RAG_LLM_BACKEND", value = var.llm_backend },
        { name = "BEDROCK_MODEL_ID", value = var.bedrock_model_id },
        { name = "AWS_REGION", value = var.region },
        { name = "RAG_VARIANT", value = "aws" },
      ]
      dependsOn        = [{ containerName = "ingest", condition = "SUCCESS" }]
      logConfiguration = local.log
    },
  ])
}

resource "aws_ecs_service" "rag" {
  name            = "rag-evals-service"
  cluster         = aws_ecs_cluster.rag.id
  task_definition = aws_ecs_task_definition.rag.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.rag.id]
    assign_public_ip = true # no NAT gateway, no load balancer: the task's own public IP serves the API
  }
}

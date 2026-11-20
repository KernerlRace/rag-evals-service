variable "region" {
  description = "AWS region. eu-west-1 (Ireland): Fargate and Bedrock both available, cheaper than London."
  type        = string
  default     = "eu-west-1"
}

variable "allowed_cidr" {
  description = "The only address allowed to call the API, e.g. \"203.0.113.7/32\" (your IP from https://checkip.amazonaws.com)."
  type        = string
}

variable "image_tag" {
  type    = string
  default = "latest"
}

variable "bedrock_model_id" {
  description = "Any Bedrock text model that supports the Converse API. Nova Micro is the cheapest; the eu. prefix is the EU cross-region inference profile."
  type        = string
  default     = "eu.amazon.nova-micro-v1:0"
}

variable "retriever" {
  type    = string
  default = "hybrid"
}

variable "cpu" {
  description = "Fargate task vCPU units (1024 = 1 vCPU). Embedding the docs at startup is CPU-bound."
  type        = number
  default     = 1024
}

variable "memory" {
  description = "Fargate task memory in MiB: Postgres ~150 MiB + API with two ONNX models ~900 MiB, with headroom."
  type        = number
  default     = 2048
}

variable "desired_count" {
  description = "Set to 0 to stop paying for compute but keep everything else (ECR image, roles)."
  type        = number
  default     = 1
}

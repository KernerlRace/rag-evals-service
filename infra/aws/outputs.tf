output "ecr_repository_url" {
  value = aws_ecr_repository.rag.repository_url
}

output "find_public_ip" {
  description = "The task gets a new public IP on every start; this prints it."
  value       = <<-EOT
    aws ecs list-tasks --region ${var.region} --cluster ${aws_ecs_cluster.rag.name} --query "taskArns[0]" --output text |
    xargs -I{} aws ecs describe-tasks --region ${var.region} --cluster ${aws_ecs_cluster.rag.name} --tasks {} --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" --output text |
    xargs -I{} aws ec2 describe-network-interfaces --region ${var.region} --network-interface-ids {} --query "NetworkInterfaces[0].Association.PublicIp" --output text
  EOT
}

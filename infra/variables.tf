# -----------------------------------------------------------------------------
# General Variables
# -----------------------------------------------------------------------------
variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "vattenfall-mlops"
}

variable "resource_group_name" {
  description = "Name of the Azure Resource Group"
  type        = string
  default     = "rg-vattenfall-mlops"
}

variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "northeurope"
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default = {
    project     = "vattenfall-mlops"
    environment = "production"
    managed_by  = "terraform"
  }
}

# -----------------------------------------------------------------------------
# Service Principal Variables (Optional - for non-interactive auth)
# Set via environment variables: ARM_CLIENT_ID, ARM_CLIENT_SECRET, etc.
# Or uncomment and use terraform.tfvars
# -----------------------------------------------------------------------------
# variable "client_id" {
#   description = "Azure Service Principal Client ID"
#   type        = string
#   sensitive   = true
# }
#
# variable "client_secret" {
#   description = "Azure Service Principal Client Secret"
#   type        = string
#   sensitive   = true
# }
#
# variable "subscription_id" {
#   description = "Azure Subscription ID"
#   type        = string
#   sensitive   = true
# }
#
# variable "tenant_id" {
#   description = "Azure Tenant ID"
#   type        = string
#   sensitive   = true
# }

# -----------------------------------------------------------------------------
# Container Registry Variables
# -----------------------------------------------------------------------------
variable "acr_name" {
  description = "Name of the Azure Container Registry (must be globally unique, alphanumeric only)"
  type        = string
  default     = "acrvattenfall"
}

# -----------------------------------------------------------------------------
# Container App Variables
# -----------------------------------------------------------------------------
variable "image_name" {
  description = "Name of the container image"
  type        = string
  default     = "vattenfall-ml"
}

variable "image_tag" {
  description = "Tag of the container image to deploy"
  type        = string
  default     = "latest"
}

variable "container_cpu" {
  description = "CPU allocation for the container (in cores)"
  type        = number
  default     = 0.5
}

variable "container_memory" {
  description = "Memory allocation for the container"
  type        = string
  default     = "1Gi"
}

variable "min_replicas" {
  description = "Minimum number of container replicas"
  type        = number
  default     = 0
}

variable "max_replicas" {
  description = "Maximum number of container replicas"
  type        = number
  default     = 3
}

# -----------------------------------------------------------------------------
# Secrets (pass via environment or tfvars)
# -----------------------------------------------------------------------------
variable "fingrid_api_key" {
  description = "Fingrid API key for fetching electricity data"
  type        = string
  sensitive   = true
}

# Azure Container Apps Infrastructure for Vattenfall MLOps
# This Terraform configuration creates:
# - Resource Group
# - Azure Container Registry (ACR)
# - Container Apps Environment
# - Container App with the ML prediction service

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
  required_version = ">= 1.0"
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
  # Disable automatic provider registration - we manage it explicitly
  skip_provider_registration = true
}

# -----------------------------------------------------------------------------
# Resource Provider Registrations (required for Container Apps)
# These ensure the subscription has the necessary providers enabled.
# -----------------------------------------------------------------------------
resource "azurerm_resource_provider_registration" "app" {
  name = "Microsoft.App"
}

resource "azurerm_resource_provider_registration" "operational_insights" {
  name = "Microsoft.OperationalInsights"
}

resource "azurerm_resource_provider_registration" "container_registry" {
  name = "Microsoft.ContainerRegistry"
}

# -----------------------------------------------------------------------------
# Resource Group
# -----------------------------------------------------------------------------
resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.location

  tags = var.tags
}

# -----------------------------------------------------------------------------
# Azure Container Registry
# -----------------------------------------------------------------------------
resource "azurerm_container_registry" "main" {
  name                = var.acr_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = false  # Use managed identity instead

  tags = var.tags

  depends_on = [azurerm_resource_provider_registration.container_registry]
}

# -----------------------------------------------------------------------------
# Log Analytics Workspace (required for Container Apps)
# -----------------------------------------------------------------------------
resource "azurerm_log_analytics_workspace" "main" {
  name                = "${var.project_name}-logs"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "PerGB2018"
  retention_in_days   = 30

  tags = var.tags

  depends_on = [azurerm_resource_provider_registration.operational_insights]
}

# -----------------------------------------------------------------------------
# Container Apps Environment
# -----------------------------------------------------------------------------
resource "azurerm_container_app_environment" "main" {
  name                       = "${var.project_name}-env"
  resource_group_name        = azurerm_resource_group.main.name
  location                   = azurerm_resource_group.main.location
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id

  tags = var.tags

  depends_on = [azurerm_resource_provider_registration.app]
}

# -----------------------------------------------------------------------------
# Container App - ML Prediction Service
# -----------------------------------------------------------------------------
resource "azurerm_container_app" "ml_service" {
  name                         = "${var.project_name}-app"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"

  tags = var.tags

  # Managed Identity for secure ACR access
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_app.id]
  }

  # ACR access via managed identity (no credentials needed)
  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.container_app.id
  }

  # Secrets from Key Vault
  secret {
    name                = "fingrid-api-key"
    key_vault_secret_id = azurerm_key_vault_secret.fingrid_api_key.versionless_id
    identity            = azurerm_user_assigned_identity.container_app.id
  }

  # Container template
  template {
    min_replicas = var.min_replicas
    max_replicas = var.max_replicas

    container {
      name   = "ml-service"
      image  = "${azurerm_container_registry.main.login_server}/${var.image_name}:${var.image_tag}"
      cpu    = var.container_cpu
      memory = var.container_memory

      # Environment variables
      env {
        name        = "FINGRID_API_KEY"
        secret_name = "fingrid-api-key"
      }

      env {
        name  = "PORT"
        value = "8080"
      }

      # Liveness probe
      liveness_probe {
        path                    = "/health"
        port                    = 8080
        transport               = "HTTP"
        initial_delay           = 10
        interval_seconds        = 30
        failure_count_threshold = 3
      }

      # Readiness probe
      readiness_probe {
        path                    = "/health"
        port                    = 8080
        transport               = "HTTP"
        interval_seconds        = 10
        failure_count_threshold = 3
      }
    }

    # HTTP scaling rule
    http_scale_rule {
      name                = "http-scaling"
      concurrent_requests = 50
    }
  }

  # Ingress configuration (external access)
  ingress {
    external_enabled = true
    target_port      = 8080
    transport        = "http"

    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }
}

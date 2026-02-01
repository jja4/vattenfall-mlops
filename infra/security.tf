# Security Implementation for Vattenfall MLOps Azure Infrastructure
#
# This file adds:
# - Managed Identity for Container App
# - RBAC role assignments (least privilege)
# - ACR pull permissions without admin credentials
# - Azure Key Vault for secrets
# - Diagnostic settings for audit logging

# -----------------------------------------------------------------------------
# User Assigned Managed Identity for Container App
# -----------------------------------------------------------------------------
resource "azurerm_user_assigned_identity" "container_app" {
  name                = "${var.project_name}-identity"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  tags = var.tags
}

# -----------------------------------------------------------------------------
# RBAC: Grant Container App identity permission to pull from ACR
# -----------------------------------------------------------------------------
resource "azurerm_role_assignment" "acr_pull" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.container_app.principal_id
}

# -----------------------------------------------------------------------------
# Azure Key Vault for Secrets Management
# -----------------------------------------------------------------------------
data "azurerm_client_config" "current" {}

resource "azurerm_key_vault" "main" {
  name                       = "${var.project_name}-kv"
  resource_group_name        = azurerm_resource_group.main.name
  location                   = azurerm_resource_group.main.location
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  soft_delete_retention_days = 7
  purge_protection_enabled   = false

  # Network ACLs - allow Azure services
  network_acls {
    default_action = "Allow"  # Change to "Deny" in production with specific IP whitelist
    bypass         = "AzureServices"
  }

  tags = var.tags
}

# Access policy for the deploying principal (you/service principal)
resource "azurerm_key_vault_access_policy" "deployer" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = data.azurerm_client_config.current.object_id

  secret_permissions = [
    "Get",
    "List",
    "Set",
    "Delete",
    "Purge",
    "Recover"
  ]
}

# Access policy for Container App managed identity
resource "azurerm_key_vault_access_policy" "container_app" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = azurerm_user_assigned_identity.container_app.principal_id

  secret_permissions = [
    "Get",
    "List"
  ]
}

# Store Fingrid API Key in Key Vault
resource "azurerm_key_vault_secret" "fingrid_api_key" {
  name         = "fingrid-api-key"
  value        = var.fingrid_api_key
  key_vault_id = azurerm_key_vault.main.id

  depends_on = [azurerm_key_vault_access_policy.deployer]
}

# -----------------------------------------------------------------------------
# Diagnostic Settings for Audit Logging
# -----------------------------------------------------------------------------
resource "azurerm_monitor_diagnostic_setting" "acr" {
  name                       = "${var.project_name}-acr-diagnostics"
  target_resource_id         = azurerm_container_registry.main.id
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id

  enabled_log {
    category = "ContainerRegistryRepositoryEvents"
  }

  enabled_log {
    category = "ContainerRegistryLoginEvents"
  }

  metric {
    category = "AllMetrics"
    enabled  = true
  }
}

resource "azurerm_monitor_diagnostic_setting" "key_vault" {
  name                       = "${var.project_name}-kv-diagnostics"
  target_resource_id         = azurerm_key_vault.main.id
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id

  enabled_log {
    category = "AuditEvent"
  }

  metric {
    category = "AllMetrics"
    enabled  = true
  }
}

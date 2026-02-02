# -----------------------------------------------------------------------------
# Azure Storage Account for MLOps Pipeline Data
# 
# This storage account holds:
# - Raw data from Fingrid API (DLT destination)
# - DLT pipeline state (incremental cursors)
# - Feature-engineered datasets
# - Training artifacts
# -----------------------------------------------------------------------------

resource "azurerm_resource_provider_registration" "storage" {
  name = "Microsoft.Storage"
}

resource "azurerm_storage_account" "mlops_data" {
  name                     = "${replace(var.project_name, "-", "")}data"  # Must be alphanumeric, globally unique
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"  # Locally redundant - sufficient for non-critical ML data
  
  # Enable blob versioning for data recovery
  blob_properties {
    versioning_enabled = true
    
    delete_retention_policy {
      days = 7
    }
    
    container_delete_retention_policy {
      days = 7
    }
  }
  
  # Security settings
  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false
  
  tags = var.tags

  depends_on = [azurerm_resource_provider_registration.storage]
}

# -----------------------------------------------------------------------------
# Storage Containers
# -----------------------------------------------------------------------------

# Raw data from Fingrid API (DLT writes here)
resource "azurerm_storage_container" "raw_data" {
  name                  = "raw"
  storage_account_name  = azurerm_storage_account.mlops_data.name
  container_access_type = "private"
}

# Feature-engineered datasets
resource "azurerm_storage_container" "features" {
  name                  = "features"
  storage_account_name  = azurerm_storage_account.mlops_data.name
  container_access_type = "private"
}

# DLT pipeline state (incremental cursors, schemas)
resource "azurerm_storage_container" "dlt_state" {
  name                  = "dlt-state"
  storage_account_name  = azurerm_storage_account.mlops_data.name
  container_access_type = "private"
}

# -----------------------------------------------------------------------------
# RBAC: Grant Container App identity access to storage
# (for reading features and models at runtime)
# -----------------------------------------------------------------------------
resource "azurerm_role_assignment" "storage_blob_reader" {
  scope                = azurerm_storage_account.mlops_data.id
  role_definition_name = "Storage Blob Data Reader"
  principal_id         = azurerm_user_assigned_identity.container_app.principal_id
}

# -----------------------------------------------------------------------------
# Store Storage Connection String in Key Vault
# (for GitHub Actions and local development)
# -----------------------------------------------------------------------------
resource "azurerm_key_vault_secret" "storage_connection_string" {
  name         = "storage-connection-string"
  value        = azurerm_storage_account.mlops_data.primary_connection_string
  key_vault_id = azurerm_key_vault.main.id

  depends_on = [azurerm_key_vault_access_policy.deployer]
}

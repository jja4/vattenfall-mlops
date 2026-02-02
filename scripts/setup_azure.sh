#!/bin/bash
# =============================================================================
# Bootstrap Azure Infrastructure for Vattenfall MLOps
# =============================================================================
# Complete setup script that:
#   1. Creates Azure Service Principal for GitHub Actions
#   2. Runs Terraform to provision infrastructure (Storage, ACR, Container Apps)
#   3. Extracts all secrets needed for GitHub Actions
#
# Prerequisites:
#   - Azure CLI installed and logged in (az login)
#   - Terraform installed
#   - jq installed
#   - FINGRID_API_KEY and WANDB_API_KEY in .env file
#
# Usage:
#   ./scripts/setup_azure.sh                      # Full bootstrap
#   ./scripts/setup_azure.sh --skip-terraform     # SP only, skip infra
#   ./scripts/setup_azure.sh --set-github-secrets # Also configure GitHub
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INFRA_DIR="$PROJECT_ROOT/infra"

SP_NAME="vattenfall-mlops-github"
REPO="your-github-username/vattenfall-mlops"  # Update this to your repo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
SKIP_TERRAFORM=false
SET_GITHUB_SECRETS=false
for arg in "$@"; do
    case $arg in
        --skip-terraform) SKIP_TERRAFORM=true ;;
        --set-github-secrets) SET_GITHUB_SECRETS=true ;;
    esac
done

echo "=============================================="
echo "🚀 Vattenfall MLOps - Azure Bootstrap"
echo "=============================================="

# -----------------------------------------------------------------------------
# Step 0: Prerequisites Check
# -----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}📋 Checking prerequisites...${NC}"

# Check Azure CLI
if ! command -v az &>/dev/null; then
    echo -e "${RED}❌ Azure CLI not installed${NC}"
    echo "Install: brew install azure-cli"
    exit 1
fi

# Check Terraform
if ! command -v terraform &>/dev/null; then
    echo -e "${RED}❌ Terraform not installed${NC}"
    echo "Install: brew install terraform"
    exit 1
fi

# Check jq
if ! command -v jq &>/dev/null; then
    echo -e "${RED}❌ jq not installed${NC}"
    echo "Install: brew install jq"
    exit 1
fi

# Check Azure login
if ! az account show &>/dev/null; then
    echo -e "${RED}❌ Not logged in to Azure CLI${NC}"
    echo "Run: az login"
    exit 1
fi

# Check for API keys in .env
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
fi

if [ -z "$FINGRID_API_KEY" ]; then
    echo -e "${RED}❌ FINGRID_API_KEY not set${NC}"
    echo "Add to .env file: FINGRID_API_KEY=your_key"
    exit 1
fi

if [ -z "$WANDB_API_KEY" ]; then
    echo -e "${RED}❌ WANDB_API_KEY not set${NC}"
    echo "Add to .env file: WANDB_API_KEY=your_key"
    exit 1
fi

echo -e "${GREEN}✅ All prerequisites met${NC}"

# Get subscription info
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
TENANT_ID=$(az account show --query tenantId -o tsv)
SUBSCRIPTION_NAME=$(az account show --query name -o tsv)

echo ""
echo "📋 Azure Context:"
echo "   Subscription: $SUBSCRIPTION_NAME"
echo "   Subscription ID: $SUBSCRIPTION_ID"
echo "   Tenant ID: $TENANT_ID"

# -----------------------------------------------------------------------------
# Step 1: Create Service Principal
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo -e "${BLUE}🔐 Step 1: Service Principal${NC}"
echo "=============================================="

# Check if SP already exists
EXISTING_SP=$(az ad sp list --display-name "$SP_NAME" --query "[0].appId" -o tsv 2>/dev/null || echo "")

if [ -n "$EXISTING_SP" ]; then
    echo -e "${YELLOW}⚠️  Service Principal '$SP_NAME' already exists${NC}"
    echo "   App ID: $EXISTING_SP"
    echo ""
    
    # Try to load existing secret from .env.azure
    if [ -f "$PROJECT_ROOT/.env.azure" ]; then
        EXISTING_SECRET=$(grep "^ARM_CLIENT_SECRET=" "$PROJECT_ROOT/.env.azure" | cut -d'=' -f2-)
        if [ -n "$EXISTING_SECRET" ] && [ "$EXISTING_SECRET" != "<existing - check .env.azure or create new>" ]; then
            echo -e "${GREEN}✅ Found existing secret in .env.azure${NC}"
            echo ""
            read -p "Use existing SP with saved secret? (Y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                CLIENT_ID="$EXISTING_SP"
                CLIENT_SECRET="$EXISTING_SECRET"
                echo -e "${GREEN}✅ Using existing SP and secret${NC}"
            fi
        fi
    fi
    
    # If we didn't load a valid secret, offer to recreate
    if [ -z "$CLIENT_SECRET" ]; then
        echo "No valid secret found in .env.azure"
        echo ""
        read -p "Delete and recreate SP with new secret? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Deleting existing Service Principal..."
            az ad sp delete --id "$EXISTING_SP"
            sleep 5  # Wait for deletion to propagate
        else
            echo -e "${RED}❌ Cannot proceed without valid SP credentials${NC}"
            echo "Either:"
            echo "  1. Rerun and choose to recreate the SP"
            echo "  2. Delete manually: az ad sp delete --id $EXISTING_SP"
            exit 1
        fi
    fi
fi

# Create new SP if needed
if [ -z "$CLIENT_ID" ]; then
    echo "🔧 Creating Service Principal: $SP_NAME"
    SP_OUTPUT=$(az ad sp create-for-rbac \
        --name "$SP_NAME" \
        --role contributor \
        --scopes "/subscriptions/$SUBSCRIPTION_ID" \
        --query "{clientId:appId, clientSecret:password}" \
        -o json)

    CLIENT_ID=$(echo "$SP_OUTPUT" | jq -r '.clientId')
    CLIENT_SECRET=$(echo "$SP_OUTPUT" | jq -r '.clientSecret')
    echo -e "${GREEN}✅ Service Principal created: $CLIENT_ID${NC}"
fi

# -----------------------------------------------------------------------------
# Step 2: Run Terraform
# -----------------------------------------------------------------------------
if [ "$SKIP_TERRAFORM" = false ]; then
    echo ""
    echo "=============================================="
    echo -e "${BLUE}🏗️  Step 3: Terraform Infrastructure${NC}"
    echo "=============================================="
    
    cd "$INFRA_DIR"
    
    # Export ARM credentials for Terraform
    export ARM_CLIENT_ID="$CLIENT_ID"
    export ARM_CLIENT_SECRET="$CLIENT_SECRET"
    export ARM_SUBSCRIPTION_ID="$SUBSCRIPTION_ID"
    export ARM_TENANT_ID="$TENANT_ID"
    
    # Terraform variables
    export TF_VAR_fingrid_api_key="$FINGRID_API_KEY"
    export TF_VAR_wandb_api_key="$WANDB_API_KEY"
    
    echo "📦 Terraform init..."
    terraform init -upgrade
    
    echo ""
    echo "🔍 Checking if Key Vault access needs updating..."
    # First, try to apply just the access policy to grant permissions
    # This allows Terraform to read secrets in subsequent operations
    terraform apply -target=azurerm_key_vault_access_policy.deployer -auto-approve \
        -var="use_wandb_registry=true" \
        -var="fingrid_api_key=$FINGRID_API_KEY" \
        -var="wandb_api_key=$WANDB_API_KEY" 2>&1 | grep -v "^data\." || true
    
    echo ""
    echo "🔍 Terraform plan (full infrastructure)..."
    terraform plan -out=tfplan -var="use_wandb_registry=true"
    
    echo ""
    read -p "Apply Terraform changes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🚀 Applying Terraform..."
        terraform apply tfplan
        rm -f tfplan
        
        # Extract outputs
        echo ""
        echo "📤 Extracting Terraform outputs..."
        STORAGE_CONN_STRING=$(az keyvault secret show \
            --vault-name "$(terraform output -raw key_vault_uri | sed 's|https://||' | sed 's|.vault.*||')" \
            --name "storage-connection-string" \
            --query value -o tsv 2>/dev/null || echo "")
        
        if [ -z "$STORAGE_CONN_STRING" ]; then
            # Fallback: get directly from storage account
            STORAGE_NAME=$(terraform output -raw storage_account_name)
            STORAGE_CONN_STRING=$(az storage account show-connection-string \
                --name "$STORAGE_NAME" \
                --resource-group "$(terraform output -raw resource_group_name)" \
                --query connectionString -o tsv)
        fi
        
        ACR_LOGIN_SERVER=$(terraform output -raw acr_login_server)
        APP_URL=$(terraform output -raw container_app_url 2>/dev/null || echo "pending")
        
        echo -e "${GREEN}✅ Infrastructure deployed${NC}"
    else
        echo -e "${YELLOW}⚠️  Terraform skipped${NC}"
        STORAGE_CONN_STRING="${AZURE_STORAGE_CONNECTION_STRING:-<run terraform to get>}"
    fi
    
    cd "$PROJECT_ROOT"
else
    echo ""
    echo -e "${YELLOW}⏭️  Skipping Terraform (--skip-terraform)${NC}"
    STORAGE_CONN_STRING="${AZURE_STORAGE_CONNECTION_STRING:-<run terraform to get>}"
fi

# -----------------------------------------------------------------------------
# Step 3: Output All Secrets
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo -e "${BLUE}📝 Step 3: GitHub Secrets${NC}"
echo "=============================================="
echo ""
echo "Go to: https://github.com/$REPO/settings/secrets/actions"
echo ""
echo "Add these 7 secrets (full values in .env.azure):"
echo ""
echo -e "${GREEN}ARM_CLIENT_ID${NC}"
echo "$CLIENT_ID"
echo ""
echo -e "${GREEN}ARM_CLIENT_SECRET${NC}"
echo "$CLIENT_SECRET"
echo ""
echo -e "${GREEN}ARM_SUBSCRIPTION_ID${NC}"
echo "$SUBSCRIPTION_ID"
echo ""
echo -e "${GREEN}ARM_TENANT_ID${NC}"
echo "$TENANT_ID"
echo ""
echo -e "${GREEN}FINGRID_API_KEY${NC}"
echo "$FINGRID_API_KEY"
echo ""
echo -e "${GREEN}WANDB_API_KEY${NC}"
echo "$WANDB_API_KEY"
echo ""
echo -e "${GREEN}AZURE_STORAGE_CONNECTION_STRING${NC}"
echo "$STORAGE_CONN_STRING"

# Save all secrets to .env.azure
ENV_FILE="$PROJECT_ROOT/.env.azure"
cat > "$ENV_FILE" << EOF
# =============================================================================
# Azure Secrets for GitHub Actions
# =============================================================================
# Created: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# Service Principal: $SP_NAME
#
# ⚠️  DO NOT COMMIT THIS FILE
#
# To set all GitHub secrets at once (requires gh CLI):
#   ./scripts/setup_azure.sh --set-github-secrets
# =============================================================================

# Azure Service Principal (for Terraform/deployment)
ARM_CLIENT_ID=$CLIENT_ID
ARM_CLIENT_SECRET=$CLIENT_SECRET
ARM_SUBSCRIPTION_ID=$SUBSCRIPTION_ID
ARM_TENANT_ID=$TENANT_ID

# External API Keys
FINGRID_API_KEY=$FINGRID_API_KEY
WANDB_API_KEY=$WANDB_API_KEY

# Azure Storage (from Terraform)
AZURE_STORAGE_CONNECTION_STRING=$STORAGE_CONN_STRING
EOF

echo ""
echo -e "${GREEN}💾 All secrets saved to .env.azure${NC}"

# -----------------------------------------------------------------------------
# Step 4: Optionally set GitHub secrets
# -----------------------------------------------------------------------------
if [ "$SET_GITHUB_SECRETS" = true ]; then
    echo ""
    echo "=============================================="
    echo -e "${BLUE}🔐 Step 4: Setting GitHub Secrets${NC}"
    echo "=============================================="
    
    if ! command -v gh &>/dev/null; then
        echo -e "${YELLOW}⚠️  GitHub CLI not installed. Install with: brew install gh${NC}"
        echo "Then run: ./scripts/setup_azure.sh --set-github-secrets"
    else
        if ! gh auth status &>/dev/null; then
            echo -e "${YELLOW}⚠️  Not logged in to GitHub CLI${NC}"
            echo "Run: gh auth login"
        else
            echo "Setting secrets for $REPO..."
            gh secret set ARM_CLIENT_ID --body "$CLIENT_ID" --repo "$REPO"
            gh secret set ARM_CLIENT_SECRET --body "$CLIENT_SECRET" --repo "$REPO"
            gh secret set ARM_SUBSCRIPTION_ID --body "$SUBSCRIPTION_ID" --repo "$REPO"
            gh secret set ARM_TENANT_ID --body "$TENANT_ID" --repo "$REPO"
            gh secret set FINGRID_API_KEY --body "$FINGRID_API_KEY" --repo "$REPO"
            gh secret set WANDB_API_KEY --body "$WANDB_API_KEY" --repo "$REPO"
            gh secret set AZURE_STORAGE_CONNECTION_STRING --body "$STORAGE_CONN_STRING" --repo "$REPO"
            echo -e "${GREEN}✅ All 7 GitHub secrets configured!${NC}"
        fi
    fi
fi

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo -e "${GREEN}🎉 Bootstrap Complete${NC}"
echo "=============================================="
echo ""
echo "Next steps:"
if [ "$SET_GITHUB_SECRETS" = false ]; then
    echo "1. Add secrets to GitHub: https://github.com/$REPO/settings/secrets/actions"
    echo "   Or run: ./scripts/setup_azure.sh --set-github-secrets"
    echo "2. Push to main branch to trigger deployment"
else
    echo "1. Push to main branch to trigger deployment"
fi
echo ""
if [ -n "$APP_URL" ] && [ "$APP_URL" != "pending" ]; then
    echo "🌐 Your app will be available at: $APP_URL"
fi
echo ""

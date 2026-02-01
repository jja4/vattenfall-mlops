# Vattenfall MLOps

ML Service for predicting Finland's electricity imbalance prices using real-time Fingrid API data.

## Live Service

**Production URL**: https://vattenfall-mlops-app.niceriver-dbaaae1a.northeurope.azurecontainerapps.io

- `/health` - Health check endpoint
- `/predict` - Real-time price prediction
- `/dashboard` - Interactive data visualization

## Structure
- `app/`: FastAPI application
- `ingestion/`: Tools for fetching and storing Fingrid data
- `models/`: Training scripts and model artifacts
- `data/`: Local data storage (ignored by git)
- `infra/`: Terraform IaC for Azure deployment
- `scripts/`: Deployment and maintenance scripts

## Local Development

### Prerequisites
- [uv](https://docs.astral.sh/uv/) - Python package manager
- [Docker](https://www.docker.com/) - For containerization
- [Terraform](https://www.terraform.io/) - For infrastructure (optional)
- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/) - For deployment (optional)

### Setup
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Create .env file with required keys
cp .env.example .env  # Then edit with your values
```

### Running Locally
```bash
# Run the API
uv run uvicorn app.main:app --reload

# Train a new model
uv run python models/train.py

# Build and run Docker container
docker build -t vattenfall-ml .
docker run -p 8080:8080 --env-file .env vattenfall-ml
```

## Azure Deployment

### ⚠️ Security First
**Before deploying**, review [docs/SECURITY.md](docs/SECURITY.md) to set up:
- Service Principal for Terraform (instead of personal credentials)
- Managed Identity for Container App
- Azure Key Vault for secrets
- RBAC and audit logging

### Quick Start (Development)
```bash
# Login to Azure (for testing only - use service principal in production)
az login

# Deploy infrastructure
./scripts/setup-azure.sh
```

### Production Deployment
```bash
# Set service principal credentials
export ARM_CLIENT_ID="your-sp-client-id"
export ARM_CLIENT_SECRET="your-sp-secret"
export ARM_SUBSCRIPTION_ID="your-subscription-id"
export ARM_TENANT_ID="your-tenant-id"

# Deploy without az login
cd infra
terraform apply -var="fingrid_api_key=${FINGRID_API_KEY}"
```

### Update After Model Retraining
```bash
# Quick image update with timestamped tag
./scripts/update-image.sh

# Or full deployment
./scripts/deploy.sh
```

### Infrastructure as Code
All Azure resources are managed via Terraform in `infra/`:
- Resource Group
- Azure Container Registry (admin disabled, uses managed identity)
- Container Apps Environment
- Container App with auto-scaling
- User Assigned Managed Identity
- Azure Key Vault for secrets
- Log Analytics with audit logging

## Environment Variables
- `FINGRID_API_KEY` - Required for Fingrid API access
- `WANDB_API_KEY` - Optional for Weights & Biases logging

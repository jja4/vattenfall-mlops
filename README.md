# Vattenfall MLOps

Real-time electricity imbalance price prediction service for Finland, powered by machine learning and Fingrid API data.

## 🌐 Live Service

**Production URL**: https://vattenfall-mlops-app.niceriver-dbaaae1a.northeurope.azurecontainerapps.io

| Endpoint | Description |
|----------|-------------|
| [`/docs`](https://vattenfall-mlops-app.niceriver-dbaaae1a.northeurope.azurecontainerapps.io/docs) | FastAPI interactive API docs |
| [`/health`](https://vattenfall-mlops-app.niceriver-dbaaae1a.northeurope.azurecontainerapps.io/health) | Health check & model status |
| [`/predict`](https://vattenfall-mlops-app.niceriver-dbaaae1a.northeurope.azurecontainerapps.io/predict) | Real-time price prediction |
| [`/dashboard`](https://vattenfall-mlops-app.niceriver-dbaaae1a.northeurope.azurecontainerapps.io/dashboard) | Interactive data visualization |


---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              AZURE CLOUD (North Europe)                         │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                     Resource Group: rg-vattenfall-mlops                 │    │
│  │                                                                         │    │
│  │   ┌──────────────┐      ┌──────────────────────────────────────────┐    │    │
│  │   │   Azure      │      │     Container Apps Environment           │    │    │
│  │   │  Container   │      │  ┌────────────────────────────────────┐  │    │    │
│  │   │  Registry    │─────▶│  │     Container App (FastAPI)        │  │    │    │
│  │   │  (ACR)       │ pull │  │                                    │  │    │    │
│  │   │              │      │  │  ┌────────┐  ┌─────────┐  ┌─────┐  │  │    │    │
│  │   │  Images:     │      │  │  │/health │  │/predict │  │/dash│  │  │    │    │
│  │   │  vattenfall- │      │  │  └────────┘  └─────────┘  └─────┘  │  │    │    │
│  │   │  ml:latest   │      │  │                                    │  │    │    │
│  │   └──────────────┘      │  │  Auto-scaling: 0-3 replicas        │  │    │    │
│  │          ▲              │  │  CPU: 0.5 cores | Memory: 1Gi      │  │    │    │
│  │          │              │  └────────────────────────────────────┘  │    │    │
│  │   Managed Identity      └──────────────────────────────────────────┘    │    │
│  │   (AcrPull role)                         │                              │    │
│  │          │                               │ reads secrets                │    │
│  │          ▼                               ▼                              │    │
│  │   ┌──────────────┐              ┌──────────────┐                        │    │
│  │   │    User      │              │  Azure Key   │                        │    │
│  │   │  Assigned    │─────────────▶│    Vault     │                        │    │
│  │   │  Managed     │  Get Secret  │              │                        │    │
│  │   │  Identity    │              │  Secrets:    │                        │    │
│  │   └──────────────┘              │  - fingrid-  │                        │    │
│  │                                 │    api-key   │                        │    │
│  │   ┌──────────────┐              └──────────────┘                        │    │
│  │   │ Log Analytics│◀── Audit logs from ACR, Key Vault, Container App     │    │
│  │   │  Workspace   │                                                      │    │
│  │   └──────────────┘                                                      │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ HTTPS
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              EXTERNAL DATA SOURCE                               │
│                                                                                 │
│   ┌──────────────────────────────────────────────────────────────────────┐      │
│   │                         Fingrid Open Data API                        │      │
│   │                                                                      │      │
│   │   Dataset #181: Wind Power Production (MW)                           │      │
│   │   Dataset #342: mFRR Activation (MW)                                 │      │
│   │   Dataset #319: Imbalance Price (EUR/MWh) ◀── Target Variable        │      │
│   │                                                                      │      │
│   │   Resolution: 15-minute intervals | Coverage: Real-time + Historical │      │
│   └──────────────────────────────────────────────────────────────────────┘      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Ingestion**: Fetches real-time data from Fingrid API (wind power, mFRR activation, imbalance prices)
2. **Processing**: Resamples to 15-minute intervals, creates lag features, rolling means, hour-of-day encoding
3. **Prediction**: RandomForest model (42 features) predicts next-hour imbalance price
4. **Serving**: FastAPI returns prediction with auto-scaling based on load

### Security Model

- **Secrets in Key Vault**: API keys stored securely, accessed via managed identity
- **RBAC**: Least-privilege roles (AcrPull for container, Get/List for secrets)
- **Audit logging**: All access to ACR and Key Vault logged to Log Analytics

---

## 📡 API Usage

### Health Check

```bash
curl https://vattenfall-mlops-app.niceriver-dbaaae1a.northeurope.azurecontainerapps.io/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_features": 42,
  "timestamp": "2026-02-01T18:05:24.498798Z"
}
```

### Get Prediction

```bash
curl https://vattenfall-mlops-app.niceriver-dbaaae1a.northeurope.azurecontainerapps.io/predict
```

**Response:**
```json
{
  "predicted_price": 73.73,
  "unit": "EUR/MWh",
  "prediction_for": "2026-02-01T17:15:00Z",
  "data_timestamp": "2026-02-01T17:00:00Z",
  "model_version": "2026-02-01T14:46:00.629991"
}
```

### Interactive Dashboard

Open in browser: https://vattenfall-mlops-app.niceriver-dbaaae1a.northeurope.azurecontainerapps.io/dashboard

Features:
- Real-time price chart (last 5 hours)
- Wind power production overlay
- mFRR activation events
- Hourly pattern analysis

---

## 📁 Project Structure

```
vattenfall-mlops/
├── app/                    # FastAPI application
│   ├── main.py            # API endpoints (/health, /predict, /dashboard)
│   └── schemas.py         # Pydantic models
├── ingestion/             # Data pipeline
│   ├── client.py          # Fingrid API client
│   ├── processor.py       # Feature engineering
│   └── storage.py         # Parquet I/O helpers
├── models/                # ML artifacts
│   ├── train.py           # Training script (W&B integration)
│   └── model.pkl          # Serialized RandomForest model
├── infra/                 # Terraform IaC
│   ├── main.tf            # Azure resources
│   ├── security.tf        # Managed identity, Key Vault, RBAC
│   ├── variables.tf       # Configuration
│   └── outputs.tf         # Deployment outputs
├── docs/                  # Documentation
│   └── project_plan.md    # Development roadmap
├── Dockerfile             # Container image definition
├── pyproject.toml         # Python dependencies (uv)
└── .env                   # Local secrets (gitignored)
```

---

## 🚀 Azure Deployment

### Prerequisites

| Tool | Installation | Purpose |
|------|--------------|---------|
| Azure CLI | `brew install azure-cli` | Azure authentication |
| Terraform | `brew install terraform` | Infrastructure as Code |
| Docker | [docker.com](https://docker.com) | Container builds |

### Step 1: Initial Setup

```bash
# Clone repository
git clone <repo-url>
cd vattenfall-mlops

# Login to Azure
az login

# Set required environment variables
export FINGRID_API_KEY="your-fingrid-api-key"

# Deploy cloud resources
cd infra
terraform init
terraform apply -var="fingrid_api_key=$FINGRID_API_KEY"
```


## 💻 Local Development

### Setup

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Create environment file
cat > .env << EOF
FINGRID_API_KEY=your-api-key
WANDB_API_KEY=your-wandb-key  # Optional
EOF
```

### Run Locally

```bash
# Start FastAPI server
uv run uvicorn app.main:app --reload --port 8000

# Open in browser
open http://localhost:8000/dashboard
```

### Train Model

```bash
# Train with W&B logging
uv run python models/train.py

# Model saved to models/model.pkl
```

### Docker Testing

```bash
# Build for local testing
docker build -t vattenfall-ml .

# Run container
docker run -p 8080:8080 --env-file .env vattenfall-ml

# Test
curl http://localhost:8080/health
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FINGRID_API_KEY` | Yes | API key from [Fingrid Open Data](https://data.fingrid.fi/) |
| `WANDB_API_KEY` | No | Weights & Biases for experiment tracking |

### Terraform Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `project_name` | `vattenfall-mlops` | Resource naming prefix |
| `location` | `northeurope` | Azure region |
| `min_replicas` | `0` | Scale to zero when idle |
| `max_replicas` | `3` | Maximum container instances |
| `container_cpu` | `0.5` | CPU cores per instance |
| `container_memory` | `1Gi` | Memory per instance |

---

## 📊 Model Details

- **Algorithm**: RandomForestRegressor (scikit-learn)
- **Features**: 42 engineered features including:
  - Lag features (1h, 2h, 3h, 6h, 12h, 24h)
  - Rolling means (3h, 6h, 12h, 24h windows)
  - Hour-of-day cyclical encoding
  - Day-of-week indicators
- **Target**: Imbalance price (EUR/MWh)
- **Training Data**: 1 year of hourly Fingrid data (2025)
- **Experiment Tracking**: Weights & Biases

---

## 📚 Documentation

- [Project Plan](docs/project_plan.md) - Development roadmap and completed steps




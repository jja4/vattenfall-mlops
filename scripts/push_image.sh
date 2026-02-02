#!/bin/bash
# Deploy script for Vattenfall MLOps
# Builds, pushes, and deploys the latest code to Azure Container Apps

set -e  # Exit on error

# Get ACR server from Terraform
cd "$(dirname "$0")/../infra"
ACR=$(terraform output -raw acr_login_server)
ACR_NAME=${ACR%%.*}
cd ..

# Generate unique tag using git commit or timestamp
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "local")
TIMESTAMP=$(date +%Y%m%d%H%M%S)
IMAGE_TAG="${GIT_SHA}-${TIMESTAMP}"

echo "🔐 Logging into Azure Container Registry..."
az acr login --name $ACR_NAME

echo "🏗️  Building Docker image (linux/amd64)..."
docker build --no-cache --platform linux/amd64 -t $ACR/vattenfall-mlops:$IMAGE_TAG -t $ACR/vattenfall-mlops:latest .

echo "📤 Pushing image to ACR..."
docker push $ACR/vattenfall-mlops:$IMAGE_TAG
docker push $ACR/vattenfall-mlops:latest

echo "🚀 Deploying to Azure Container Apps..."
az containerapp update \
  -n vattenfall-mlops-app \
  -g rg-vattenfall-mlops \
  --image $ACR/vattenfall-mlops:$IMAGE_TAG

echo "✅ Deployment complete!"
echo "   Image: $ACR/vattenfall-mlops:$IMAGE_TAG"
echo "   URL: https://vattenfall-mlops-app.niceriver-dbaaae1a.northeurope.azurecontainerapps.io"
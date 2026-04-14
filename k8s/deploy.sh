#!/bin/bash
# ============================================================
# CMP Pipeline - Kubernetes Deployment Script
# Prerequisites: minikube, kubectl, helm
# ============================================================
set -e

echo "============================================"
echo " CMP Intelligent Pipeline - K8s Deployment"
echo "============================================"

# ----------------------------------------------------------
# Step 1: Start minikube (if not running)
# ----------------------------------------------------------
echo ""
echo "[1/6] Checking minikube status..."
if ! minikube status | grep -q "Running"; then
    echo "Starting minikube..."
    minikube start --cpus=4 --memory=8192 --driver=docker
else
    echo "minikube is already running."
fi

# ----------------------------------------------------------
# Step 2: Create namespaces
# ----------------------------------------------------------
echo ""
echo "[2/6] Creating namespaces..."
kubectl apply -f namespaces.yaml

# ----------------------------------------------------------
# Step 3: Create secrets
#   IMPORTANT: Edit secrets.yaml with real base64 values first!
# ----------------------------------------------------------
echo ""
echo "[3/6] Applying secrets..."
echo "  WARNING: Make sure you've edited secrets.yaml with real values!"
kubectl apply -f secrets.yaml

# ----------------------------------------------------------
# Step 4: Deploy PostgreSQL
# ----------------------------------------------------------
echo ""
echo "[4/6] Deploying PostgreSQL..."
kubectl apply -f postgres/statefulset.yaml
kubectl apply -f postgres/service.yaml
echo "  Waiting for PostgreSQL to be ready..."
kubectl wait --for=condition=ready pod/postgres-0 -n data --timeout=120s

# ----------------------------------------------------------
# Step 5: Deploy Airflow via Helm
# ----------------------------------------------------------
echo ""
echo "[5/6] Deploying Airflow via Helm..."
helm repo add apache-airflow https://airflow.apache.org 2>/dev/null || true
helm repo update

helm upgrade --install airflow apache-airflow/airflow \
    -n airflow \
    -f values-airflow.yaml \
    --timeout 10m

echo "  Waiting for Airflow webserver..."
kubectl wait --for=condition=ready pod -l component=webserver -n airflow --timeout=300s

# ----------------------------------------------------------
# Step 6: Deploy Superset
# ----------------------------------------------------------
echo ""
echo "[6/6] Deploying Superset..."
kubectl apply -f superset/deployment.yaml
kubectl apply -f superset/service.yaml

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
echo ""
echo "============================================"
echo " Deployment Complete!"
echo "============================================"
echo ""
echo " Airflow UI:"
echo "   minikube service airflow-webserver -n airflow"
echo "   Or: kubectl port-forward svc/airflow-webserver 8080:8080 -n airflow"
echo "   Login: admin / admin"
echo ""
echo " Superset UI:"
echo "   minikube service superset-svc -n data"
echo "   Or: kubectl port-forward svc/superset-svc 8088:8088 -n data"
echo "   Login: admin / admin"
echo ""
echo " Useful commands:"
echo "   kubectl get pods -n airflow    # Check Airflow pods"
echo "   kubectl get pods -n data       # Check DB + Superset pods"
echo "   kubectl logs -f <pod> -n airflow  # View logs"
echo "   minikube dashboard             # K8s Dashboard"
echo "============================================"

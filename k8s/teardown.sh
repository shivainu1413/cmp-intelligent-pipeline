#!/bin/bash
# ============================================================
# CMP Pipeline - Kubernetes Teardown Script
# Removes all K8s resources. PVCs are preserved by default.
# ============================================================
set -e

echo "Tearing down CMP Pipeline..."

echo "[1/4] Removing Airflow Helm release..."
helm uninstall airflow -n airflow 2>/dev/null || echo "  (not found, skipping)"

echo "[2/4] Removing Superset..."
kubectl delete -f superset/ -n data 2>/dev/null || echo "  (not found, skipping)"

echo "[3/4] Removing PostgreSQL..."
kubectl delete -f postgres/ -n data 2>/dev/null || echo "  (not found, skipping)"

echo "[4/4] Removing namespaces..."
kubectl delete -f namespaces.yaml 2>/dev/null || echo "  (not found, skipping)"

echo ""
echo "Done. Note: PVCs may still exist. To fully clean up:"
echo "  kubectl delete pvc --all -n data"
echo "  minikube stop && minikube delete  (nuclear option)"

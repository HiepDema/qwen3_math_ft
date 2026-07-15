#!/bin/bash
set -e

# ============================================================
# Deploy Math Reasoning API on Lambda Cloud (A10 GPU)
#
# Usage:
#   # Option 1: Docker Compose (recommended for single node)
#   bash scripts/deploy.sh compose
#
#   # Option 2: Kubernetes (for cluster)
#   bash scripts/deploy.sh k8s
#
#   # Option 3: Direct run (no Docker, simplest)
#   bash scripts/deploy.sh direct
#
#   # Run benchmark after deployment
#   bash scripts/deploy.sh benchmark
#
#   # Stop everything
#   bash scripts/deploy.sh stop
# ============================================================

SERVER_IP="${SERVER_IP:-0.0.0.0}"
MODEL_PATH="${MODEL_PATH:-hiep-2/qwen3-0.6b-math-cpt-sft}"
BACKEND="${BACKEND:-vllm}"

echo "============================================"
echo "  Math Reasoning API Deployment"
echo "============================================"
echo "  Model:   $MODEL_PATH"
echo "  Backend: $BACKEND"
echo "  Server:  $SERVER_IP"
echo "============================================"

deploy_direct() {
    echo ""
    echo "[1/3] Installing dependencies..."
    pip install fastapi uvicorn prometheus_client vllm transformers huggingface_hub

    echo ""
    echo "[2/3] Starting model server (background)..."
    nohup python scripts/serve_model_monitored.py \
        --model-path "$MODEL_PATH" \
        --backend "$BACKEND" \
        --host 0.0.0.0 \
        --port 8000 \
        > logs/serve.log 2>&1 &
    echo "  PID: $!"
    echo "  Log: logs/serve.log"

    echo ""
    echo "[3/3] Waiting for server to be ready..."
    mkdir -p logs
    for i in $(seq 1 60); do
        if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
            echo "  Server is ready!"
            curl -s http://localhost:8000/health | python -m json.tool
            echo ""
            echo "  API:     http://$SERVER_IP:8000"
            echo "  Health:  http://$SERVER_IP:8000/health"
            echo "  Metrics: http://$SERVER_IP:8000/metrics"
            echo ""
            echo "  Test: curl -X POST http://$SERVER_IP:8000/solve -H 'Content-Type: application/json' -d '{\"question\": \"What is 2+2?\"}'"
            return 0
        fi
        echo "  Waiting... ($i/60)"
        sleep 5
    done
    echo "  ERROR: Server did not start within 5 minutes"
    echo "  Check logs: tail -f logs/serve.log"
    return 1
}

deploy_compose() {
    echo ""
    echo "[1/2] Starting Docker Compose stack..."
    docker compose up -d --build

    echo ""
    echo "[2/2] Waiting for services..."
    echo "  Waiting for model to load (this may take 2-3 minutes)..."
    for i in $(seq 1 60); do
        if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
            echo "  All services ready!"
            echo ""
            echo "  API:        http://$SERVER_IP:8000"
            echo "  Prometheus: http://$SERVER_IP:9090"
            echo "  Grafana:    http://$SERVER_IP:3000 (admin/admin123)"
            echo "  Metrics:    http://$SERVER_IP:8000/metrics"
            echo ""
            echo "  Docker status:"
            docker compose ps
            return 0
        fi
        echo "  Waiting... ($i/60)"
        sleep 5
    done
    echo "  WARNING: API not responding yet, but containers may still be loading"
    docker compose ps
    docker compose logs math-api --tail 20
}

deploy_k8s() {
    echo ""
    echo "[1/4] Creating namespace..."
    kubectl apply -f k8s/namespace.yaml

    echo ""
    echo "[2/4] Deploying Prometheus + Grafana..."
    kubectl apply -f k8s/prometheus-config.yaml
    kubectl apply -f k8s/prometheus.yaml
    kubectl apply -f k8s/grafana.yaml

    echo ""
    echo "[3/4] Deploying Math Reasoning API..."
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml
    kubectl apply -f k8s/hpa.yaml

    echo ""
    echo "[4/4] Waiting for pods..."
    kubectl -n math-reasoning wait --for=condition=available deployment/math-reasoning-api --timeout=300s || true

    echo ""
    echo "  Pod status:"
    kubectl -n math-reasoning get pods
    echo ""
    echo "  Services:"
    kubectl -n math-reasoning get svc
    echo ""
    echo "  API:        http://$SERVER_IP:30080"
    echo "  Prometheus: http://$SERVER_IP:30090"
    echo "  Grafana:    http://$SERVER_IP:30030 (admin/admin123)"
}

run_benchmark() {
    URL="${1:-http://localhost:8000}"
    echo ""
    echo "Running benchmark against $URL..."
    mkdir -p outputs

    python scripts/benchmark_deployment.py \
        --url "$URL" \
        --num-requests 20 \
        --output outputs/benchmark_results.json

    echo ""
    echo "Results saved to outputs/benchmark_results.json"
}

run_monitoring_test() {
    echo ""
    echo "Testing monitoring endpoints..."

    echo ""
    echo "[1] Health check:"
    curl -s http://localhost:8000/health | python -m json.tool

    echo ""
    echo "[2] Prometheus metrics (sample):"
    curl -s http://localhost:8000/metrics | head -30

    echo ""
    echo "[3] Sending test requests to generate metrics..."
    for i in $(seq 1 5); do
        curl -s -X POST http://localhost:8000/solve \
            -H "Content-Type: application/json" \
            -d '{"question": "What is '"$i"' + '"$i"'?"}' > /dev/null
        echo "  Request $i sent"
    done

    echo ""
    echo "[4] Prometheus metrics after requests:"
    curl -s http://localhost:8000/metrics | grep -E "^(request_count|request_latency|model_loaded|gpu_memory)"

    echo ""
    echo "Monitoring test complete!"
    if curl -sf http://localhost:9090/-/healthy > /dev/null 2>&1; then
        echo "  Prometheus: http://localhost:9090 (healthy)"
    fi
    if curl -sf http://localhost:3000/api/health > /dev/null 2>&1; then
        echo "  Grafana:    http://localhost:3000 (healthy)"
    fi
}

stop_all() {
    echo ""
    echo "Stopping all services..."

    # Stop direct
    pkill -f "serve_model_monitored.py" 2>/dev/null && echo "  Stopped direct server" || true

    # Stop Docker Compose
    if command -v docker &>/dev/null; then
        docker compose down 2>/dev/null && echo "  Stopped Docker Compose stack" || true
    fi

    # Stop K8s
    if command -v kubectl &>/dev/null; then
        kubectl delete namespace math-reasoning 2>/dev/null && echo "  Deleted K8s namespace" || true
    fi

    echo "  Done!"
}

# ============================================================
# Main
# ============================================================
case "${1:-direct}" in
    direct)
        deploy_direct
        ;;
    compose)
        deploy_compose
        ;;
    k8s)
        deploy_k8s
        ;;
    benchmark)
        run_benchmark "${2:-http://localhost:8000}"
        ;;
    monitor-test)
        run_monitoring_test
        ;;
    stop)
        stop_all
        ;;
    *)
        echo "Usage: bash scripts/deploy.sh {direct|compose|k8s|benchmark|monitor-test|stop}"
        echo ""
        echo "  direct       - Run model server directly (no Docker)"
        echo "  compose      - Deploy with Docker Compose (API + Prometheus + Grafana)"
        echo "  k8s          - Deploy to Kubernetes cluster"
        echo "  benchmark    - Run performance benchmark"
        echo "  monitor-test - Test monitoring endpoints"
        echo "  stop         - Stop all services"
        exit 1
        ;;
esac

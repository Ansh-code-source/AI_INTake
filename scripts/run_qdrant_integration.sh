#!/usr/bin/env bash
set -euo pipefail

# Starts a local Qdrant container, runs the Qdrant integration test, and cleans up.
# Requires: docker installed and user can run docker commands.

CONTAINER_NAME=ai_intake_qdrant_dev
IMAGE=qdrant/qdrant:v1.2.0

docker run -d --name "$CONTAINER_NAME" -p 6333:6333 "$IMAGE"

# Wait for Qdrant health
for i in {1..30}; do
  if curl -sSf http://localhost:6333/health >/dev/null 2>&1; then
    echo "Qdrant healthy"
    break
  fi
  echo "Waiting for Qdrant... ($i)"
  sleep 1
done

python -m pip install --upgrade pip
pip install -e ".[dev]" langchain_qdrant langchain_openai qdrant-client openai

export RUN_QDRANT_INTEGRATION=1
pytest tests/test_qdrant_wrapper.py::test_qdrant_wrapper_from_documents_integration -q

# Clean up
docker rm -f "$CONTAINER_NAME" || true

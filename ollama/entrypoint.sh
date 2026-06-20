#!/bin/bash
# Start the Ollama server, then pull the chat (and optionally embedding) models on
# first boot so `docker compose up` is the only command a reviewer needs. Pulled models
# live on a named volume, so subsequent starts are instant and fully offline.
#
# Deliberately NOT using `set -e`: we want retries plus a controlled shutdown path that
# fails fast instead of hanging forever behind an unmet healthcheck.

ollama serve &
server_pid=$!

# Forward termination signals to the server for a graceful `docker compose down`.
trap 'kill -TERM "$server_pid" 2>/dev/null; wait "$server_pid"; exit 0' TERM INT

echo "Waiting for Ollama to be ready..."
until ollama list >/dev/null 2>&1; do
  sleep 1
done

pull_with_retry() {
  local model="$1"
  local attempt=1
  until ollama pull "$model"; do
    if [ "$attempt" -ge 5 ]; then
      echo "ERROR: failed to pull ${model} after ${attempt} attempts." >&2
      return 1
    fi
    echo "Pull of ${model} failed (attempt ${attempt}); retrying in 5s..." >&2
    attempt=$((attempt + 1))
    sleep 5
  done
}

CHAT_MODEL="${OLLAMA_MODEL:-llama3.2:3b}"
echo "Pulling chat model: ${CHAT_MODEL}"
if ! pull_with_retry "${CHAT_MODEL}"; then
  kill -TERM "${server_pid}" 2>/dev/null
  wait "${server_pid}" 2>/dev/null
  exit 1
fi

# Only needed if you switch embeddings to the Ollama backend (EMBED_BACKEND=ollama).
if [ "${PULL_EMBED_MODEL:-false}" = "true" ]; then
  EMBED_MODEL_NAME="${EMBED_MODEL:-nomic-embed-text}"
  echo "Pulling embedding model: ${EMBED_MODEL_NAME}"
  if ! pull_with_retry "${EMBED_MODEL_NAME}"; then
    kill -TERM "${server_pid}" 2>/dev/null
    wait "${server_pid}" 2>/dev/null
    exit 1
  fi
fi

echo "Models ready."
wait "${server_pid}"

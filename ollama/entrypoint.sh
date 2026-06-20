#!/bin/bash
# Start the Ollama server, then pull the chat (and optionally embedding) models on
# first boot so `docker compose up` is the only command a reviewer needs. Pulled models
# live on a named volume, so subsequent starts are instant and fully offline.
set -e

ollama serve &
server_pid=$!

echo "Waiting for Ollama to be ready..."
until ollama list >/dev/null 2>&1; do
  sleep 1
done

CHAT_MODEL="${OLLAMA_MODEL:-llama3.1:8b}"
echo "Pulling chat model: ${CHAT_MODEL}"
ollama pull "${CHAT_MODEL}"

# Only needed if you switch embeddings to the Ollama backend (EMBED_BACKEND=ollama).
if [ "${PULL_EMBED_MODEL:-false}" = "true" ]; then
  EMBED_MODEL_NAME="${EMBED_MODEL:-nomic-embed-text}"
  echo "Pulling embedding model: ${EMBED_MODEL_NAME}"
  ollama pull "${EMBED_MODEL_NAME}"
fi

echo "Models ready."
wait "${server_pid}"

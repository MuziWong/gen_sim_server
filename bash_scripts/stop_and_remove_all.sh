#!/usr/bin/env bash
set -euo pipefail

declare -a CONTAINERS=(
"gensim-sam3d-server"
"gensim-sam3-server"
"gensim-z-image-server"
)

for container in "${CONTAINERS[@]}"; do
    if ! docker ps -a --format '{{.Names}}' | grep -qx "$container"; then
        echo "[SKIP]    $container: container does not exist"
        continue
    fi

    if [ "$(docker inspect -f '{{.State.Running}}' "$container")" = "true" ]; then
        docker stop "$container" >/dev/null
        echo "[STOPPED] $container"
    fi

    docker rm "$container" >/dev/null
    echo "[REMOVED] $container"
done

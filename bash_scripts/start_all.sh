#!/usr/bin/env bash
set -euo pipefail

IMAGE="gensim-engine:wrc"

# name | conda environment | working directory | script
declare -a SERVICES=(
"sam3d-server|sam3d-objects|/workspace/source/sam-3d-server/server|flask_api.py"
"sam3-server|sam3|/workspace/source/sam3-server/server|flask_api.py"
"z-image-server|z-image|/workspace/source/z-image-server/server|flask_api.py"
)

start_service() {
    local name="$1"
    local env_name="$2"
    local workdir="$3"
    local script_name="$4"
    local container="gensim-${name}"

    if docker ps -a --format '{{.Names}}' | grep -qx "$container"; then
        if [ "$(docker inspect -f '{{.State.Running}}' "$container")" != "true" ]; then
            docker start "$container" >/dev/null
            echo "[STARTED] $container"
        else
            echo "[RUNNING] $container"
        fi
        return
    fi

    docker run -d \
        --name "$container" \
        --restart unless-stopped \
        --network host \
        --gpus all \
        "$IMAGE" \
        bash -lc "
            source /opt/conda/etc/profile.d/conda.sh
            conda activate ${env_name}
            cd ${workdir}
            exec python ./${script_name}
        "

    echo "[CREATED] $container"
}

check_service() {
    local name="$1"
    local _env_name="$2"
    local _workdir="$3"
    local script_name="$4"
    local container="gensim-${name}"

    if [ "$(docker inspect -f '{{.State.Running}}' "$container")" != "true" ]; then
        echo "[FAILED] $container: container is not running"
        docker logs --tail 30 "$container" || true
        return 1
    fi

    if docker exec "$container" pgrep -af "$script_name" >/dev/null 2>&1; then
        echo "[READY]  $container: ${script_name}"
    else
        echo "[FAILED] $container: service process is not running"
        docker logs --tail 30 "$container" || true
        return 1
    fi
}

for service in "${SERVICES[@]}"; do
    IFS='|' read -r name env_name workdir script_name <<< "$service"
    start_service "$name" "$env_name" "$workdir" "$script_name"
done

sleep 5

echo
echo "Service status:"

for service in "${SERVICES[@]}"; do
    IFS='|' read -r name env_name workdir script_name <<< "$service"
    check_service "$name" "$env_name" "$workdir" "$script_name" || true
done
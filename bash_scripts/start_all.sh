#!/usr/bin/env bash
set -euo pipefail

IMAGE="gensim-engine-5000pro:wrc"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/../source"

# name | conda environment | working directory | repository entry point (relative to source/) | health port
declare -a SERVICES=(
"sam3d-server|sam3d-objects|/workspace/source/sam-3d-server/server|sam3d-server/server/flask_api.py|8686"
"sam3-server|sam3|/workspace/source/sam3-server/server|sam3-server/server/inference.py|8685"
"z-image-server|z-image|/workspace/source/z-image-server/server|z-image-server/server/inference.py|8687"
)

# Check all deployment sources before stopping any running container.
python3 - "$SOURCE_DIR" <<'PY'
import ast
import sys
from pathlib import Path
root = Path(sys.argv[1])
for relative in ["common/gpu_memory.py", "sam3-server/server/inference.py", "sam3d-server/server/flask_api.py", "z-image-server/server/inference.py"]:
    path = root / relative
    ast.parse(path.read_text(), filename=str(path))
print("[VALIDATED] service entry points and shared memory helper")
PY

start_service() {
    local name="$1" env_name="$2" workdir="$3" entrypoint="$4" port="$5"
    local container="gensim-${name}" changed=0 was_running=false
    local scratch attempt

    if ! docker container inspect "$container" >/dev/null 2>&1; then
        # Copy the deployment sources before the first process starts.
        docker create \
            --name "$container" \
            --restart unless-stopped \
            --network host \
            --gpus all \
            -e HF_OFFLINE=1 \
            -e HF_HUB_OFFLINE=1 \
            -e TRANSFORMERS_OFFLINE=1 \
            -e HF_DATASETS_OFFLINE=1 \
            "$IMAGE" \
            bash -lc "
                source /opt/conda/etc/profile.d/conda.sh
                conda activate ${env_name}
                cd ${workdir}
                exec python ./flask_api.py
            " >/dev/null
    fi

    was_running="$(docker inspect -f '{{.State.Running}}' "$container")"
    scratch="$(mktemp -d)"
    if ! docker cp "${container}:${workdir}/flask_api.py" "$scratch/flask_api.py" 2>/dev/null \
        || ! cmp -s "$SOURCE_DIR/$entrypoint" "$scratch/flask_api.py"; then
        changed=1
    fi
    if ! docker cp "${container}:${workdir}/gpu_memory.py" "$scratch/gpu_memory.py" 2>/dev/null \
        || ! cmp -s "$SOURCE_DIR/common/gpu_memory.py" "$scratch/gpu_memory.py"; then
        changed=1
    fi
    rm -rf -- "$scratch"

    if [[ "$changed" == 1 ]]; then
        if [[ "$was_running" == true ]]; then
            docker stop -t 30 "$container" >/dev/null
        fi
        docker cp "$SOURCE_DIR/$entrypoint" "${container}:${workdir}/flask_api.py"
        docker cp "$SOURCE_DIR/common/gpu_memory.py" "${container}:${workdir}/gpu_memory.py"
        docker start "$container" >/dev/null
        echo "[UPDATED] $container: deployment sources installed and service started"
    elif [[ "$was_running" != true ]]; then
        docker start "$container" >/dev/null
        echo "[STARTED] $container"
    else
        echo "[RUNNING] $container: deployment sources unchanged"
    fi

    # A live Python process does not prove that the model has loaded.
    local ready=0
    for ((attempt = 0; attempt < 60; attempt++)); do
        if docker exec "$container" python -c \
            "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${port}/health', timeout=2)" \
            >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 5
    done
    if [[ "$ready" != 1 ]]; then
        echo "[FAILED] $container: health endpoint did not become ready" >&2
        docker logs --tail 30 "$container" >&2 || true
        return 1
    fi
    echo "[READY] $container: model health HTTP 200"
}

for service in "${SERVICES[@]}"; do
    IFS='|' read -r name env_name workdir entrypoint port <<< "$service"
    start_service "$name" "$env_name" "$workdir" "$entrypoint" "$port"
done

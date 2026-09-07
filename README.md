# GenSim Server

GenSim Server runs the SAM3D, SAM3, and Z-Image services in Docker containers.

## Docker Image

The scripts use the following Docker image:

```text
gensim-engine-5000pro:wrc
```

Make sure this image is available locally before starting the services.

The image already contains the Conda environments required by the three
services:

- `sam3d-objects` for SAM3D
- `sam3` for SAM3
- `z-image` for Z-Image

This repository does not contain a Dockerfile or everything needed to rebuild
that image. The image supplies model code, weights, dependencies, and the
service configuration. The startup script installs the API entry points and
the shared memory helper from this checkout into the containers.

The `source/` directory also records server backends and checkpoint locations:

- `source/sam3d-server/sam-3d-objects/checkpoints/`
- `source/sam3-server/sam3/checkpoints/`
- `source/z-image-server/Z-Image/ckpts/`

These directories are kept in Git with `.gitkeep` files. The actual model
weights are not included in this repository.

The upstream projects are:

- [SAM3](https://github.com/facebookresearch/sam3)
- [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects)
- [Z-Image](https://github.com/Tongyi-MAI/Z-Image)

## Bash Scripts

The scripts are located in `bash_scripts/`.

### Start all services

```bash
bash bash_scripts/start_all.sh
```

Run this on the deployment server with the complete checkout available. Paths
are resolved relative to the script, so it can be invoked from any directory.
The script manages these services:

- `gensim-sam3d-server`
- `gensim-sam3-server`
- `gensim-z-image-server`

Before touching containers, it checks the Python syntax of all managed entry
points and the shared helper. New containers receive these files before their
first start. For existing containers, it compares their installed files with
this checkout, stops and updates only services whose files changed, and starts
stopped services. Repeated runs leave unchanged running services running.
Each service must pass its HTTP `/health` check before deployment continues;
a timeout fails the script and prints recent container logs.

An update stops the affected service and can interrupt active requests. SAM3D
keeps jobs and generated assets in process memory, so restarting loses them.
Deploy during an idle period. The script uses host networking, `--gpus all`,
and `unless-stopped`, matching the existing server deployment. GPU selection
is application-level through the image's `/workspace/source/service_config.json`.

| Service | Repository entry point | Container server directory | Physical GPU | Health port |
|---|---|---|---|---|
| SAM3 | `source/sam3-server/server/inference.py` | `/workspace/source/sam3-server/server` | 0 | 8685 |
| SAM3D | `source/sam3d-server/server/flask_api.py` | `/workspace/source/sam-3d-server/server` | 1 | 8686 |
| Z-Image | `source/z-image-server/server/inference.py` | `/workspace/source/z-image-server/server` | 2 | 8687 |

Every entry point is installed as `flask_api.py` in its container server
directory. `source/common/gpu_memory.py` is installed alongside it. The SAM3D
repository directory is named `sam3d-server`; its image directory is named
`sam-3d-server`. This mapping is intentional. When running directly from the
checkout in a suitable model environment, entry points also resolve the
helper from `source/common/`.

The script does **not** synchronize `service_config.json`, SAM3D's backend
`inference.py`, or model weights. Changes to those require a separate image
update or explicit installation. The health ports above must match the image
configuration. Existing container environment variables and image selection
are retained; new containers use the listed image and offline model-loading
environment variables. Qwen is outside this script's scope.

## Idle GPU memory reclamation

All three services track pending inference work. After all work finishes and
the service stays idle for one second, a background callback runs garbage
collection, CUDA synchronization, and `torch.cuda.empty_cache()`. Model weights
stay loaded. New work cancels a pending callback or waits for a callback already
in progress. SAM3D counts queued and running jobs, including failure and reset
paths, and exports each object's GPU output to GLB and CPU pose data before
processing the next mask.

The APIs and response formats are unchanged. Reclamation frees unused allocator
cache, so it does not reduce memory below the live model and library allocations.
It also introduces cleanup and subsequent allocation overhead; continuous work
does not trigger cleanup between requests.

See [MEMORY_CLEANUP.md](MEMORY_CLEANUP.md) for the policy, logs, and deployment
validation boundaries.

```bash
docker logs --since 10m gensim-sam3d-server 2>&1 | grep '\[gpu-memory\]'
nvidia-smi
```

### Stop and remove all services

```bash
bash bash_scripts/stop_and_remove_all.sh
```

This script stops and removes the three service containers created by
`start_all.sh`.

## Requirements

- Docker
- NVIDIA Docker support and a compatible GPU
- Python 3 on the deployment host for syntax validation
- The `gensim-engine-5000pro:wrc` image

## Local validation

These checks use a fake CUDA allocator and fake Docker; they do not load models
or change real containers:

```bash
bash -n bash_scripts/start_all.sh bash_scripts/stop_and_remove_all.sh
python3 -m unittest discover -s tests -v
```

After deployment, validate real inference and the `idle_reclaim` log following
the last request. A successful health check alone does not verify generation
or GPU memory reclamation.

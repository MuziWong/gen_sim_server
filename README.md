# GenSim Server

GenSim Server runs the SAM3D, SAM3, and Z-Image services in Docker containers.

## Docker Image

The scripts use the following Docker image:

```text
gensim-engine:wrc
```

Make sure this image is available locally before starting the services.

The image already contains the Conda environments required by the three
services:

- `sam3d-objects` for SAM3D
- `sam3` for SAM3
- `z-image` for Z-Image

The `source/` directory in this repository contains the files copied into
`/workspace/` in the Docker image. It mainly includes the server backends,
the related source code, and the model checkpoint locations:

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

This script creates and starts the service containers, or starts existing
containers if they already exist. It launches:

- `gensim-sam3d-server`
- `gensim-sam3-server`
- `gensim-z-image-server`

It also checks whether each service process is running and prints its status.

### Stop and remove all services

```bash
bash bash_scripts/stop_and_remove_all.sh
```

This script stops and removes the three service containers created by
`start_all.sh`.

## Requirements

- Docker
- NVIDIA Docker support and a compatible GPU
- The `gensim-engine:wrc` image

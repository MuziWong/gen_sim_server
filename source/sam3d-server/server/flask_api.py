#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from typing import Any
import uuid

# Repository entry points share source/common; deployed containers receive the
# helper beside flask_api.py and use the normal script-directory import path.
_COMMON_DIR = Path(__file__).resolve().parents[2] / "common"
if _COMMON_DIR.is_dir():
    sys.path.insert(0, str(_COMMON_DIR))
from gpu_memory import IdleCudaReclaimer


def _load_service_settings(service_name: str) -> dict[str, Any]:
    default_path = Path(__file__).resolve().parents[2] / "service_config.json"
    config_path = (
        Path(os.getenv("SERVICE_CONFIG_PATH", str(default_path))).expanduser().resolve()
    )
    try:
        settings = json.loads(config_path.read_text(encoding="utf-8"))[service_name]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Failed to load service {service_name!r} from {config_path}: {exc}"
        ) from exc
    if not isinstance(settings, dict):
        raise RuntimeError(f"Service config {service_name!r} must be an object")
    if not isinstance(settings.get("host"), str) or not settings["host"]:
        raise RuntimeError(f"Service config {service_name!r} has invalid host")
    if not isinstance(settings.get("port"), int) or not 1 <= settings["port"] <= 65535:
        raise RuntimeError(f"Service config {service_name!r} has invalid port")
    if not isinstance(settings.get("gpu"), int) or settings["gpu"] < 0:
        raise RuntimeError(f"Service config {service_name!r} has invalid gpu")
    return settings


_SERVICE_SETTINGS = _load_service_settings("sam3d")
# This must happen before torch and pytorch3d are imported.
os.environ["CUDA_VISIBLE_DEVICES"] = str(_SERVICE_SETTINGS["gpu"])

from flask import Flask, jsonify, request, send_file
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename

__all__ = ["create_app", "main"]

_ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_ALLOWED_MASK_SUFFIXES = {".png"}


@dataclass(frozen=True)
class Sam3DServerConfig:
    model_config_path: Path
    host: str = "0.0.0.0"
    port: int = 8686


@dataclass
class _Job:
    job_id: str
    image_bytes: bytes
    masks: list[tuple[str, bytes]]
    status: str = "queued"
    error: str | None = None
    objects: list[dict[str, Any]] | None = None
    assets: dict[str, bytes] | None = None
    cancelled: bool = False
    done: threading.Event = field(default_factory=threading.Event)


class _State:
    """Single-worker in-memory task manager."""

    def __init__(self, cfg: Sam3DServerConfig) -> None:
        self.cfg = cfg
        self.lock = threading.RLock()
        self.cv = threading.Condition(self.lock)

        self.jobs: dict[str, _Job] = {}
        self.queue: deque[str] = deque()
        self.current_job_id: str | None = None
        self.cancel_current = False

        self.inference_module: Any | None = None
        self.inference: Any | None = None
        self.init_error: str | None = None
        self.inference_lock = threading.RLock()

        self._load_model()
        self._memory = IdleCudaReclaimer(torch, "sam3d")

        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def model_status(self) -> tuple[bool, str]:
        if self.init_error is not None:
            return False, self.init_error
        return True, "ready"

    def enqueue(
        self, image_bytes: bytes, masks: list[tuple[str, bytes]]
    ) -> tuple[_Job, int]:
        job = _Job(job_id=uuid.uuid4().hex, image_bytes=image_bytes, masks=masks)
        with self.lock:
            self._memory.begin()
            self.jobs[job.job_id] = job
            self.queue.append(job.job_id)
            waiting = len(self.queue) - 1 + (1 if self.current_job_id else 0)
            self.cv.notify_all()
        return job, waiting

    def reset(self) -> dict[str, Any]:
        with self.lock:
            queued_ids = list(self.queue)
            self.queue.clear()
            for job_id in queued_ids:
                job = self.jobs.get(job_id)
                if job is not None and not job.done.is_set():
                    job.status = "cancelled"
                    job.cancelled = True
                    job.error = "reset"
                    job.done.set()
                    self._memory.end()

            current_id = self.current_job_id
            if current_id is not None:
                self.cancel_current = True
                current = self.jobs.get(current_id)
                if current is not None and current.status in {"queued", "running"}:
                    current.status = "cancelling"

            self.cv.notify_all()

        return {
            "ok": True,
            "status": "reset",
            "cancel_current": current_id is not None,
            "cleared_queue": len(queued_ids),
        }

    def get_job(self, job_id: str) -> _Job | None:
        with self.lock:
            return self.jobs.get(job_id)

    def waiting_count(self, job_id: str) -> int:
        with self.lock:
            try:
                idx = list(self.queue).index(job_id)
                return idx + 1 + (1 if self.current_job_id else 0)
            except ValueError:
                return 0

    def get_asset_bytes(self, job_id: str, filename: str) -> bytes | None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None or not job.assets:
                return None
            return job.assets.get(filename)

    def _worker_loop(self) -> None:
        while True:
            with self.lock:
                while not self.queue:
                    self.current_job_id = None
                    self.cv.wait()

                job_id = self.queue.popleft()
                job = self.jobs.get(job_id)
                if job is None:
                    continue

                self.current_job_id = job_id
                job.status = "running"

            try:
                objects, assets = self._run_job(job)
                with self.lock:
                    if job.cancelled:
                        job.status = "cancelled"
                        job.error = job.error or "cancelled"
                    else:
                        job.status = "succeeded"
                        job.objects = objects
                        job.assets = assets
            except Exception as exc:  # pragma: no cover
                import traceback

                print(
                    f"[sam3d-server] job {job.job_id} failed: {exc}",
                    flush=True,
                )
                traceback.print_exc()

                with self.lock:
                    job.status = "failed"
                    job.error = str(exc)
            finally:
                with self.lock:
                    if self.current_job_id == job.job_id:
                        self.current_job_id = None
                    self.cancel_current = False
                    job.done.set()
                    self.cv.notify_all()
                self._memory.end()

    def _run_job(self, job: _Job) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
        if self.inference_module is None or self.inference is None:
            raise RuntimeError(self.init_error or "SAM3D model is not initialized.")

        image = _decode_image(self.inference_module, job.image_bytes)
        shared_pointmap = _compute_shared_pointmap(
            self.inference_module, self.inference, image
        )
        assets: dict[str, bytes] = {}
        objects: list[dict[str, Any]] = []
        for object_id, mask_bytes in job.masks:
            with self.lock:
                if self.cancel_current:
                    job.cancelled = True
                    job.error = "cancelled by reset"
                    return [], {}
            with self.inference_lock:
                obj, object_assets = self._run_object_cpu(
                    job.job_id, object_id, image, mask_bytes, shared_pointmap
                )
            objects.append(obj)
            assets.update(object_assets)
        return objects, assets

    def _run_object_cpu(
        self, job_id: str, object_id: str, image: Any, mask_bytes: bytes, pointmap: Any
    ) -> tuple[dict[str, Any], dict[str, bytes]]:
        """Export one object to CPU-only results before the next inference."""
        mask = _decode_mask(self.inference_module, mask_bytes)
        output = self.inference(image, mask, seed=42, pointmap=pointmap)
        assets: dict[str, bytes] = {}
        objects: list[dict[str, Any]] = []
        filename = f"{object_id}.glb"
        glb_bytes = _export_output_as_glb_bytes(output)
        assets[filename] = glb_bytes
        sam3d_rotation = _extract_vector(
            output,
            # SAM3D uses "rotation" internally. The public API below
            # serializes a converted quaternion with an explicit wxyz name.
            key="rotation",
            expected_len=4,
            default=[1.0, 0.0, 0.0, 0.0],
        )
        sam3d_translation = _extract_vector(
            output,
            key="translation",
            expected_len=3,
            default=[0.0, 0.0, 0.0],
        )
        sam3d_scale = _extract_vector(
            output,
            key="scale",
            expected_len=3,
            default=[1.0, 1.0, 1.0],
        )
        rotation, translation, scale = _sam3d_pose_to_glb_y_up(
            rotation_quaternion_wxyz=sam3d_rotation,
            translation=sam3d_translation,
            scale=sam3d_scale,
        )
        transform_filename = f"{object_id}.json"
        assets[transform_filename] = (
            json.dumps(
                {
                    "name": object_id,
                    "pose_coordinate_system": "glb_y_up",
                    "rotation_quaternion_wxyz": rotation,
                    "translation": translation,
                    "scale": scale,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
        objects.append(
            {
                "name": object_id,
                "mesh": f"/assets/{job_id}/{filename}",
                "transform": f"/assets/{job_id}/{transform_filename}",
                "pose_coordinate_system": "glb_y_up",
                "rotation_quaternion_wxyz": rotation,
                "translation": translation,
                "scale": scale,
            }
        )
        return objects[0], assets

    def _load_model(self) -> None:
        try:
            module = importlib.import_module("inference")
            cfg = self.cfg.model_config_path
            if not cfg.is_file():
                raise FileNotFoundError(f"SAM3D model config not found: {cfg}")
            self.inference_module = module
            self.inference = module.Inference(str(cfg), compile=False)
            self.init_error = None
        except Exception as exc:  # pragma: no cover
            self.inference_module = None
            self.inference = None
            self.init_error = str(exc)
            raise RuntimeError(f"Failed to load SAM3D model: {exc}") from exc


def create_app(cfg: Sam3DServerConfig) -> Flask:
    state = _State(cfg)
    app = Flask(__name__)

    @app.get("/health")
    def health() -> Any:
        ok, detail = state.model_status()
        if ok:
            return jsonify({"ok": True, "status": "ready"})
        return jsonify({"ok": False, "status": "error", "error": detail}), 503

    @app.post("/generate_multiple_objects")
    def generate_multiple_objects() -> Any:
        try:
            image_file = request.files.get("image")
            if image_file is None:
                raise BadRequest("Missing uploaded image file field: image")

            mask_files = request.files.getlist("masks")
            if not mask_files:
                raise BadRequest("Missing uploaded mask files field: masks")

            image_bytes = _read_upload_bytes(
                image_file,
                field_name="image",
                allowed_suffixes=_ALLOWED_IMAGE_SUFFIXES,
            )
            masks = [
                (
                    _object_id_from_mask_name(mask_file.filename, index=index),
                    _read_upload_bytes(
                        mask_file,
                        field_name=f"mask_{index}",
                        allowed_suffixes=_ALLOWED_MASK_SUFFIXES,
                    ),
                )
                for index, mask_file in enumerate(mask_files)
            ]

            job, waiting = state.enqueue(image_bytes=image_bytes, masks=masks)
            if waiting > 0:
                return jsonify(
                    {
                        "ok": True,
                        "status": f"waiting {waiting}",
                        "request_id": job.job_id,
                        "status_url": f"/tasks/{job.job_id}",
                    }
                )

            job.done.wait()
            if job.status == "succeeded":
                return jsonify(
                    {
                        "ok": True,
                        "request_id": job.job_id,
                        "result": {"objects": job.objects},
                    }
                )
            if job.status == "cancelled":
                return (
                    jsonify(
                        {
                            "ok": False,
                            "request_id": job.job_id,
                            "status": "cancelled",
                            "error": job.error,
                        }
                    ),
                    409,
                )
            return (
                jsonify(
                    {
                        "ok": False,
                        "request_id": job.job_id,
                        "status": job.status,
                        "error": job.error,
                    }
                ),
                500,
            )
        except BadRequest as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.get("/tasks/<job_id>")
    def task_status(job_id: str) -> Any:
        job = state.get_job(job_id)
        if job is None:
            return jsonify({"ok": False, "error": "task not found"}), 404

        payload: dict[str, Any] = {
            "ok": True,
            "request_id": job.job_id,
            "status": job.status,
        }
        if job.status == "queued":
            payload["status"] = f"waiting {state.waiting_count(job.job_id)}"
        if job.status == "succeeded":
            payload["result"] = {"objects": job.objects}
        if job.status in {"failed", "cancelled", "cancelling"}:
            payload["error"] = job.error

        return jsonify(payload)

    @app.post("/reset")
    def reset() -> Any:
        return jsonify(state.reset())

    @app.get("/assets/<job_id>/<filename>")
    def get_asset(job_id: str, filename: str) -> Any:
        safe = secure_filename(filename)
        if not safe or safe != filename:
            return jsonify({"ok": False, "error": "invalid asset filename"}), 400

        data = state.get_asset_bytes(job_id, filename)
        if data is None:
            return jsonify({"ok": False, "error": "asset not found"}), 404

        return send_file(
            io.BytesIO(data),
            mimetype=(
                "application/json"
                if filename.lower().endswith(".json")
                else "model/gltf-binary"
            ),
            as_attachment=False,
            download_name=filename,
        )

    return app


def _read_upload_bytes(
    file: FileStorage,
    *,
    field_name: str,
    allowed_suffixes: set[str],
) -> bytes:
    filename = secure_filename(file.filename or "")
    if not filename:
        raise BadRequest(f"Uploaded field {field_name} has empty filename.")

    suffix = Path(filename).suffix.lower()
    if suffix not in allowed_suffixes:
        raise BadRequest(
            f"Uploaded field {field_name} only supports: {sorted(allowed_suffixes)}"
        )

    data = file.read()
    if not data:
        raise BadRequest(f"Uploaded field {field_name} is empty.")
    return data


def _object_id_from_mask_name(filename: str | None, *, index: int) -> str:
    stem = Path(secure_filename(filename or "")).stem
    return stem if stem else f"object_{index:03d}"


def _decode_image(module: Any, image_bytes: bytes) -> Any:
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        tmp.write(image_bytes)
        tmp.flush()
        return module.load_image(tmp.name)


def _decode_mask(module: Any, mask_bytes: bytes) -> Any:
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        tmp.write(mask_bytes)
        tmp.flush()
        return module.load_mask(tmp.name)


def _compute_shared_pointmap(module: Any, inference: Any, image: Any) -> Any:
    """Compute the image pointmap once so all object poses share one scene frame."""
    rgba_image = module.np.concatenate(
        [
            image[..., :3],
            module.np.full((*image.shape[:2], 1), 255, dtype=module.np.uint8),
        ],
        axis=-1,
    )
    with module.torch.inference_mode():
        pointmap = inference._pipeline.compute_pointmap(rgba_image)["pointmap"]

    # compute_pointmap returns C,H,W, while pipeline.run(pointmap=...) expects
    # H,W,C. Keeping the cached tensor on CPU mirrors SAM3D's documented usage.
    return pointmap.detach().cpu().permute(1, 2, 0).contiguous()


def _export_output_as_glb_bytes(output: Any) -> bytes:
    with tempfile.TemporaryDirectory(prefix="sam3d-glb-") as tmp_dir:
        tmp_glb = Path(tmp_dir) / "obj.glb"
        glb = (
            output.get("glb")
            if isinstance(output, dict)
            else getattr(output, "glb", None)
        )
        if glb is None:
            raise RuntimeError("SAM3D output does not contain a GLB mesh")
        glb.export(tmp_glb)
        return tmp_glb.read_bytes()


def _sam3d_pose_to_glb_y_up(
    *,
    rotation_quaternion_wxyz: list[float],
    translation: list[float],
    scale: list[float],
) -> tuple[list[float], list[float], list[float]]:
    """Convert SAM3D's reconstruction-frame SRT to the exported GLB frame.

    ``output[\"glb\"]`` is written by SAM3D's ``to_glb()`` in GLB y-up local
    coordinates. Its predicted ``rotation``, ``translation``, and ``scale``
    instead map from SAM3D's reconstruction coordinates into its scene frame.
    Returning those raw values together with the GLB makes a client apply one
    transform in two incompatible coordinate systems.

    The basis below maps a SAM3D coordinate vector to GLB y-up coordinates:
    ``(x, y, z)_sam3d -> (x, z, -y)_glb_y_up``. Apply it to the complete SRT
    transform, rather than only permuting translation, so rotations and a
    possible non-uniform scale stay consistent with the unchanged GLB mesh.
    PyTorch3D uses and returns quaternions in ``[w, x, y, z]`` order.
    """
    sam_to_glb_y_up = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    rotation_sam3d = torch.tensor(rotation_quaternion_wxyz, dtype=torch.float64)
    translation_sam3d = torch.tensor(translation, dtype=torch.float64)
    scale_sam3d = torch.tensor(scale, dtype=torch.float64)

    # This is the complete rotation basis change B @ R_sam3d @ B.T. The
    # quaternion helpers both use SAM3D/PyTorch3D's [w, x, y, z] convention.
    rotation_glb_y_up = (
        sam_to_glb_y_up @ quaternion_to_matrix(rotation_sam3d) @ sam_to_glb_y_up.T
    )
    # B @ [x, y, z] = [x, z, -y].
    translation_glb_y_up = sam_to_glb_y_up @ translation_sam3d
    # B @ diag([sx, sy, sz]) @ B.T = diag([sx, sz, sy]).
    scale_glb_y_up = torch.stack([scale_sam3d[0], scale_sam3d[2], scale_sam3d[1]])
    quaternion_glb_y_up = matrix_to_quaternion(rotation_glb_y_up)

    return (
        [float(value) for value in quaternion_glb_y_up.tolist()],
        [float(value) for value in translation_glb_y_up.tolist()],
        [float(value) for value in scale_glb_y_up.tolist()],
    )


def _extract_vector(
    output: Any,
    *,
    key: str,
    expected_len: int,
    default: list[float],
) -> list[float]:
    value = output.get(key) if isinstance(output, dict) else getattr(output, key, None)
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()

    # SAM3D tensors have a batch dimension: [1, 4] or [1, 3].
    if (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(value[0], (list, tuple))
    ):
        value = value[0]
    if not isinstance(value, (list, tuple)) or len(value) != expected_len:
        return default
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return default


def _default_model_config_path() -> Path:
    if env_path := os.getenv("SAM3D_CONFIG_PATH"):
        return Path(env_path).expanduser().resolve()
    tag = os.getenv("SAM3D_CHECKPOINT_TAG", "hf")
    workdir = Path(os.getenv("SAM3D_WORKDIR", os.getcwd())).expanduser().resolve()
    return (
        workdir / ".." / "sam-3d-objects" / "checkpoints" / "pipeline.yaml"
    ).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM3D Flask geometry generation server"
    )
    parser.add_argument("--host", default=_SERVICE_SETTINGS["host"])
    parser.add_argument("--port", type=int, default=_SERVICE_SETTINGS["port"])
    parser.add_argument(
        "--config-path", type=Path, default=_default_model_config_path()
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    app = create_app(
        Sam3DServerConfig(
            model_config_path=args.config_path.expanduser().resolve(),
            host=args.host,
            port=args.port,
        )
    )
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()

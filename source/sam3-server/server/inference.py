#!/usr/bin/env python3
"""HTTP API for text-prompted SAM3 image segmentation."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import threading
from typing import Any, Callable

from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import BadRequest, HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

__all__ = ["Sam3ServerConfig", "create_app", "main"]

_ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


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


_SERVICE_SETTINGS = _load_service_settings("sam3")
# This runs before _State imports torch, so CUDA only sees the configured GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = str(_SERVICE_SETTINGS["gpu"])


@dataclass(frozen=True)
class Sam3ServerConfig:
    checkpoint_path: Path
    device: str = "cuda"
    confidence_threshold: float = 0.5
    host: str = "0.0.0.0"
    port: int = 8685
    max_upload_bytes: int = 32 * 1024 * 1024
    compile_model: bool = False


class _State:
    """Own the model for the process and serialize access to its GPU state."""

    def __init__(self, cfg: Sam3ServerConfig) -> None:
        import torch
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        if not 0.0 <= cfg.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not cfg.checkpoint_path.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {cfg.checkpoint_path}")
        if cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("SAM3 was configured for CUDA, but CUDA is unavailable")

        self._torch = torch
        self._device = cfg.device
        model = build_sam3_image_model(
            checkpoint_path=str(cfg.checkpoint_path),
            load_from_HF=False,
            device=cfg.device,
            eval_mode=True,
            compile=cfg.compile_model,
        )
        self._processor = Sam3Processor(
            model,
            device=cfg.device,
            confidence_threshold=cfg.confidence_threshold,
        )
        self._lock = threading.Lock()

    def segment(self, image: Image.Image, prompt: str) -> list[dict[str, Any]]:

        # Sam3Processor stores its threshold and find-stage tensors, so treat it
        # as a single-owner GPU object even though each request gets fresh state.
        with self._lock, self._inference_context():
            state = self._processor.set_image(image)
            state = self._processor.set_text_prompt(prompt=prompt, state=state)
            masks = state["masks"].squeeze(1)
            rles = _encode_uncompressed_rles(masks)
            boxes = state["boxes"].detach().float().cpu().tolist()
            scores = state["scores"].detach().float().cpu().tolist()

        return [
            {
                "mask_rle": rle,
                "score": float(score),
                "box_xyxy": [float(value) for value in box],
            }
            for rle, score, box in zip(rles, scores, boxes)
        ]

    def _inference_context(self) -> Any:
        if not self._device.startswith("cuda"):
            return nullcontext()
        return self._torch.autocast(device_type="cuda", dtype=self._torch.bfloat16)


def _encode_uncompressed_rles(masks: Any) -> list[dict[str, Any]]:
    """Encode N,H,W masks using the client's row-major uncompressed RLE."""
    if masks.ndim != 3:
        raise RuntimeError(
            f"SAM3 masks must have shape N,H,W; got {tuple(masks.shape)}"
        )

    height, width = (int(value) for value in masks.shape[1:])
    encoded: list[dict[str, Any]] = []
    for mask in masks:
        # The client reconstructs the bytes directly with PIL.Image.frombytes,
        # which consumes pixels row by row rather than in COCO's column order.
        flat = mask.reshape(-1).bool()
        change_indices = (
            ((flat[1:] != flat[:-1]).nonzero(as_tuple=False).flatten() + 1)
            .cpu()
            .tolist()
        )
        boundaries = [0, *change_indices, int(flat.numel())]
        counts = [
            boundaries[index + 1] - boundaries[index]
            for index in range(len(boundaries) - 1)
        ]
        if bool(flat[0].item()):
            counts.insert(0, 0)
        encoded.append(
            {"size": [height, width], "counts": counts, "starts_with": 0}
        )
    return encoded


def create_app(
    cfg: Sam3ServerConfig,
    *,
    state_factory: Callable[[Sam3ServerConfig], Any] = _State,
) -> Flask:
    """Create the app; ``state_factory`` keeps HTTP behavior unit-testable."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = cfg.max_upload_bytes
    state = state_factory(cfg)

    @app.get("/health")
    def health() -> Any:
        return jsonify({"ok": True, "status": "ready"})

    @app.post("/segment_by_prompt")
    def segment_by_prompt() -> Any:
        image_file = request.files.get("image")
        if image_file is None:
            raise BadRequest("Missing uploaded image file field: image")

        prompt = request.form.get("prompt", "").strip()
        if not prompt:
            raise BadRequest("Missing non-empty form field: prompt")

        filename = secure_filename(image_file.filename or "")
        if not filename:
            raise BadRequest("Uploaded image has an empty filename")
        if Path(filename).suffix.lower() not in _ALLOWED_IMAGE_SUFFIXES:
            raise BadRequest(
                f"Uploaded image only supports: {sorted(_ALLOWED_IMAGE_SUFFIXES)}"
            )

        image_bytes = image_file.read()
        if not image_bytes:
            raise BadRequest("Uploaded image is empty")
        try:
            with Image.open(io.BytesIO(image_bytes)) as uploaded_image:
                uploaded_image.load()
                image = uploaded_image.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise BadRequest("Uploaded file is not a valid image") from exc

        instances = state.segment(image, prompt)
        masks = [instance["mask_rle"] for instance in instances]
        return jsonify(
            {
                "ok": True,
                "result": {
                    "prompt": prompt,
                    "masks": masks,
                    "instances": instances,
                },
            }
        )

    @app.errorhandler(BadRequest)
    def bad_request(exc: BadRequest) -> Any:
        return jsonify({"ok": False, "error": exc.description}), 400

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_: RequestEntityTooLarge) -> Any:
        return jsonify({"ok": False, "error": "Uploaded image is too large"}), 413

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException) -> Any:
        return jsonify({"ok": False, "error": exc.description}), exc.code or 500

    @app.errorhandler(Exception)
    def internal_error(exc: Exception) -> Any:
        app.logger.exception("SAM3 request failed")
        return jsonify({"ok": False, "error": str(exc)}), 500

    return app


def _default_checkpoint_path() -> Path:
    default = Path(__file__).resolve().parents[1] / "sam3" / "checkpoints" / "sam3.pt"
    return Path(os.getenv("SAM3_CHECKPOINT_PATH", str(default))).expanduser().resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 image segmentation server")
    parser.add_argument("--host", default=_SERVICE_SETTINGS["host"])
    parser.add_argument("--port", type=int, default=_SERVICE_SETTINGS["port"])
    parser.add_argument(
        "--checkpoint-path", type=Path, default=_default_checkpoint_path()
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=float(os.getenv("SAM3_CONFIDENCE_THRESHOLD", "0.5")),
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=os.getenv("SAM3_COMPILE", "0") == "1",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = Sam3ServerConfig(
        checkpoint_path=args.checkpoint_path.expanduser().resolve(),
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        host=args.host,
        port=args.port,
        compile_model=args.compile,
    )
    create_app(cfg).run(host=cfg.host, port=cfg.port, threaded=True)


if __name__ == "__main__":
    main()


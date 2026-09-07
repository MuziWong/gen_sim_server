#!/usr/bin/env python3
"""HTTP API for single-image Z-Image generation."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any, Callable


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


_SERVICE_SETTINGS = _load_service_settings("zimage")
# Select the physical GPU before importing torch or any Z-Image module.
os.environ["CUDA_VISIBLE_DEVICES"] = str(_SERVICE_SETTINGS["gpu"])

from flask import Flask, jsonify, request, send_file
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

# Repository entry points share source/common; deployed containers receive the
# helper beside flask_api.py and use the normal script-directory import path.
_COMMON_DIR = Path(__file__).resolve().parents[2] / "common"
if _COMMON_DIR.is_dir():
    sys.path.insert(0, str(_COMMON_DIR))
from gpu_memory import IdleCudaReclaimer

__all__ = ["ZImageServerConfig", "create_app", "main"]


@dataclass(frozen=True)
class ZImageServerConfig:
    checkpoint_path: Path
    source_path: Path
    host: str = "0.0.0.0"
    port: int = 8687
    device: str = "cuda:0"
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 8
    guidance_scale: float = 0.0
    seed: int = 42
    attention_backend: str = "_native_flash"
    compile_model: bool = False
    max_request_bytes: int = 1024 * 1024


class _State:
    """Own one preloaded Z-Image model and serialize GPU inference."""

    def __init__(self, cfg: ZImageServerConfig) -> None:
        if not cfg.checkpoint_path.is_dir():
            raise FileNotFoundError(
                f"Z-Image checkpoint directory not found: {cfg.checkpoint_path}"
            )
        if not (cfg.checkpoint_path / "model_index.json").is_file():
            raise FileNotFoundError(
                "Z-Image checkpoint is incomplete; model_index.json not found in "
                f"{cfg.checkpoint_path}"
            )
        if not cfg.source_path.is_dir():
            raise FileNotFoundError(
                f"Z-Image source directory not found: {cfg.source_path}"
            )

        source_text = str(cfg.source_path)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)

        import torch
        from utils import load_from_local_dir, set_attention_backend
        from zimage import generate

        if cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "Z-Image was configured for CUDA, but CUDA is unavailable"
            )
        if cfg.height <= 0 or cfg.width <= 0:
            raise ValueError("Z-Image height and width must be positive")
        if cfg.num_inference_steps <= 0:
            raise ValueError("Z-Image num_inference_steps must be positive")

        self._cfg = cfg
        self._torch = torch
        self._generate = generate
        self._lock = threading.Lock()
        self._components = load_from_local_dir(
            cfg.checkpoint_path,
            device=cfg.device,
            dtype=torch.bfloat16,
            compile=cfg.compile_model,
        )
        set_attention_backend(cfg.attention_backend)
        self._memory = IdleCudaReclaimer(torch, "z-image", cfg.device)

    def generate_png(self, prompt: str) -> bytes:
        return self._memory.run(lambda: self._generate_png_cpu(prompt))

    def _generate_png_cpu(self, prompt: str) -> bytes:
        with self._lock, self._inference_context():
            generator = self._torch.Generator(self._cfg.device).manual_seed(
                self._cfg.seed
            )
            images = self._generate(
                prompt=prompt,
                **self._components,
                height=self._cfg.height,
                width=self._cfg.width,
                num_inference_steps=self._cfg.num_inference_steps,
                guidance_scale=self._cfg.guidance_scale,
                generator=generator,
            )
        if not images:
            raise RuntimeError("Z-Image returned no images")
        output = io.BytesIO()
        images[0].save(output, format="PNG")
        return output.getvalue()

    def _inference_context(self) -> Any:
        if not self._cfg.device.startswith("cuda"):
            return nullcontext()
        return self._torch.inference_mode()


def create_app(
    cfg: ZImageServerConfig,
    *,
    state_factory: Callable[[ZImageServerConfig], Any] = _State,
) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = cfg.max_request_bytes
    state = state_factory(cfg)

    @app.get("/health")
    def health() -> Any:
        return jsonify({"ok": True, "status": "ready"})

    @app.post("/generate_image_by_prompt")
    def generate_image() -> Any:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise BadRequest("Request body must be a JSON object")
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise BadRequest("Missing non-empty JSON field: prompt")

        png_bytes = state.generate_png(prompt.strip())
        return send_file(
            io.BytesIO(png_bytes),
            mimetype="image/png",
            as_attachment=False,
            download_name="generated.png",
            max_age=0,
        )

    @app.errorhandler(BadRequest)
    def bad_request(exc: BadRequest) -> Any:
        return jsonify({"ok": False, "error": exc.description}), 400

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_: RequestEntityTooLarge) -> Any:
        return jsonify({"ok": False, "error": "Request body is too large"}), 413

    @app.errorhandler(Exception)
    def internal_error(exc: Exception) -> Any:
        app.logger.exception("Z-Image request failed")
        return jsonify({"ok": False, "error": str(exc)}), 500

    return app


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_checkpoint_path() -> Path:
    return (_project_root() / "Z-Image" / "ckpts" / "Z-Image-Turbo").resolve()


def _default_source_path() -> Path:
    return (_project_root() / "Z-Image" / "src").resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Z-Image Flask generation server")
    parser.add_argument("--host", default=_SERVICE_SETTINGS["host"])
    parser.add_argument("--port", type=int, default=_SERVICE_SETTINGS["port"])
    parser.add_argument(
        "--checkpoint-path", type=Path, default=_default_checkpoint_path()
    )
    parser.add_argument("--source-path", type=Path, default=_default_source_path())
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--attention-backend",
        default=os.getenv("ZIMAGE_ATTENTION", "_native_flash"),
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=os.getenv("ZIMAGE_COMPILE", "0") == "1",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = ZImageServerConfig(
        checkpoint_path=args.checkpoint_path.expanduser().resolve(),
        source_path=args.source_path.expanduser().resolve(),
        host=args.host,
        port=args.port,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        attention_backend=args.attention_backend,
        compile_model=args.compile,
    )
    create_app(cfg).run(host=cfg.host, port=cfg.port, threaded=True)


if __name__ == "__main__":
    main()

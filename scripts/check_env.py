from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import load_config


def status_icon(level: str) -> str:
    return {
        "OK": "[OK]",
        "WARN": "[WARN]",
        "ERROR": "[ERROR]",
    }[level]


def print_status(level: str, message: str) -> None:
    print(f"{status_icon(level)} {message}")


def check_python() -> bool:
    version = sys.version_info
    ok = (version.major, version.minor) == (3, 10)
    if ok:
        print_status("OK", f"Python {version.major}.{version.minor}.{version.micro} detectado")
    else:
        print_status(
            "WARN",
            f"Python recomendado 3.10, detectado {version.major}.{version.minor}.{version.micro}",
        )
    return ok


def check_module(name: str) -> bool:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "sin __version__")
        print_status("OK", f"Módulo '{name}' disponible ({version})")
        return True
    except Exception as exc:  # noqa: BLE001
        print_status("ERROR", f"Módulo '{name}' no disponible: {exc}")
        return False


def check_torch_cuda() -> None:
    try:
        import torch

        cuda = torch.cuda.is_available()
        print_status("OK", f"torch.cuda.is_available() = {cuda}")
        if cuda:
            try:
                name = torch.cuda.get_device_name(0)
                print_status("OK", f"GPU detectada: {name}")
            except Exception as exc:  # noqa: BLE001
                print_status("WARN", f"CUDA disponible pero no se pudo leer nombre de GPU: {exc}")
        else:
            print_status("WARN", "CUDA no disponible; la app usará CPU")
    except Exception as exc:  # noqa: BLE001
        print_status("ERROR", f"No se pudo validar torch/CUDA: {exc}")


def check_config() -> None:
    config = load_config()
    print_status("OK", f"MODEL_ID: {config.model_id}")
    print_status("OK", f"DEFAULT_PRESET: {config.default_preset}")
    print_status("OK", f"LOG_DIR: {config.log_dir}")

    if "/" not in config.model_id:
        print_status("WARN", "MODEL_ID no parece un repo Hugging Face estándar (owner/model)")

    if config.force_cpu:
        print_status("WARN", "FORCE_CPU=true -> se desactiva GPU manualmente")

    repo_root = Path(__file__).resolve().parents[1]
    run_app = repo_root / "run_app.py"
    if run_app.exists():
        print_status("OK", f"Entrypoint detectado: {run_app}")
    else:
        print_status("ERROR", "No se encontró run_app.py")

    config.log_dir.mkdir(parents=True, exist_ok=True)
    print_status("OK", f"Ruta de logs lista: {config.log_dir.resolve()}")


def main() -> int:
    print("=== Local Note Illustrator | check_env ===")
    check_python()
    torch_ok = check_module("torch")
    diffusers_ok = check_module("diffusers")
    check_module("customtkinter")
    check_module("docx")

    if torch_ok:
        check_torch_cuda()

    check_config()

    if not torch_ok or not diffusers_ok:
        print_status("ERROR", "Faltan dependencias críticas para generar imágenes")
        return 1

    print_status("OK", "Chequeo de entorno completado")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Launch a paper-setting experiment through the installed native Flower CLI."""

import argparse
from contextlib import contextmanager
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import torch
from .config import defaults, validate, METHODS


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS, default="deltamask",
                   help="; ".join(f"{k}: {v}" for k, v in METHODS.items()))
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--model", default="clip-vit-b32")
    p.add_argument("--device", help="Defaults to cuda when available; otherwise choose cpu or mps explicitly")
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--nr-clients", type=int, default=30)
    p.add_argument("--participation", type=float, default=1.)
    p.add_argument("--alpha", type=float, default=10.)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=.1)
    p.add_argument("--probe-learning-rate", type=float, default=.1)
    p.add_argument("--kappa", type=float, default=.8)
    p.add_argument("--mask-epsilon", type=float, default=.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--model-root", type=Path, default=Path("checkpoints"))
    p.add_argument("--model-checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--allow-download", action="store_true", help="Permit pinned model downloads; prepare datasets separately")
    p.add_argument("--from-scratch", action="store_true", help="Explicit architecture verification; does not reproduce pretrained benchmarks")
    p.add_argument("--save-updates", action="store_true", help="Also archive every serialized client update")
    p.add_argument("--registration-timeout", type=float, default=120.)
    p.add_argument("--reply-timeout", type=float, default=3600.)
    p.add_argument("--superlink", default="local")
    p.add_argument("--client-cpus", type=int, default=1)
    p.add_argument("--client-gpus", type=float)
    p.add_argument("--parallel-clients", type=int, default=1,
                   help="Bound local simulation concurrency; size for available memory")
    p.add_argument("--dry-run", action="store_true", help="Print native command and config without launching or acquiring assets")
    return p


def build_config(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else None)
    if device is None:
        raise ValueError("CUDA is unavailable; pass --device cpu or --device mps explicitly")
    cfg = defaults() | {
        "method": args.method, "dataset": args.dataset, "model": args.model,
        "device": device, "num-rounds": args.rounds,
        "num-clients": args.nr_clients, "participation": args.participation,
        "alpha": args.alpha, "local-epochs": args.local_epochs,
        "batch-size": args.batch_size, "learning-rate": args.learning_rate,
        "probe-learning-rate": args.probe_learning_rate, "kappa": args.kappa,
        "mask-epsilon": args.mask_epsilon, "seed": args.seed, "torch-threads": args.torch_threads,
        "data-root": str(args.data_root.expanduser().resolve()),
        "model-root": str(args.model_root.expanduser().resolve()),
        "model-checkpoint": str(args.model_checkpoint.expanduser().resolve()) if args.model_checkpoint else "",
        "allow-download": args.allow_download, "pretrained": not args.from_scratch,
        "save-updates": args.save_updates,
        "registration-timeout": args.registration_timeout, "reply-timeout": args.reply_timeout,
        "output": str((args.output or Path("results") / f"{args.method}-{args.model}-{args.dataset}-seed{args.seed}.json").expanduser().resolve()),
    }
    if args.parallel_clients < 1 or args.client_cpus <= 0:
        raise ValueError("Client resources and concurrency must be positive")
    return validate(cfg)


@contextmanager
def flower_app(cfg):
    """Create only the source bundle and descriptor required by native Flower."""
    import tomli_w

    dist = distribution("deltamask")
    metadata = dist.metadata
    package = Path(__file__).resolve().parent
    assets = package.parent.parent
    if not (assets / "setup.py").is_file() or not (assets / "src/deltamask").is_dir():
        assets = Path(sys.prefix) / "share/deltamask"
    if not (assets / "setup.py").is_file():
        raise RuntimeError("Reinstall DeltaMask from requirements.txt to provide the Flower build assets")
    descriptor = {
        "build-system": {"requires": ["setuptools>=77", "wheel"], "build-backend": "setuptools.build_meta"},
        "project": {
            "name": metadata["Name"], "version": dist.version,
            "description": metadata["Summary"],
            "license": metadata.get("License-Expression") or metadata["License"],
            "license-files": ["LICENSE"],
            "requires-python": metadata["Requires-Python"],
            "dependencies": dist.requires or [],
            "readme": {"text": metadata["Summary"], "content-type": "text/plain"},
            "scripts": {entry.name: entry.value for entry in dist.entry_points if entry.group == "console_scripts"},
        },
        "tool": {"flwr": {"app": {
            "publisher": "nousphera",
            "components": {"serverapp": "deltamask.server_app:app", "clientapp": "deltamask.client_app:app"},
            "config": cfg,
        }}},
    }
    with TemporaryDirectory(prefix="deltamask-flower-") as directory:
        app = Path(directory)
        for source in package.rglob("*"):
            if source.is_file() and source.suffix in {".py", ".cpp", ".h", ".json"} and "__pycache__" not in source.parts:
                target = app / "src/deltamask" / source.relative_to(package)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        shutil.copy2(assets / "setup.py", app / "setup.py")
        for folder, pattern in (("configs", "*.toml"), ("scripts", "*.py")):
            for source in (assets / folder).glob(pattern):
                (app / folder).mkdir(exist_ok=True)
                shutil.copy2(source, app / folder / source.name)
        # This is the same consolidated license included in the native text bundle.
        (app / "LICENSE").write_text(json.loads((package / "native/sources.json").read_text())["LICENSE"])
        (app / "pyproject.toml").write_text(tomli_w.dumps(descriptor))
        yield app


def flower_command(args, cfg, run_path, app_path=None):
    binary = Path(sys.executable).with_name("flwr")
    executable = str(binary) if binary.is_file() else shutil.which("flwr")
    if executable is None:
        raise RuntimeError("Install the project's Flower simulation dependencies first")
    gpus = args.client_gpus if args.client_gpus is not None else (1. if cfg["device"].startswith("cuda") else 0.)
    if gpus < 0 or (cfg["device"].startswith("cuda") and gpus <= 0):
        raise ValueError("CUDA clients require positive --client-gpus")
    resources = (f"num-supernodes={cfg['num-clients']} client-resources-num-cpus={args.client_cpus} "
                 f"client-resources-num-gpus={gpus} init-args-num-cpus={max(1, int(args.client_cpus * args.parallel_clients))}")
    return [executable, "run", str(Path(app_path or run_path.parent).resolve()), args.superlink,
            "--run-config", str(Path(run_path).resolve()), "--federation-config", resources, "--stream"]


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        cfg = build_config(args)
        path = Path(cfg["output"])
        run_path = path.with_suffix(".run.toml")
        if args.dry_run:
            with flower_app(cfg) as app:
                print(json.dumps({"config": cfg, "command": flower_command(args, cfg, run_path, app)}, indent=2))
            return
        from .model_assets import resolve_model
        from .data import load_data
        cfg = resolve_model(cfg)
        # Fail before submitting a run if the explicitly prepared assets are absent.
        load_data(cfg)
        for target in (path, path.with_suffix(".npz"), run_path):
            if target.exists():
                raise FileExistsError(f"Output exists: {target}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with run_path.open("x") as stream:
            for key, value in cfg.items():
                stream.write(f"{key} = {json.dumps(value)}\n")
        env = os.environ.copy()
        env.setdefault("FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION", "1")
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        with flower_app(cfg) as app:
            command = flower_command(args, cfg, run_path, app)
            print(shlex.join(command), flush=True)
            subprocess.run(command, check=True, env=env)
        if not path.is_file() or not path.with_suffix(".npz").is_file():
            raise RuntimeError("Flower ended without saved results; inspect the run logs")
        print(f"Saved {path}", flush=True)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        p.exit(2, f"Experiment failed: {exc}\n")


if __name__ == "__main__":
    main()

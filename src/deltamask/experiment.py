"""Launch a paper-setting experiment through the installed native Flower CLI."""

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import torch
from .config import defaults, validate, METHODS


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS, default="deltamask",
                   help="; ".join(f"{k}: {v}" for k, v in METHODS.items()))
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--setting", choices=["iid", "noniid"], default="iid")
    p.add_argument("--model", default="clip-vit-b32")
    p.add_argument("--device", help="Defaults to cuda when available; otherwise choose cpu or mps explicitly")
    p.add_argument("--rounds", type=int)
    p.add_argument("--clients", type=int, default=30)
    p.add_argument("--participation", type=float)
    p.add_argument("--alpha", type=float)
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
    p.add_argument("--app", type=Path, default=Path.cwd())
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
        "device": device, "num-rounds": 100 if args.setting == "iid" else 300,
        "num-clients": args.clients, "participation": 1. if args.setting == "iid" else .2,
        "alpha": 10. if args.setting == "iid" else .1, "local-epochs": args.local_epochs,
        "batch-size": args.batch_size, "learning-rate": args.learning_rate,
        "probe-learning-rate": args.probe_learning_rate, "kappa": args.kappa,
        "mask-epsilon": args.mask_epsilon, "seed": args.seed, "torch-threads": args.torch_threads,
        "data-root": str(args.data_root.expanduser().resolve()),
        "model-root": str(args.model_root.expanduser().resolve()),
        "model-checkpoint": str(args.model_checkpoint.expanduser().resolve()) if args.model_checkpoint else "",
        "allow-download": args.allow_download, "pretrained": not args.from_scratch,
        "save-updates": args.save_updates,
        "registration-timeout": args.registration_timeout, "reply-timeout": args.reply_timeout,
        "output": str((args.output or Path("results") / f"{args.method}-{args.model}-{args.dataset}-{args.setting}-seed{args.seed}.json").expanduser().resolve()),
    }
    for key, value in (("num-rounds", args.rounds), ("participation", args.participation), ("alpha", args.alpha)):
        if value is not None:
            cfg[key] = value
    if args.parallel_clients < 1 or args.client_cpus <= 0:
        raise ValueError("Client resources and concurrency must be positive")
    return validate(cfg)


def flower_command(args, cfg, run_path):
    binary = Path(sys.executable).with_name("flwr")
    executable = str(binary) if binary.is_file() else shutil.which("flwr")
    if executable is None:
        raise RuntimeError("Install the project's Flower simulation dependencies first")
    gpus = args.client_gpus if args.client_gpus is not None else (1. if cfg["device"].startswith("cuda") else 0.)
    if gpus < 0 or (cfg["device"].startswith("cuda") and gpus <= 0):
        raise ValueError("CUDA clients require positive --client-gpus")
    resources = (f"num-supernodes={cfg['num-clients']} client-resources-num-cpus={args.client_cpus} "
                 f"client-resources-num-gpus={gpus} init-args-num-cpus={max(1, int(args.client_cpus * args.parallel_clients))}")
    return [executable, "run", str(args.app.expanduser().resolve()), args.superlink,
            "--run-config", str(Path(run_path).resolve()), "--federation-config", resources, "--stream"]


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        cfg = build_config(args)
        path = Path(cfg["output"])
        run_path = path.with_suffix(".run.toml")
        command = flower_command(args, cfg, run_path)
        if args.dry_run:
            print(json.dumps({"config": cfg, "command": command}, indent=2))
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
        print(shlex.join(command), flush=True)
        env = os.environ.copy()
        env.setdefault("FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION", "1")
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        subprocess.run(command, check=True, env=env)
        if not path.is_file() or not path.with_suffix(".npz").is_file():
            raise RuntimeError("Flower ended without saved results; inspect the run logs")
        print(f"Saved {path}", flush=True)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        p.exit(2, f"Experiment failed: {exc}\n")


if __name__ == "__main__":
    main()

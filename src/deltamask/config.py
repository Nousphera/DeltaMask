"""Explicit experiment defaults, shared by the CLI and Flower applications."""

import math


def defaults():
    return {
        "num-rounds": 2,
        "num-clients": 2,
        "participation": 1.0,
        "local-epochs": 1,
        "batch-size": 1,
        "learning-rate": 0.1,
        "probe-learning-rate": 0.1,
        "kappa": 0.8,
        "seed": 42,
        "dataset": "cifar10",
        "model": "clip-vit-b32",
        "pretrained": True,
        "torch-threads": 1,
        "data-root": "data",
        "allow-download": False,
        "alpha": 10.0,
        "samples-per-client": 1,
        "output": "results/clip-cifar10.json",
        "device": "cpu",
        "model-revision": "",
        "model-checkpoint": "",
    }


def validate(cfg):
    for key in [
        "num-rounds",
        "num-clients",
        "local-epochs",
        "batch-size",
        "samples-per-client",
    ]:
        if type(cfg[key]) != int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ["learning-rate", "probe-learning-rate", "alpha"]:
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 < cfg["participation"] <= 1 or not 0 <= cfg["kappa"] <= 1:
        raise ValueError("Invalid participation or kappa")
    if cfg["model"] not in {
        "clip-vit-b32",
        "clip-vit-l14",
        "dinov2-base",
        "dinov2-small",
        "convmixer-768-32",
    }:
        raise ValueError("model must be one of the paper backbones")
    if type(cfg["pretrained"]) is not bool:
        raise ValueError("pretrained must be a boolean")
    if type(cfg["torch-threads"]) is not int or cfg["torch-threads"] < 1:
        raise ValueError("torch-threads must be a positive integer")
    if not cfg["pretrained"] and cfg["model-checkpoint"]:
        raise ValueError(
            "pretrained=false initializes from scratch; omit model-checkpoint"
        )
    if cfg["seed"] < 0 or cfg["seed"] >= 2**32:
        raise ValueError("seed must be a uint32")
    return cfg


def kappa_at(cfg, round_id):
    # Scheduler horizon (R-1)//16, clamped at its endpoint instead of cycling.
    horizon = max(1, (cfg["num-rounds"] - 1) // 16)
    fraction = min(1.0, max(0.0, (round_id - 1) / horizon))
    return 1 - (1 - cfg["kappa"]) * 0.5 * (1 + math.cos(math.pi * fraction))

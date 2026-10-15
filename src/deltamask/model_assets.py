"""Pinned model acquisition, offline resolution and checkpoint identity."""

import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

MODELS = {
    "clip-vit-b32": {
        "repo": "openai/clip-vit-base-patch32", "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        "weight": "pytorch_model.bin", "sha256": "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f",
    },
    "clip-vit-l14": {
        "repo": "openai/clip-vit-large-patch14", "revision": "32bd64288804d66eefd0ccbe215aa642df71cc41",
        "weight": "model.safetensors", "sha256": "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb",
    },
    "dinov2-base": {
        "repo": "facebook/dinov2-base", "revision": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
        "weight": "model.safetensors", "sha256": "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841",
    },
    "dinov2-small": {
        "repo": "facebook/dinov2-small", "revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
        "weight": "model.safetensors", "sha256": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
    },
    "convmixer-768-32": {
        "repo": "timm/convmixer_768_32.in1k", "revision": "09fa91bbf9d93bf517098727fdd0ff580e69e035",
        "weight": "model.safetensors", "sha256": "62ed5ec91004012a90d1a1ad043c4a74724ad444c82a277a6fff10803391743b",
    },
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify(name, directory):
    spec = MODELS[name]
    alternatives = [p.name for p in directory.iterdir() if p.is_file() and (p.suffix in (".safetensors", ".bin", ".pth") or p.name.endswith(".index.json")) and p.name != spec["weight"]]
    if alternatives:
        raise ValueError(f"Unexpected weights beside managed checkpoint: {alternatives}")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("model") != name or manifest.get("revision") != spec["revision"]:
        raise ValueError("Model manifest does not match pinned revision")
    for filename in ("config.json", spec["weight"]):
        actual = sha256(directory / filename)
        expected = spec["sha256"] if filename == spec["weight"] else manifest["files"][filename]
        if actual != expected or manifest["files"].get(filename) != actual:
            raise ValueError(f"Model checksum mismatch: {directory / filename}")
    return manifest


def prepare_model(name="clip-vit-b32", root="checkpoints"):
    spec = MODELS[name]
    directory = Path(root).expanduser().resolve() / name
    if (directory / "manifest.json").is_file():
        return _verify(name, directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for filename in ("config.json", spec["weight"]):
        print(f"{name}: {filename}", flush=True)
        path = hf_hub_download(spec["repo"], filename, revision=spec["revision"], local_dir=directory)
        files[filename] = sha256(path)
    if files[spec["weight"]] != spec["sha256"]:
        raise ValueError(f"Model checksum mismatch: {directory / spec['weight']}")
    manifest = {"schema_version": 1, "model": name, "source": spec["repo"], "revision": spec["revision"], "files": files}
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(directory / "manifest.json")
    return manifest


def resolve_model(cfg):
    cfg = dict(cfg)
    if not cfg.get("pretrained", True):
        return cfg
    name = cfg["model"]
    spec = MODELS[name]
    checkpoint = cfg.get("model-checkpoint", "")
    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        directory = path if path.is_dir() else path.parent
        if (directory / "manifest.json").is_file() and (path.is_dir() or path.name == spec["weight"]):
            manifest = _verify(name, directory)
            cfg["model-revision"] = manifest["revision"]
        cfg["model-checkpoint"] = str(path / spec["weight"] if name == "convmixer-768-32" and path.is_dir() else path)
        return cfg
    revision = cfg.get("model-revision", "")
    if revision and revision != spec["revision"]:
        raise ValueError("Use an explicit model-checkpoint for a different model revision")
    directory = Path(cfg.get("model-root", "checkpoints")).expanduser().resolve() / name
    if not (directory / "manifest.json").is_file():
        if cfg.get("allow-download", False):
            prepare_model(name, directory.parent)
        else:
            raise FileNotFoundError(f"Prepare weights: python scripts/prepare_model.py --model {name} --root '{directory.parent}'")
    manifest = _verify(name, directory)
    cfg["model-revision"] = manifest["revision"]
    cfg["model-checkpoint"] = str(directory / spec["weight"] if name == "convmixer-768-32" else directory)
    return cfg


def model_identity(cfg):
    if not cfg.get("pretrained", True):
        return {"model": cfg["model"], "initialization": "random", "seed": cfg.get("seed", 42)}
    resolved = resolve_model(cfg)
    path = Path(resolved["model-checkpoint"])
    directory = path if path.is_dir() else path.parent
    if (directory / "manifest.json").is_file() and (path.is_dir() or path.name == MODELS[cfg["model"]]["weight"]):
        return _verify(cfg["model"], directory)
    if path.is_file():
        files = {path.name: sha256(path)}
    else:
        paths = sorted(p for p in path.iterdir() if p.name == "config.json" or p.suffix in (".safetensors", ".bin", ".pth", ".json"))
        if not paths:
            raise ValueError("Local model directory has no configuration or weights")
        files = {p.name: sha256(p) for p in paths if p.is_file()}
    return {"model": cfg["model"], "source": "local", "files": files}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS) + ("all",), default="clip-vit-b32")
    parser.add_argument("--root", type=Path, default=Path("checkpoints"))
    args = parser.parse_args(argv)
    for name in MODELS if args.model == "all" else (args.model,):
        try:
            prepare_model(name, args.root)
        except (OSError, ValueError, RuntimeError) as exc:
            parser.exit(2, f"Model preparation failed: {exc}\n")
        print(f"Ready: {args.root / name}", flush=True)

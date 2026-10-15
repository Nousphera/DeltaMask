"""Restore a saved adaptation checkpoint and rerun its final test evaluation.

This command evaluates only. Optimizer and Bayesian accumulator state are not
saved, so these artifacts do not implement training resume.
"""

import argparse
import json
from pathlib import Path
import zipfile
import numpy as np
import torch
from .config import defaults, validate, MASK_METHODS
from .model import make_model, mask_scores, set_mask_scores, evaluate
from .baselines import parameter_state, set_parameters, validate_state
from .data import load_data


def restore_checkpoint(model, path, result):
    cfg = result["config"]
    masked = cfg["method"] in MASK_METHODS
    model.configure_method(cfg["method"])
    if masked:
        head = model.head.layer.state_dict()
        expected = {"scores": (sum(p.numel() for n, p in model.named_parameters()
                                   if n.endswith("_mask")),)} | {
                   "head." + n: tuple(v.shape) for n, v in head.items()}
    else:
        parameters = parameter_state(model)
        expected = {"parameter." + n: tuple(v.shape) for n, v in parameters.items()}
    limit = sum(int(np.prod(shape)) * 4 for shape in expected.values()) + 4096 * (len(expected) + 1)
    try:
        with zipfile.ZipFile(path) as archive:
            if (len(archive.infolist()) != len(expected) + 1
                    or sum(item.file_size for item in archive.infolist()) > limit):
                raise ValueError("Invalid checkpoint archive size")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected) | {"metadata"}:
                raise ValueError("Checkpoint names differ")
            raw_meta = archive["metadata"]
            if raw_meta.shape != () or raw_meta.dtype.kind != "U":
                raise ValueError("Invalid checkpoint metadata")
            metadata = json.loads(str(raw_meta))
            wanted = {"version": 1, "method": cfg["method"], "model": cfg["model"],
                      "classes": result["classes"], "evaluation_seed": result["evaluation_seed"],
                      "model_identity": result["model_identity"]}
            if metadata != wanted:
                raise ValueError("Checkpoint metadata differs from result")
            state = {}
            for name, shape in expected.items():
                value = archive[name]
                if value.shape != shape or value.dtype != np.dtype("float32") or not np.isfinite(value).all():
                    raise ValueError("Checkpoint shape, dtype or finiteness mismatch")
                state[name] = value.copy()
    except (OSError, EOFError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        raise ValueError("Malformed checkpoint") from exc
    # No learned values are applied until the entire checkpoint is validated.
    if masked:
        head_state = {n: torch.from_numpy(state["head." + n]) for n in head}
        validate_state(head_state, head)
        set_mask_scores(model, state["scores"])
        model.head.layer.load_state_dict(head_state)
    else:
        set_parameters(model, {n: torch.from_numpy(state["parameter." + n]) for n in parameters})


def evaluate_result(result_path, *, checkpoint_path=None, device=None):
    from .model_assets import model_identity
    from .server_app import file_sha256
    from .dataset_io import manifest_digest, validate_dataset

    path = Path(result_path).expanduser().resolve()
    result = json.loads(path.read_text())
    if result.get("schema_version") != 1 or not result.get("rounds"):
        raise ValueError("Unsupported result format")
    cfg = validate(defaults() | result["config"])
    if result["method"] != cfg["method"]:
        raise ValueError("Result method mismatch")
    expected_seed = cfg["seed"] + 10000 + cfg["num-rounds"]
    if result["evaluation_seed"] != expected_seed:
        raise ValueError("Result evaluation seed mismatch")
    checkpoint = Path(checkpoint_path) if checkpoint_path else path.with_suffix(".npz")
    if file_sha256(checkpoint) != result["checkpoint_sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    if device is not None:
        cfg["device"] = device
    if cfg["dataset"] != "synthetic-images":
        # A manifest digest alone cannot detect replaced image contents.
        validate_dataset(cfg["dataset"], cfg["data-root"])
    if model_identity(cfg) != result["model_identity"]:
        raise ValueError("Base model identity differs from the saved run")
    if manifest_digest(cfg["dataset"], cfg["data-root"]) != result["data_manifest_sha256"]:
        raise ValueError("Dataset manifest differs from the saved run")
    _, test, classes = load_data(cfg)
    if classes != result["classes"]:
        raise ValueError("Class count differs from checkpoint")
    model = make_model(cfg, classes)
    restore_checkpoint(model, checkpoint, result)
    return {"method": cfg["method"], "checkpoint": str(checkpoint),
            "evaluation_seed": expected_seed, **evaluate(model, test, cfg, expected_seed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()
    print(json.dumps(evaluate_result(args.result, checkpoint_path=args.checkpoint, device=args.device), indent=2))


if __name__ == "__main__":
    main()

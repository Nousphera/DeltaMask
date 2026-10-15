"""Flower orchestration shared by DeltaMask and the supported baselines."""

import hashlib
import json
import math
import platform
import time
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit, logit
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.serverapp import ServerApp, Grid
from .aggregate import BayesianMask
from .baselines import WeightedMean, parameter_state, set_parameters, reply_metrics
from .codec import decode, decode_binary, decode_parameters, encode_parameters
from .config import defaults, validate, kappa_at, MASK_METHODS, METHODS
from .data import load_data
from .model import make_model, mask_scores, set_mask_scores, evaluate

app = ServerApp()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = validate(defaults() | context.run_config)
    path = Path(cfg["output"]).expanduser().resolve()
    cfg["output"] = str(path)
    checkpoint, updates = path.with_suffix(".npz"), path.with_suffix(".updates")
    if path.exists() or checkpoint.exists() or (cfg["save-updates"] and updates.exists()):
        raise FileExistsError("Output exists; choose a new output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    result = run_server(grid, cfg, checkpoint_path=checkpoint,
                        update_dir=updates if cfg["save-updates"] else None)
    with path.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(path), "method": cfg["method"], "rounds": result["rounds"]}, indent=2))


def wait_for_nodes(grid, count, timeout, poll_interval=.25):
    deadline = time.monotonic() + timeout
    while True:
        raw = list(grid.get_node_ids())
        if len(raw) != len(set(raw)) or any(type(n) != int or n < 0 for n in raw):
            raise ValueError("Invalid Flower node IDs")
        nodes = sorted(raw)
        if len(nodes) == count:
            return nodes
        if len(nodes) > count:
            raise ValueError("Number of Flower SuperNodes exceeds num-clients")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Only {len(nodes)}/{count} Flower SuperNodes registered")
        time.sleep(min(poll_interval, remaining))


def checked_replies(grid, messages, timeout=3600):
    """Validate and consume replies incrementally; avoid retaining client models."""
    expected = {m.metadata.dst_node_id: m for m in messages}
    if len(expected) != len(messages):
        raise ValueError("Duplicate destinations")
    seen = set()
    for reply in grid.send_and_receive(messages, timeout=timeout):
        node = reply.metadata.src_node_id
        if node not in expected or node in seen or reply.has_error():
            raise RuntimeError("Round requires one successful reply from every selected client")
        if reply.metadata.group_id != expected[node].metadata.group_id:
            raise ValueError("Reply round mismatch")
        seen.add(node)
        yield reply
    if seen != set(expected):
        raise RuntimeError("Round requires one successful reply from every selected client")


def _save_update(directory, round_id, cid, wire):
    if directory is None:
        return
    folder = Path(directory) / f"round-{round_id:04d}"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"client-{cid:04d}.bin").open("xb") as stream:
        stream.write(wire)


def _common_dimension(model):
    """Method-independent denominator: all final-five Linear/Conv parameters.

    ConvMixer fine-tuning adapts depthwise and pointwise convolutions; masking
    adapts pointwise only. The common denominator includes both, while the
    separate mask_dimension records the actual transmitted mask layout.
    """
    modules = [m for b in model.blocks for m in b.modules()
               if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d))] + [model.head]
    if model.kind == "clip":
        modules.append(model.backbone.visual_projection)
    return sum(p.numel() for m in modules for p in m.parameters(recurse=False))


def run_server(grid, cfg, checkpoint_path=None, update_dir=None):
    from .model_assets import resolve_model, model_identity
    from .dataset_io import manifest_digest, validate_dataset

    cfg = validate(defaults() | cfg)
    cfg["data-root"] = str(Path(cfg["data-root"]).expanduser().resolve())
    if cfg["dataset"] != "synthetic-images":
        # Verify content once at experiment entry, not on each client or round.
        validate_dataset(cfg["dataset"], cfg["data-root"])
    cfg = resolve_model(cfg)
    nodes = wait_for_nodes(grid, cfg["num-clients"], cfg["registration-timeout"])
    _, test, classes = load_data(cfg)
    model = make_model(cfg, classes)
    identity = model_identity(cfg)
    common_dimension = _common_dimension(model)
    head = ArrayRecord(model.head.state_dict())
    masked = cfg["method"] in MASK_METHODS
    if update_dir is not None:
        Path(update_dir).mkdir(parents=True, exist_ok=False)

    def message(node, phase, rnd, scores=None, parameters=None):
        records = {"config": ConfigRecord({"phase": phase, "round": rnd,
                   "method": cfg["method"], "kappa": kappa_at(cfg, rnd)})}
        if phase == "probe" or masked:
            records["head"] = head
        if scores is not None:
            records["scores"] = ArrayRecord([scores])
        if parameters is not None:
            records["parameters"] = parameters
        return Message(RecordDict(records), dst_node_id=node, message_type="train", group_id=str(rnd))

    # Every method begins with the same one-round weighted federated head probe.
    messages = [message(node, "probe", 0) for node in nodes]
    mean = WeightedMean(model.head.state_dict())
    mapping, probe_bytes, probe_loss, probe_samples = {}, 0, 0., 0
    for reply in checked_replies(grid, messages, cfg["reply-timeout"]):
        cid, count, loss = reply_metrics(reply, num_clients=cfg["num-clients"])
        if cid in mapping:
            raise ValueError("Client partition IDs must be unique and complete")
        mapping[cid] = reply.metadata.src_node_id
        state = reply.content["head"].to_torch_state_dict()
        mean.add(state, count)
        probe_bytes += reply.content["head"].count_bytes()
        probe_loss += loss * count
        probe_samples += count
        _save_update(update_dir, 0, cid, encode_parameters(state))
    if set(mapping) != set(range(cfg["num-clients"])):
        raise ValueError("Client partition IDs must be unique and complete")
    model.head.load_state_dict(mean.result())
    probe_state = {n: v.detach().cpu().clone() for n, v in model.head.state_dict().items()}
    head = ArrayRecord(probe_state)
    del mean, state, reply, messages
    node_to_client = {node: cid for cid, node in mapping.items()}
    model.configure_method(cfg["method"])
    scores = mask_scores(model).astype(np.float32) if masked else None
    if masked:
        aggregator = BayesianMask(len(scores), cfg["participation"], cfg["mask-epsilon"], cfg["numerical-epsilon"])
    rng = np.random.default_rng(cfg["seed"])
    history = []
    for rnd in range(1, cfg["num-rounds"] + 1):
        selected_ids = sorted(rng.choice(cfg["num-clients"],
                              max(1, math.floor(cfg["num-clients"] * cfg["participation"])),
                              replace=False).tolist())
        parameters = None if masked else ArrayRecord(parameter_state(model))
        messages = [message(mapping[cid], "train", rnd, scores, parameters) for cid in selected_ids]
        if masked:
            q = expit(scores.astype(np.float64))
            ones = np.zeros(len(scores), dtype=np.uint32)
        else:
            mean = WeightedMean(parameter_state(model))
        payload_bytes, png_bytes, loss_total, num_examples = 0, 0, 0., 0
        clients = []
        for reply in checked_replies(grid, messages, cfg["reply-timeout"]):
            cid, count, loss = reply_metrics(reply, num_clients=cfg["num-clients"],
                                            expected_id=node_to_client[reply.metadata.src_node_id])
            if reply.content.array_records:
                raise ValueError("Training reply must contain serialized update bytes")
            wire = reply.content["update"]["payload"]
            if not isinstance(wire, bytes) or reply.content["metrics"]["payload_bytes"] != len(wire):
                raise ValueError("Invalid update payload or byte count")
            if cfg["method"] == "deltamask":
                decoded = decode(wire, q, seed=cfg["seed"], round_id=rnd)
                ones += decoded
            elif masked:
                decoded = decode_binary(wire, len(scores), seed=cfg["seed"], round_id=rnd, client_id=cid)
                ones += decoded
            else:
                decoded = decode_parameters(wire, mean.expected)
                mean.add(decoded, count)
            png = reply.content["metrics"]["png_bytes"]
            if type(png) != int or not 0 <= png <= len(wire):
                raise ValueError("Invalid PNG byte count")
            _save_update(update_dir, rnd, cid, wire)
            payload_bytes += len(wire)
            png_bytes += png
            loss_total += loss * count
            num_examples += count
            clients.append({"client_id": cid, "num_examples": count, "loss": loss,
                            "payload_bytes": len(wire), "sha256": hashlib.sha256(wire).hexdigest()})
            del decoded
        if masked:
            scores = logit(aggregator.update_counts(ones, len(selected_ids), rnd)).astype(np.float32)
            set_mask_scores(model, scores)
            del ones, q
        else:
            set_parameters(model, mean.result())
            del mean
        metrics = evaluate(model, test, cfg, cfg["seed"] + 10000 + rnd)
        if not all(math.isfinite(v) for v in metrics.values()):
            raise ValueError("Nonfinite evaluation metrics")
        history.append({
            "round": rnd, "clients": len(selected_ids), "client_updates": clients,
            "kappa": kappa_at(cfg, rnd) if cfg["method"] == "deltamask" else None,
            "train_loss": loss_total / num_examples, "train_examples": num_examples,
            "payload_bytes": payload_bytes, "png_bytes": png_bytes,
            "uplink_bpp_per_client": 8 * payload_bytes / (len(selected_ids) * common_dimension),
            "head_downlink_bytes": len(selected_ids) * head.count_bytes() if masked else 0,
            "score_downlink_bytes": sum(m.content["scores"].count_bytes() for m in messages) if masked else 0,
            "parameter_downlink_bytes": len(selected_ids) * parameters.count_bytes() if parameters else 0,
            **metrics,
        })
        print(json.dumps({"method": cfg["method"], "round": rnd, "loss": metrics["loss"],
                          "accuracy": metrics["accuracy"], "uplink_bytes": payload_bytes}), flush=True)
        del messages, reply, wire, parameters
    result = {
        "schema_version": 1, "method": cfg["method"], "method_description": METHODS[cfg["method"]],
        "data_manifest_sha256": manifest_digest(cfg["dataset"], cfg["data-root"]),
        "model_identity": identity,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "flower": version("flwr"),
                        "transformers": version("transformers"), "timm": version("timm")},
        "config": cfg, "classes": classes, "communication_dimension": common_dimension,
        "communication_dimension_definition": "All Linear/Conv parameters in the last five blocks, plus CLIP projection and classifier; shared across methods.",
        "mask_dimension": len(scores) if masked else 0,
        "trained_parameter_dimension": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "probe_uplink_bytes": probe_bytes, "probe_head_bytes": head.count_bytes(),
        "probe_train_loss": probe_loss / probe_samples, "rounds": history,
        "evaluation_seed": cfg["seed"] + 10000 + cfg["num-rounds"],
        "client_nodes": {str(cid): str(node) for cid, node in mapping.items()},
        "byte_accounting": "Serialized method payloads / Flower ArrayRecord bytes; excludes network and Flower metadata overhead.",
        "transport": "four-wise-bfuse8-png-shared-client-rng" if cfg["method"] == "deltamask" else
                     "lossless-packbits-independent-client-rng" if masked else "lossless-float32-npz",
    }
    if masked:
        result["final_score_sha256"] = hashlib.sha256(scores.astype("<f4").tobytes()).hexdigest()
    if checkpoint_path is not None:
        checkpoint = {"scores": scores, **{"head." + n: v.numpy() for n, v in probe_state.items()}} if masked else {
            "parameter." + n: v.numpy() for n, v in parameter_state(model).items()}
        metadata = {"version": 1, "method": cfg["method"], "model": cfg["model"],
                    "classes": classes, "evaluation_seed": result["evaluation_seed"], "model_identity": identity}
        checkpoint["metadata"] = np.array(json.dumps(metadata, sort_keys=True))
        with Path(checkpoint_path).open("xb") as stream:
            np.savez_compressed(stream, **checkpoint)
        result["checkpoint"] = str(Path(checkpoint_path).resolve())
        result["checkpoint_sha256"] = file_sha256(checkpoint_path)
    return result

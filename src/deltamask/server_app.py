"""Custom Flower server loop: decode mask bytes and apply Beta MAP aggregation."""

import hashlib
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit, logit
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.serverapp import ServerApp, Grid
from .aggregate import BayesianMask
from .codec import decode
from .config import defaults, validate, kappa_at
from .data import load_data
from .model import make_model, mask_scores, set_mask_scores, evaluate

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = validate(defaults() | context.run_config)
    path = Path(cfg["output"])
    checkpoint = path.with_suffix(".npz")
    if path.exists() or checkpoint.exists():
        raise FileExistsError("Output exists; choose a new output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    result = run_server(grid, cfg, checkpoint_path=checkpoint)
    with path.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(path), "rounds": result["rounds"]}, indent=2))


def checked_replies(grid, messages):
    replies = list(grid.send_and_receive(messages, timeout=3600))
    expected = {m.metadata.dst_node_id for m in messages}
    actual = [r.metadata.src_node_id for r in replies]
    if (
        len(actual) != len(expected)
        or set(actual) != expected
        or any(r.has_error() for r in replies)
    ):
        raise RuntimeError(
            "Round requires one successful reply from every selected client"
        )
    return sorted(replies, key=lambda r: r.metadata.src_node_id)


def run_server(grid, cfg, checkpoint_path=None):
    nodes = sorted(grid.get_node_ids())
    if len(nodes) != cfg["num-clients"]:
        raise ValueError("Number of Flower SuperNodes must match num-clients")
    _, test, classes = load_data(cfg)
    model = make_model(cfg, classes)
    head = ArrayRecord(model.head.state_dict())

    def message(node, phase, rnd, scores=None, kappa=1.0):
        records = {
            "head": head,
            "config": ConfigRecord({"phase": phase, "round": rnd, "kappa": kappa}),
        }
        if scores is not None:
            records["scores"] = ArrayRecord([scores])
        return Message(
            RecordDict(records),
            dst_node_id=node,
            message_type="train",
            group_id=str(rnd),
        )

    # Single federated linear-probing round. Only classifier weights are averaged.
    replies = checked_replies(grid, [message(node, "probe", 0) for node in nodes])
    identities = [r.content["metrics"]["client-id"] for r in replies]
    if sorted(identities) != list(range(cfg["num-clients"])):
        raise ValueError("Client partition IDs must be unique and complete")
    replies.sort(key=lambda r: r.content["metrics"]["client-id"])
    nodes = [r.metadata.src_node_id for r in replies]
    sizes = [r.content["metrics"]["num-examples"] for r in replies]
    heads = [r.content["head"].to_torch_state_dict() for r in replies]
    state = {
        name: sum(h[name] * (n / sum(sizes)) for h, n in zip(heads, sizes))
        for name in heads[0]
    }
    model.head.load_state_dict(state)
    head = ArrayRecord(state)
    probe_bytes = sum(r.content["head"].count_bytes() for r in replies)
    model.enable_masks()
    scores = mask_scores(model).astype(np.float32)
    aggregator = BayesianMask(len(scores), cfg["participation"])
    rng = np.random.default_rng(cfg["seed"])
    history = []
    for rnd in range(1, cfg["num-rounds"] + 1):
        selected = sorted(
            nodes[i]
            for i in rng.choice(
                len(nodes),
                max(1, math.floor(len(nodes) * cfg["participation"])),
                replace=False,
            )
        )
        messages = [
            message(node, "mask", rnd, scores, kappa_at(cfg, rnd)) for node in selected
        ]
        replies = checked_replies(grid, messages)
        replies.sort(key=lambda r: r.content["metrics"]["client-id"])
        q = expit(scores.astype(np.float64))
        masks = []
        payload_bytes = 0
        png_bytes = 0
        for reply in replies:
            if reply.content.array_records:
                raise ValueError(
                    "Mask training reply must contain encoded bytes, not score arrays"
                )
            wire = reply.content["update"]["payload"]
            masks.append(decode(wire, q, seed=cfg["seed"], round_id=rnd))
            payload_bytes += len(wire)
            png_bytes += int(reply.content["metrics"]["png_bytes"])
        scores = logit(aggregator.update(masks, rnd)).astype(np.float32)
        set_mask_scores(model, scores)
        metrics = evaluate(model, test, cfg, cfg["seed"] + 10000 + rnd)
        history.append(
            {
                "round": rnd,
                "clients": len(selected),
                "kappa": kappa_at(cfg, rnd),
                "payload_bytes": payload_bytes,
                "png_bytes": png_bytes,
                "uplink_bpp_per_client": 8
                * payload_bytes
                / (len(selected) * len(scores)),
                "head_downlink_bytes": len(selected) * head.count_bytes(),
                "score_downlink_bytes": sum(
                    m.content["scores"].count_bytes() for m in messages
                ),
                **metrics,
            }
        )
    if checkpoint_path is not None:
        with Path(checkpoint_path).open("xb") as stream:
            np.savez_compressed(
                stream,
                scores=scores,
                **{"head." + n: v.detach().cpu().numpy() for n, v in state.items()},
            )
    return {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "flower": version("flwr"),
        },
        "config": cfg,
        "mask_dimension": len(scores),
        "probe_uplink_bytes": probe_bytes,
        "probe_head_bytes": head.count_bytes(),
        "rounds": history,
        "final_score_sha256": hashlib.sha256(
            scores.astype("<f4").tobytes()
        ).hexdigest(),
    }

"""Flower 1.40 ClientApp: clients alone train and encode their mask updates."""

import numpy as np
from scipy.special import expit
from torch.utils.data import Subset
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.clientapp import ClientApp
from .config import defaults, validate
from .data import load_data, partitions, targets
from .model import make_model, probe, train_local, mask_scores, set_mask_scores
from .codec import encode

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    cfg = validate(defaults() | context.run_config)
    cid = int(context.node_config["partition-id"])
    if not 0 <= cid < cfg["num-clients"]:
        raise ValueError("Client partition does not match experiment")
    dataset, _, classes = load_data(cfg)
    splits = partitions(targets(dataset), cfg["num-clients"], cfg["alpha"], cfg["seed"])
    local = Subset(dataset, splits[cid].tolist())
    model = make_model(cfg, classes)
    model.head.load_state_dict(msg.content["head"].to_torch_state_dict())
    config = msg.content["config"]
    round_id = int(config["round"])
    seed = cfg["seed"] + round_id * cfg["num-clients"] + cid
    if config["phase"] == "probe":
        loss = probe(model, local, cfg, seed)
        content = RecordDict(
            {
                "head": ArrayRecord(model.head.state_dict()),
                "metrics": MetricRecord(
                    {"num-examples": len(local), "client-id": cid, "loss": loss}
                ),
            }
        )
    elif config["phase"] == "mask":
        model.enable_masks()
        global_scores = msg.content["scores"].to_numpy_ndarrays()[0]
        set_mask_scores(model, global_scores)
        loss = train_local(model, local, cfg, seed)
        # Downlink scores are transmitted float32; both sides canonicalize through float64 sigmoid.
        q = expit(global_scores.astype(np.float64))
        p = expit(mask_scores(model).astype(np.float64))
        wire, stats = encode(
            p, q, seed=cfg["seed"], round_id=round_id, kappa=float(config["kappa"])
        )
        content = RecordDict(
            {
                "update": ConfigRecord({"payload": wire}),
                "metrics": MetricRecord(
                    {
                        "num-examples": len(local),
                        "client-id": cid,
                        "loss": loss,
                        **stats,
                    }
                ),
            }
        )
    else:
        raise ValueError("Unknown training phase")
    return Message(content=content, reply_to=msg)

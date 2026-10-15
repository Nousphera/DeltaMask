"""One Flower lifecycle for all methods; clients train and serialize updates."""

import math
import numpy as np
from scipy.special import expit
from torch.utils.data import Subset
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from .config import defaults, validate, MASK_METHODS
from .data import load_data, partitions, targets
from .model import make_model, probe, train_local, mask_scores, set_mask_scores
from .baselines import parameter_state, set_parameters, validate_state
from .codec import encode, encode_binary, sample_client_mask, encode_parameters

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    cfg = validate(defaults() | context.run_config)
    cid = context.node_config["partition-id"]
    if type(cid) != int or not 0 <= cid < cfg["num-clients"]:
        raise ValueError("Client partition does not match experiment")
    config = msg.content["config"]
    phase, round_id = config["phase"], config["round"]
    if (phase not in {"probe", "train"} or type(round_id) != int
            or (phase == "probe" and round_id != 0)
            or (phase == "train" and not 1 <= round_id <= cfg["num-rounds"])
            or config["method"] != cfg["method"]):
        raise ValueError("Invalid training message context")
    if not math.isfinite(config["kappa"]) or not 0 <= config["kappa"] <= 1:
        raise ValueError("Invalid kappa")
    dataset, _, classes = load_data(cfg)
    splits = partitions(targets(dataset), cfg["num-clients"], cfg["alpha"], cfg["seed"])
    local = Subset(dataset, splits[cid].tolist())
    model = make_model(cfg, classes)
    seed = cfg["seed"] + round_id * cfg["num-clients"] + cid
    if phase == "probe" or cfg["method"] in MASK_METHODS:
        head = msg.content["head"].to_torch_state_dict()
        validate_state(head, model.head.state_dict())
        model.head.load_state_dict(head)
    if phase == "probe":
        loss = probe(model, local, cfg, seed)
        content = RecordDict({
            "head": ArrayRecord(model.head.state_dict()),
            "metrics": MetricRecord({"num-examples": len(local), "client-id": cid, "loss": loss}),
        })
    else:
        model.configure_method(cfg["method"])
        if cfg["method"] in MASK_METHODS:
            arrays = msg.content["scores"].to_numpy_ndarrays()
            if len(arrays) != 1:
                raise ValueError("Expected one flat score array")
            global_scores = arrays[0]
            set_mask_scores(model, global_scores)
        else:
            set_parameters(model, msg.content["parameters"].to_torch_state_dict())
        loss = train_local(model, local, cfg, seed)
        if cfg["method"] in MASK_METHODS:
            p = expit(mask_scores(model).astype(np.float64))
            if cfg["method"] == "deltamask":
                q = expit(global_scores.astype(np.float64))
                wire, stats = encode(p, q, seed=cfg["seed"], round_id=round_id,
                                     kappa=float(config["kappa"]))
            else:
                # This FedMask variant uses recovered threshold training plus
                # stochastic Bayesian transport, not all original-paper mechanisms.
                mask = sample_client_mask(p, cfg["seed"], round_id, cid)
                wire, stats = encode_binary(mask, seed=cfg["seed"], round_id=round_id, client_id=cid)
        else:
            wire = encode_parameters(parameter_state(model))
            stats = {"payload_bytes": len(wire), "png_bytes": 0}
        content = RecordDict({
            "update": ConfigRecord({"payload": wire}),
            "metrics": MetricRecord({"num-examples": len(local), "client-id": cid,
                                     "loss": loss, **stats}),
        })
    return Message(content=content, reply_to=msg)

"""Offline runner using the same Flower Message, ClientApp and ServerApp handlers."""

import argparse
import json
from pathlib import Path
import tomllib
import torch
from flwr.app import Context, RecordDict
from flwr.supercore.task_identity import TaskIdentity
from .client_app import app as client_app
from .server_app import app as server_app
from .config import defaults, validate


class LocalGrid:
    """Bounded in-process transport; flwr run uses Flower's real simulation backend."""

    def __init__(self, cfg):
        self.cfg = cfg

    def get_node_ids(self):
        return range(1, self.cfg["num-clients"] + 1)

    def send_and_receive(self, messages, timeout=None):
        for message in messages:
            cid = message.metadata.dst_node_id - 1
            context = Context(
                run_id=1,
                node_id=cid + 1,
                node_config={"partition-id": cid},
                state=RecordDict(),
                run_config=self.cfg,
            )
            yield client_app(message, context)


def run(cfg):
    cfg = validate(defaults() | cfg)
    torch.set_num_threads(1)
    context = Context(
        run_id=1, node_id=0, node_config={}, state=RecordDict(), run_config=cfg
    )
    previous = (TaskIdentity._run_id, TaskIdentity._node_id, TaskIdentity._task_id)
    try:
        TaskIdentity.run_id = 1
        TaskIdentity.node_id = 0
        TaskIdentity.task_id = 0
        server_app(LocalGrid(cfg), context)
    finally:
        TaskIdentity._run_id, TaskIdentity._node_id, TaskIdentity._task_id = previous
    return json.loads(Path(cfg["output"]).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--clients", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Permit pretrained model downloads; prepare datasets separately",
    )
    args = parser.parse_args()
    cfg = defaults()
    if args.config:
        cfg.update(tomllib.loads(args.config.read_text()))
    for key, value in [
        ("num-rounds", args.rounds),
        ("num-clients", args.clients),
        ("seed", args.seed),
        ("output", args.output),
    ]:
        if value is not None:
            cfg[key] = value
    if args.download:
        cfg["allow-download"] = True
    run(cfg)


if __name__ == "__main__":
    main()

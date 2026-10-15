"""Shared parameter selection and streamed weighted FedAvg."""

import math
import torch


def parameter_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def validate_state(values, expected):
    if set(values) != set(expected):
        raise ValueError("Parameter names differ")
    for name, target in expected.items():
        value = values[name]
        if (not isinstance(value, torch.Tensor) or value.shape != target.shape
                or value.dtype != target.dtype or not torch.isfinite(value).all()):
            raise ValueError("Parameter shape, dtype or finiteness mismatch")


def set_parameters(model, values):
    expected = {n: p for n, p in model.named_parameters() if p.requires_grad}
    validate_state(values, expected)
    with torch.no_grad():
        for name, target in expected.items():
            target.copy_(values[name].to(target.device))


class WeightedMean:
    """Retain only one accumulator, never a list of client model copies."""

    def __init__(self, expected):
        self.expected = expected
        self.sums = None
        self.total = 0

    def add(self, values, count):
        validate_state(values, self.expected)
        if type(count) != int or count < 1:
            raise ValueError("Invalid client sample count")
        if self.sums is None:
            self.sums = {n: torch.zeros_like(v, dtype=torch.float64, device="cpu")
                         for n, v in values.items()}
        for name, value in values.items():
            self.sums[name].add_(value.detach().to(device="cpu", dtype=torch.float64), alpha=count)
        self.total += count

    def result(self):
        if self.sums is None:
            raise ValueError("Empty aggregation")
        return {n: (value / self.total).to(self.expected[n].dtype) for n, value in self.sums.items()}


def reply_metrics(reply, *, num_clients, expected_id=None):
    metrics = reply.content["metrics"]
    cid, count, loss = metrics["client-id"], metrics["num-examples"], metrics["loss"]
    if (type(cid) != int or not 0 <= cid < num_clients
            or (expected_id is not None and cid != expected_id)
            or type(count) != int or count < 1
            or not isinstance(loss, (float, int)) or not math.isfinite(loss)):
        raise ValueError("Invalid client identity, sample count or loss")
    return cid, count, loss

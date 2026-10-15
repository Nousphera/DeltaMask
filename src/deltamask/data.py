"""Seeded data adapters. Downloads require an explicit configuration opt-in."""

import numpy as np
import torch
from torch.utils.data import TensorDataset, Subset


def partitions(labels, num_clients, alpha, seed):
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    if labels.ndim != 1 or len(labels) < num_clients or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Insufficient samples or invalid alpha")
    result = [[] for _ in range(num_clients)]
    for label in np.unique(labels):
        indexes = np.flatnonzero(labels == label)
        rng.shuffle(indexes)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        proportions *= np.asarray([len(part) < len(labels) / num_clients for part in result])
        proportions /= proportions.sum()
        cuts = (np.cumsum(proportions) * len(indexes)).astype(int)[:-1]
        result = [part + values.tolist() for part, values in zip(result, np.split(indexes, cuts))]
    # The recovered partitioner can leave empty clients on very small datasets.
    # Deterministically repair only that degenerate case so every selected client trains.
    for part in result:
        if not part:
            donor = max(range(num_clients), key=lambda j: len(result[j]))
            part.append(result[donor].pop())
    for part in result:
        rng.shuffle(part)
    return [np.asarray(part, dtype=np.int64) for part in result]


def targets(ds):
    if isinstance(ds, TensorDataset):
        return ds.tensors[1].numpy()
    if isinstance(ds, Subset):
        return targets(ds.dataset)[ds.indices]
    if hasattr(ds, "targets"):
        return np.asarray(ds.targets)
    if hasattr(ds, "labels"):
        return np.asarray(ds.labels)
    if hasattr(ds, "_labels"):
        return np.asarray(ds._labels)
    return np.asarray([y for _, y in ds.samples])


def load_data(cfg):
    name = cfg["dataset"]
    seed = cfg["seed"]
    if name == "synthetic-images":
        # Explicit offline execution fixture: full-resolution images, never vector inputs.
        generator = torch.Generator().manual_seed(seed)

        def generate(n):
            x = torch.randn(n, 3, 224, 224, generator=generator)
            y = torch.arange(n) % 3
            return TensorDataset(x, y)

        return generate(cfg["num-clients"] * cfg["samples-per-client"]), generate(2), 3
    from torchvision import transforms as T

    # Match the recovered model-specific preprocessing.
    convmixer = cfg["model"] == "convmixer-768-32"
    mean = (0.485, 0.456, 0.406) if convmixer else (0.48145466, 0.4578275, 0.40821073)
    std = (0.229, 0.224, 0.225) if convmixer else (0.26862954, 0.26130258, 0.27577711)
    transform = T.Compose(
        [
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.Lambda(lambda image: image.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    from .dataset_io import load_prepared

    return load_prepared(name, cfg["data-root"], transform)

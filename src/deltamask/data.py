"""Seeded data adapters. Downloads require an explicit configuration opt-in."""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, Subset


def partitions(labels, num_clients, alpha, seed):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    if len(labels) < num_clients or alpha <= 0:
        raise ValueError("Insufficient samples or invalid alpha")
    result = [[] for _ in range(num_clients)]
    for label in np.unique(labels):
        indexes = np.flatnonzero(labels == label)
        rng.shuffle(indexes)
        sizes = rng.multinomial(
            len(indexes), rng.dirichlet(np.full(num_clients, alpha))
        )
        for i, part in enumerate(np.split(indexes, np.cumsum(sizes)[:-1])):
            result[i].extend(part.tolist())
    # Explicit deterministic empty-client repair moves one sample from the largest client.
    for i, part in enumerate(result):
        if not part:
            donor = max(range(num_clients), key=lambda j: len(result[j]))
            part.append(result[donor].pop())
    return [rng.permutation(part).astype(int) for part in result]


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
    if name == "synthetic":
        rng = np.random.default_rng(seed)
        teacher = rng.normal(size=(8, 3))

        def generate(n):
            x = rng.normal(size=(n, 8)).astype("float32")
            y = (x @ teacher).argmax(1)
            return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))

        return generate(cfg["num-clients"] * cfg["samples-per-client"]), generate(64), 3
    from torchvision import datasets as D, transforms as T

    is_clip = cfg["model"].startswith("clip")
    mean = (0.48145466, 0.4578275, 0.40821073) if is_clip else (0.485, 0.456, 0.406)
    std = (0.26862954, 0.26130258, 0.27577711) if is_clip else (0.229, 0.224, 0.225)
    transform = T.Compose(
        [
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.Lambda(lambda image: image.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    root = Path(cfg["data-root"]) / name
    download = cfg["allow-download"]
    if name in ["cifar10", "cifar100", "fashion-mnist", "emnist"]:
        cls = {
            "cifar10": D.CIFAR10,
            "cifar100": D.CIFAR100,
            "fashion-mnist": D.FashionMNIST,
            "emnist": D.EMNIST,
        }[name]
        extra = {"split": "balanced"} if name == "emnist" else {}
        train = cls(root, train=True, transform=transform, download=download, **extra)
        test = cls(root, train=False, transform=transform, download=download, **extra)
    elif name in ["svhn", "food101"]:
        cls = D.SVHN if name == "svhn" else D.Food101
        train = cls(root, split="train", transform=transform, download=download)
        test = cls(root, split="test", transform=transform, download=download)
    elif name in ["eurosat", "cars196", "imagefolder"]:
        # Require an explicit fixed split; never guess a benchmark's missing split recipe.
        train = D.ImageFolder(root / "train", transform=transform)
        test = D.ImageFolder(root / "test", transform=transform)
        if train.class_to_idx != test.class_to_idx:
            raise ValueError("Train/test class directories differ")
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return train, test, int(max(targets(train).max(), targets(test).max()) + 1)

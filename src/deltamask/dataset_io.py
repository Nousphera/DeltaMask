"""Checksum-verified preparation of the eight DeltaMask image benchmarks."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from PIL import Image
from torchvision import datasets as D
from torch.utils.data import Subset

PAPER_DATASETS = ("cifar10", "cifar100", "svhn", "fashion-mnist", "emnist", "food101", "eurosat", "cars196")
DATASETS = PAPER_DATASETS + ("imagefolder",)
SOURCES = {
    "cifar10": "https://www.cs.toronto.edu/~kriz/cifar.html",
    "cifar100": "https://www.cs.toronto.edu/~kriz/cifar.html",
    "svhn": "http://ufldl.stanford.edu/housenumbers/",
    "fashion-mnist": "https://github.com/zalandoresearch/fashion-mnist",
    "emnist": "https://www.nist.gov/itl/products-and-services/emnist-dataset",
    "food101": "https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/",
    "eurosat": "https://github.com/phelber/EuroSAT",
    "cars196": "https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset",
}
EXPECTED = {
    "cifar10": (10, 50000, 10000), "cifar100": (100, 50000, 10000),
    "svhn": (10, 73257, 26032), "fashion-mnist": (10, 60000, 10000),
    "food101": (101, 75750, 25250), "eurosat": (10, 21600, 5400),
    "cars196": (196, 8144, 8041),
}
CARS_LAYOUT = "stanford_cars/{cars_train/, cars_test/, devkit/cars_meta.mat, devkit/cars_train_annos.mat, cars_test_annos_withlabels.mat}"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root):
    root = Path(root)
    if root.is_symlink():
        raise ValueError("Dataset roots must not be symlinks")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not allowed in datasets: {path}")
        if path.is_file() and path != root / "manifest.json":
            files[path.relative_to(root).as_posix()] = digest(path)
    if not files:
        raise ValueError("Dataset is empty")
    return files


def labels(dataset):
    if isinstance(dataset, Subset):
        return labels(dataset.dataset)[dataset.indices]
    for name in ("targets", "labels", "_labels"):
        if hasattr(dataset, name):
            return np.asarray(getattr(dataset, name), dtype=np.int64)
    return np.asarray([label for _, label in (dataset.samples if hasattr(dataset, "samples") else dataset._samples)], dtype=np.int64)


def open_dataset(name, root, transform=None, download=False, *, emnist_split="letters", split_seed=42):
    """Open a prepared root, preserving the benchmark's train/test definitions."""
    root = Path(root)
    if name in ("cifar10", "cifar100", "fashion-mnist", "emnist"):
        cls = {"cifar10": D.CIFAR10, "cifar100": D.CIFAR100, "fashion-mnist": D.FashionMNIST, "emnist": D.EMNIST}[name]
        extra = {"split": emnist_split} if name == "emnist" else {}
        train = cls(root, train=True, transform=transform, download=download, **extra)
        test = cls(root, train=False, transform=transform, download=download, **extra)
        if name == "emnist" and emnist_split == "letters":
            for dataset in (train, test):
                dataset.targets = dataset.targets - 1
                dataset.classes = list("abcdefghijklmnopqrstuvwxyz")
                dataset.class_to_idx = {v: i for i, v in enumerate(dataset.classes)}
    elif name in ("svhn", "food101", "cars196"):
        cls = {"svhn": D.SVHN, "food101": D.Food101, "cars196": D.StanfordCars}[name]
        train = cls(root, split="train", transform=transform, download=download)
        test = cls(root, split="test", transform=transform, download=download)
        if name == "cars196":
            for dataset in (train, test):
                dataset.targets = [v for _, v in dataset._samples]
                dataset.samples = dataset._samples
    elif name == "eurosat":
        dataset = D.EuroSAT(root, transform=transform, download=download)
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(split_seed)).tolist()
        cut = len(dataset) - int(.2 * len(dataset))
        train, test = Subset(dataset, order[:cut]), Subset(dataset, order[cut:])
        for part in (train, test):
            part.classes = dataset.classes
            part.class_to_idx = dataset.class_to_idx
            part.samples = [dataset.samples[i] for i in part.indices]
    elif name == "imagefolder":
        if download:
            raise ValueError("imagefolder requires --source with train/test folders")
        train = D.ImageFolder(root / "train", transform=transform)
        test = D.ImageFolder(root / "test", transform=transform)
        if train.class_to_idx != test.class_to_idx:
            raise ValueError("Train/test class directories differ")
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return train, test


def describe(name, root, *, verify_images=False, **settings):
    train, test = open_dataset(name, root, **settings)
    class_to_idx = getattr(train, "class_to_idx", None)
    if class_to_idx is None:
        classes = getattr(train, "classes", [str(i) for i in range(10)])
        class_to_idx = {str(label): i for i, label in enumerate(classes)}
    expected = EXPECTED.get(name)
    if name == "emnist":
        expected = (26, 124800, 20800) if settings.get("emnist_split", "letters") == "letters" else (47, 112800, 18800)
    if expected and (len(class_to_idx), len(train), len(test)) != expected:
        raise ValueError(f"Incomplete {name}: expected {expected} (classes, train, test)")
    splits, image_hashes = {}, []
    for split, dataset in (("train", train), ("test", test)):
        y = labels(dataset)
        if len(y) == 0 or y.shape != (len(dataset),) or y.min() < 0 or y.max() >= len(class_to_idx):
            raise ValueError(f"Invalid {split} label range or sample count")
        splits[split] = {"samples": len(y), "labels_sha256": hashlib.sha256(y.astype("<i8").tobytes()).hexdigest()}
        if isinstance(dataset, Subset):
            splits[split]["indices_sha256"] = hashlib.sha256(np.asarray(dataset.indices, dtype="<i8").tobytes()).hexdigest()
        if verify_images and hasattr(dataset, "samples"):
            hashes = set()
            for filename, _ in dataset.samples:
                with Image.open(filename) as image:
                    image.load()
                hashes.add(digest(filename))
            image_hashes.append(hashes)
    if len(image_hashes) == 2 and image_hashes[0] & image_hashes[1]:
        raise ValueError("Exact duplicate image content appears in train and test")
    return {"class_to_idx": class_to_idx, "splits": splits}


def _manifest(name, directory):
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run python scripts/prepare_data.py --dataset {name}")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") not in (1, 2) or manifest.get("dataset") != name:
        raise ValueError("Manifest schema/dataset mismatch")
    return manifest


def _settings(manifest):
    if manifest["schema_version"] == 1 and manifest["dataset"] == "emnist":
        return {"emnist_split": "balanced"}
    return manifest.get("settings", {})


def validate_dataset(name, root):
    directory = Path(root) / name
    manifest = _manifest(name, directory)
    if inventory(directory) != manifest["files"]:
        raise ValueError("Dataset files differ from manifest; prepared data was changed")
    actual = describe(name, directory, verify_images=True, **_settings(manifest))
    if any(actual[key] != manifest[key] for key in actual):
        raise ValueError("Dataset labels or split sizes differ from manifest")
    return manifest


def load_prepared(name, root, transform=None):
    directory = Path(root) / name
    manifest = _manifest(name, directory)
    train, test = open_dataset(name, directory, transform=transform, **_settings(manifest))
    for split, dataset in (("train", train), ("test", test)):
        y = labels(dataset)
        if len(dataset) != manifest["splits"][split]["samples"] or hashlib.sha256(y.astype("<i8").tobytes()).hexdigest() != manifest["splits"][split]["labels_sha256"]:
            raise ValueError("Dataset labels or split sizes differ from manifest")
    if getattr(train, "class_to_idx", manifest["class_to_idx"]) != manifest["class_to_idx"]:
        raise ValueError("Dataset class mapping differs from manifest")
    return train, test, len(manifest["class_to_idx"])


def manifest_digest(name, root):
    return None if name == "synthetic-images" else digest(Path(root) / name / "manifest.json")


def copy_source(name, source, stage):
    """Copy only benchmark data; never unrelated files from the source root."""
    entries = {
        "cifar10": ["cifar-10-batches-py"], "cifar100": ["cifar-100-python"],
        "svhn": ["train_32x32.mat", "test_32x32.mat"], "fashion-mnist": ["FashionMNIST"],
        "emnist": ["EMNIST"], "food101": ["food-101"],
        "eurosat": ["eurosat"], "cars196": ["stanford_cars"],
    }.get(name, ["train", "test"])
    stage.mkdir()
    for entry in entries:
        path = source / entry
        if path.is_symlink():
            raise ValueError(f"Symlinks are not allowed in datasets: {path}")
        if path.is_dir():
            inventory(path)
            shutil.copytree(path, stage / entry)
        elif path.is_file():
            shutil.copy2(path, stage / entry)
        else:
            raise FileNotFoundError(f"Missing dataset source: {path}")


def prepare_dataset(name, root, *, source=None, download=None, raw_root="raw", emnist_split="letters", split_seed=42):
    if name not in DATASETS:
        raise ValueError(f"Unsupported dataset: {name}")
    if emnist_split not in ("letters", "balanced"):
        raise ValueError("emnist_split must be letters or balanced")
    if not isinstance(split_seed, int) or not 0 <= split_seed < 2**32:
        raise ValueError("split_seed must be a uint32")
    parent, destination = Path(root), Path(root) / name
    settings = {"emnist_split": emnist_split} if name == "emnist" else ({"split_seed": split_seed} if name == "eurosat" else {})
    if destination.exists():
        manifest = validate_dataset(name, root)
        if _settings(manifest) != settings:
            raise ValueError("Prepared split settings differ; choose a new --root")
        return manifest
    if source and download:
        raise ValueError("Choose --source or --download, not both")
    candidate = Path(raw_root) / name
    if source is None and download is not True and candidate.is_dir():
        source = candidate
    if name in ("cars196", "imagefolder") and source is None:
        layout = CARS_LAYOUT if name == "cars196" else "{train,test}/CLASS/IMAGE"
        raise FileNotFoundError(f"Download {name} via {SOURCES.get(name, 'your data provider')} and extract to {candidate}/{layout}")
    if source and (Path(source).resolve() == parent.resolve() or Path(source).resolve() in parent.resolve().parents):
        raise ValueError("Output root must not be inside the imported source")
    parent.mkdir(parents=True, exist_ok=True)
    # A persistent download stage lets torchvision reuse complete archives on retry.
    stage = parent / f".{name}-download"
    temporary = None
    try:
        if source:
            source = Path(source)
            if source.is_symlink() or not source.is_dir():
                raise ValueError("Source must be a real extracted directory")
            temporary = tempfile.TemporaryDirectory(prefix=f".{name}-import-", dir=parent)
            stage = Path(temporary.name) / name
            copy_source(name, source, stage)
        else:
            stage.mkdir(exist_ok=True)
            open_dataset(name, stage, download=True, **settings)
        manifest = {
            "schema_version": 2, "dataset": name,
            "source": SOURCES.get(name, "user-supplied train/test image folders"),
            "preparation": {"mode": "import" if source else "download"}, "settings": settings,
            **describe(name, stage, verify_images=True, **settings), "files": inventory(stage),
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if destination.exists():
            raise FileExistsError(destination)
        stage.rename(destination)
        return manifest
    finally:
        if temporary:
            temporary.cleanup()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, help="Omit to prepare all eight paper datasets")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--raw-root", type=Path, default=Path("raw"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--source", type=Path, help="Extracted source for one dataset")
    mode.add_argument("--download", action="store_true", default=None, help="Download even when raw/ contains an import")
    mode.add_argument("--validate", action="store_true", help="Verify existing prepared data")
    parser.add_argument("--emnist-split", choices=("letters", "balanced"), default="letters")
    parser.add_argument("--split-seed", type=int, default=42, help="EuroSAT 80/20 split seed")
    args = parser.parse_args(argv)
    if args.source and not args.dataset:
        parser.error("--source requires --dataset")
    failed = []
    for name in (args.dataset,) if args.dataset else PAPER_DATASETS:
        print(f"Preparing {name}…", flush=True)
        try:
            result = validate_dataset(name, args.root) if args.validate else prepare_dataset(
                name, args.root, source=args.source, download=args.download, raw_root=args.raw_root,
                emnist_split=args.emnist_split, split_seed=args.split_seed,
            )
            print(f"{name}: {result['splits']['train']['samples']:,} train · {result['splits']['test']['samples']:,} test", flush=True)
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            failed.append(f"{name}: {exc}")
            print(failed[-1], flush=True)
    if failed:
        parser.exit(2, "Preparation incomplete:\n" + "\n".join(failed) + "\n")

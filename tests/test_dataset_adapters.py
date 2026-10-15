"""Preparation uses benchmark-specific layouts and stable split identities."""

import json
import numpy as np
import pytest
import torch
from PIL import Image
from deltamask import dataset_io as io
from deltamask.data import partitions


def test_letters_are_zero_based(monkeypatch, tmp_path):
    class Letters:
        def __init__(self, root, train, transform, download, split):
            assert split == "letters"
            self.targets = torch.tensor([1, 26])
            self.classes = ["N/A"] + list("abcdefghijklmnopqrstuvwxyz")
        def __len__(self):
            return 2
    monkeypatch.setattr(io.D, "EMNIST", Letters)
    a, b = io.open_dataset("emnist", tmp_path)
    assert a.targets.tolist() == [0, 25]
    assert len(a.classes) == 26
    assert b.class_to_idx["z"] == 25


def test_eurosat_has_stable_nonoverlapping_official_image_split(monkeypatch, tmp_path):
    class Euro:
        def __init__(self, root, transform, download):
            self.targets = list(range(10)) * 10
            self.classes = list(map(str, range(10)))
            self.class_to_idx = dict(zip(self.classes, range(10)))
            self.samples = [(str(i), y) for i, y in enumerate(self.targets)]
        def __len__(self): return 100
    monkeypatch.setattr(io.D, "EuroSAT", Euro)
    a, b = io.open_dataset("eurosat", tmp_path)
    expected = torch.randperm(100, generator=torch.Generator().manual_seed(42)).tolist()
    assert a.indices == expected[:80]
    assert b.indices == expected[80:]
    assert len(io.labels(a)) == 80
    assert set(a.indices).isdisjoint(b.indices)


def test_repeat_preparation_validates_without_overwriting(tmp_path):
    source = tmp_path / "raw"
    rng = np.random.default_rng(3)
    for split in ["train", "test"]:
        for label in ["a", "b"]:
            path = source / split / label
            path.mkdir(parents=True)
            Image.fromarray(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)).save(path / "1.png")
    first = io.prepare_dataset("imagefolder", tmp_path / "data", source=source)
    second = io.prepare_dataset("imagefolder", tmp_path / "data")
    assert first == second
    assert io.prepare_dataset("imagefolder", tmp_path / "data", source=source) == first


def test_cars_missing_source_explains_exact_layout(tmp_path):
    with pytest.raises(FileNotFoundError, match="cars_test_annos_withlabels.mat"):
        io.prepare_dataset("cars196", tmp_path / "data", raw_root=tmp_path / "raw")


def test_cars_rejects_generic_class_folders(tmp_path):
    for split in ["train", "test"]:
        (tmp_path / "raw" / split / "class1").mkdir(parents=True)
    with pytest.raises((FileNotFoundError, RuntimeError)):
        io.prepare_dataset("cars196", tmp_path / "data", source=tmp_path / "raw")


def test_partition_matches_recovered_capacity_balanced_dirichlet():
    labels = np.repeat(np.arange(5), 40)
    rng = np.random.RandomState(42)
    expected = [[] for _ in range(4)]
    for label in range(5):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        p = rng.dirichlet(np.repeat(.1, 4))
        p *= np.asarray([len(v) < len(labels) / 4 for v in expected])
        cuts = (np.cumsum(p / p.sum()) * len(idx)).astype(int)[:-1]
        expected = [v + part.tolist() for v, part in zip(expected, np.split(idx, cuts))]
    for part in expected: rng.shuffle(part)
    assert [a.tolist() for a in partitions(labels, 4, .1, 42)] == expected


def test_cars_reads_labeled_official_annotations(tmp_path, monkeypatch):
    from scipy.io import savemat
    root = tmp_path / "raw" / "stanford_cars"
    (root / "devkit").mkdir(parents=True)
    savemat(root / "devkit/cars_meta.mat", {"class_names": np.array(["Car A", "Car B"], dtype=object)})
    rng = np.random.default_rng(81)
    for split in ("train", "test"):
        (root / f"cars_{split}").mkdir()
        annotations = np.zeros(2, dtype=[("bbox_x1", "O"), ("bbox_y1", "O"), ("bbox_x2", "O"), ("bbox_y2", "O"), ("class", "O"), ("fname", "O")])
        for i in range(2):
            filename = f"{i+1:05d}.jpg"
            Image.fromarray(rng.integers(0, 256, (12, 12, 3), dtype=np.uint8)).save(root / f"cars_{split}" / filename)
            annotations[i] = (1, 1, 12, 12, i+1, filename)
        target = root / ("devkit/cars_train_annos.mat" if split == "train" else "cars_test_annos_withlabels.mat")
        savemat(target, {"annotations": annotations})
    monkeypatch.setitem(io.EXPECTED, "cars196", (2, 2, 2))
    manifest = io.prepare_dataset("cars196", tmp_path / "data", source=root.parent)
    train, test, classes = io.load_prepared("cars196", tmp_path / "data")
    assert classes == 2
    assert train[0][1] == 0 and test[1][1] == 1
    assert manifest["splits"]["test"]["samples"] == 2
    assert io.validate_dataset("cars196", tmp_path / "data") == manifest


def test_omitted_dataset_processes_all_and_reports_manual_inputs(monkeypatch, capsys):
    calls = []
    def prepare(name, *args, **kwargs):
        calls.append(name)
        if name == "cars196":
            raise FileNotFoundError("manual images needed")
        return {"splits": {"train": {"samples": 1}, "test": {"samples": 1}}}
    monkeypatch.setattr(io, "prepare_dataset", prepare)
    with pytest.raises(SystemExit) as exc:
        io.main([])
    assert exc.value.code == 2
    assert tuple(calls) == io.PAPER_DATASETS
    assert "cars196: manual images needed" in capsys.readouterr().err

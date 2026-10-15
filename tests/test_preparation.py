"""Data preparation must preserve split identity and feed the actual trainer."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]


def fixture_images(root, train_samples=8, test_samples=3):
    rng = np.random.default_rng(19)
    for split, n in (("train", train_samples), ("test", test_samples)):
        for label in ("a", "b"):
            folder = root / split / label
            folder.mkdir(parents=True)
            for i in range(n):
                pixels = rng.integers(0, 256, (12, 12, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{i}.png")
    return root


def prepare(root, *args):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_data.py"),
            "--dataset",
            "imagefolder",
            "--root",
            str(root),
            *map(str, args),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_import_validate_and_reject_changed_files(tmp_path):
    source = fixture_images(tmp_path / "source")
    (source / "unrelated-private.txt").write_text("do not import")
    result = prepare(tmp_path / "data", "--source", source)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "data/imagefolder/manifest.json").read_text())
    assert manifest["splits"]["train"]["samples"] == 16
    assert manifest["splits"]["test"]["samples"] == 6
    assert manifest["class_to_idx"] == {"a": 0, "b": 1}
    assert not (tmp_path / "data/imagefolder/unrelated-private.txt").exists()
    assert prepare(tmp_path / "data", "--validate").returncode == 0
    assert prepare(tmp_path / "data", "--source", source).returncode == 0
    (tmp_path / "data/imagefolder/train/a/0.png").write_bytes(b"changed")
    assert prepare(tmp_path / "data", "--validate").returncode != 0


def test_reject_split_overlap_class_mismatch_and_symlinks(tmp_path):
    import shutil

    source = fixture_images(tmp_path / "source")
    shutil.copyfile(source / "train/a/0.png", source / "test/a/0.png")
    assert prepare(tmp_path / "data", "--source", source).returncode != 0
    assert not (tmp_path / "data/imagefolder").exists()
    (source / "test/a/0.png").unlink()
    (source / "test/b").rename(source / "test/c")
    assert prepare(tmp_path / "data", "--source", source).returncode != 0
    (source / "test/c").rename(source / "test/b")
    (source / "train/a/link.png").symlink_to(source / "test/a/1.png")
    assert prepare(tmp_path / "data", "--source", source).returncode != 0


def test_prepared_images_run_actual_apps_with_paper_dinov2(tmp_path):
    from deltamask.config import defaults
    from deltamask.train import run

    source = fixture_images(tmp_path / "source", train_samples=1, test_samples=1)
    result = prepare(tmp_path / "data", "--source", source)
    assert result.returncode == 0, result.stderr
    cfg = defaults() | {
        "dataset": "imagefolder",
        "data-root": str(tmp_path / "data"),
        "model": "dinov2-small",
        "pretrained": False,
        "output": str(tmp_path / "result.json"),
        "num-rounds": 1,
    }
    report = run(cfg)
    assert len(report["rounds"]) == 1
    assert report["rounds"][0]["payload_bytes"] > 0
    assert len(report["data_manifest_sha256"]) == 64
    assert (tmp_path / "result.npz").is_file()


def test_preparation_rejects_image_with_valid_header_but_truncated_pixels(tmp_path):
    source = fixture_images(tmp_path / "source")
    image = source / "train/a/truncated.jpg"
    Image.new("RGB", (40, 40), (20, 40, 80)).save(image)
    image.write_bytes(image.read_bytes()[:-20])
    with Image.open(image) as header:
        header.verify()  # JPEG header verification alone accepts this file.
    result = prepare(tmp_path / "data", "--source", source)
    assert result.returncode != 0
    assert not (tmp_path / "data/imagefolder").exists()

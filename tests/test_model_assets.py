from pathlib import Path
import hashlib
import json
import pytest
from deltamask import model_assets as assets


def test_missing_weights_have_preparation_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="scripts/prepare_model.py"):
        assets.resolve_model({"model": "dinov2-small", "pretrained": True, "model-root": str(tmp_path)})


def test_offline_fixture_does_not_download(tmp_path, monkeypatch):
    cfg = {"model": "dinov2-small", "pretrained": False, "model-root": str(tmp_path)}
    monkeypatch.setattr(assets, "prepare_model", lambda *a, **k: pytest.fail("unexpected download"))
    assert assets.resolve_model(cfg) == cfg
    assert assets.model_identity(cfg)["initialization"] == "random"


def test_prepared_weights_check_expected_hash_and_reuse(tmp_path, monkeypatch):
    data = b"test model bytes"
    spec = assets.MODELS["dinov2-small"] | {"sha256": hashlib.sha256(data).hexdigest()}
    monkeypatch.setitem(assets.MODELS, "dinov2-small", spec)
    def fetch(repo_id, filename, revision, local_dir, **kwargs):
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data if filename == spec["weight"] else b'{}')
        return str(target)
    monkeypatch.setattr(assets, "hf_hub_download", fetch)
    first = assets.prepare_model("dinov2-small", tmp_path)
    monkeypatch.setattr(assets, "hf_hub_download", lambda *a, **k: pytest.fail("unexpected re-download"))
    assert assets.prepare_model("dinov2-small", tmp_path) == first
    cfg = assets.resolve_model({"model": "dinov2-small", "pretrained": True, "model-root": str(tmp_path)})
    assert cfg["model-revision"] == spec["revision"]
    assert assets.model_identity(cfg)["files"][spec["weight"]] == spec["sha256"]
    (tmp_path / "dinov2-small" / spec["weight"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        assets.resolve_model({"model": "dinov2-small", "pretrained": True, "model-root": str(tmp_path)})


def test_custom_checkpoint_identity_uses_actual_bytes(tmp_path):
    file = tmp_path / "weights.safetensors"
    file.write_bytes(b"weights")
    # An unrelated manifest in the same directory must not identify this file.
    (tmp_path / "manifest.json").write_text(json.dumps({"model": "convmixer-768-32", "revision": "fake"}))
    cfg = {"model": "convmixer-768-32", "pretrained": True, "model-checkpoint": str(file)}
    identity = assets.model_identity(cfg)
    assert identity["source"] == "local"
    assert identity["files"][file.name] == hashlib.sha256(b"weights").hexdigest()


def test_managed_directory_rejects_alternate_preferred_weights(tmp_path, monkeypatch):
    folder = tmp_path / "clip-vit-b32"
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(b"alternate")
    (folder / "manifest.json").write_text('{}')
    with pytest.raises(ValueError, match="Unexpected weights"):
        assets.resolve_model({"model": "clip-vit-b32", "pretrained": True, "model-root": str(tmp_path)})

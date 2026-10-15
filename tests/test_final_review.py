"""Focused release-review regressions on real data gates and full ConvMixer."""

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import TensorDataset

from deltamask.config import defaults
from deltamask.model import MaskedLayer, make_model, mask_scores, train_local, probe, evaluate


def test_server_and_evaluator_reject_changed_test_image_before_model_work(tmp_path, monkeypatch):
    from deltamask import model_assets
    from deltamask.dataset_io import prepare_dataset, manifest_digest
    from deltamask.train import run
    from deltamask.evaluate import evaluate_result

    source, root = tmp_path / "source", tmp_path / "data"
    rng = np.random.default_rng(23)
    for split in ("train", "test"):
        for label in ("a", "b"):
            path = source / split / label / "0.png"
            path.parent.mkdir(parents=True)
            Image.fromarray(rng.integers(0, 256, (12, 12, 3), dtype=np.uint8)).save(path)
    prepare_dataset("imagefolder", root, source=source)
    cfg = defaults() | {"dataset": "imagefolder", "data-root": str(root),
                       "model": "dinov2-small", "pretrained": False,
                       "method": "linear-probe", "num-rounds": 1,
                       "output": str(tmp_path / "valid.json")}
    report = run(cfg)
    original_digest = manifest_digest("imagefolder", root)
    Image.new("RGB", (12, 12), (1, 2, 3)).save(root / "imagefolder/test/a/0.png")
    assert manifest_digest("imagefolder", root) == original_digest == report["data_manifest_sha256"]

    def forbidden_model_work(*args, **kwargs):
        pytest.fail("Changed data must be rejected before resolving or inspecting model assets")
    monkeypatch.setattr(model_assets, "resolve_model", forbidden_model_work)
    monkeypatch.setattr(model_assets, "model_identity", forbidden_model_work)
    with pytest.raises(ValueError, match="Dataset files differ"):
        run(cfg | {"output": str(tmp_path / "invalid.json")})
    with pytest.raises(ValueError, match="Dataset files differ"):
        evaluate_result(cfg["output"])


@pytest.mark.parametrize("method", ["deltamask", "fedpm", "fedmask"])
def test_full_convmixer_masks_only_final_five_pointwise_convolutions(method):
    from deltamask.server_app import _common_dimension
    cfg = defaults() | {"model": "convmixer-768-32", "pretrained": False, "method": method}
    model = make_model(cfg, 3)
    common = _common_dimension(model)
    pointwise = {f"backbone.blocks.{i}.1" for i in range(27, 32)}
    selected = sum(p.numel() for i in range(27, 32) for p in model.backbone.blocks[i][1].parameters())
    selected += sum(p.numel() for p in model.head.parameters())
    model.configure_method(method)
    actual = {n for n, layer in model.backbone.named_modules() if isinstance(layer, MaskedLayer)}
    assert {"backbone." + n for n in actual} == pointwise
    assert len(model.backbone.blocks) == 32
    assert all(isinstance(model.backbone.blocks[i][0].fn[0], torch.nn.Conv2d) for i in range(32))
    assert all(not p.requires_grad for n, p in model.named_parameters() if not n.endswith("_mask"))
    assert len(mask_scores(model)) == selected < common


@pytest.mark.parametrize("phase", ["probe", "mask", "finetune"])
def test_full_convmixer_local_bn_updates_but_learned_transport_excludes_buffers(phase):
    from deltamask.baselines import parameter_state, set_parameters
    from deltamask.codec import encode_parameters, decode_parameters
    cfg = defaults() | {"model": "convmixer-768-32", "pretrained": False,
                       "batch-size": 2, "learning-rate": .001}
    model = make_model(cfg, 3)
    if phase != "probe":
        model.configure_method("deltamask" if phase == "mask" else "finetune")
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    base_buffers = {n: value.clone() for n, value in model.named_buffers()}
    # Full official 768/32 architecture; 28x28 inputs keep this BN regression bounded.
    gen = torch.Generator().manual_seed(71)
    ds = TensorDataset(torch.randn(2, 3, 28, 28, generator=gen), torch.tensor([0, 1]))
    if phase == "probe":
        probe(model, ds, cfg, 52)
    else:
        train_local(model, ds, cfg, 52)
    assert model.training
    assert all(layer.training for layer in model.modules() if isinstance(layer, torch.nn.BatchNorm2d))
    assert any(not torch.equal(value, base_buffers[n]) for n, value in model.named_buffers()
               if n.endswith("running_mean"))
    assert all(torch.equal(dict(model.named_parameters())[n], value) for n, value in frozen.items())
    assert set(parameter_state(model)).isdisjoint(base_buffers)
    server = make_model(cfg, 3)
    if phase != "probe":
        server.configure_method("deltamask" if phase == "mask" else "finetune")
    # Apply only the learned state (the same mask/parameter boundary used by
    # checkpoint restoration); local BN running buffers never enter this state.
    wire = encode_parameters(parameter_state(model))
    set_parameters(server, decode_parameters(wire, parameter_state(server)))
    assert all(torch.equal(value, base_buffers[n]) for n, value in server.named_buffers())
    evaluate(server, ds, cfg, 53)
    assert not server.training
    assert all(torch.equal(value, base_buffers[n]) for n, value in server.named_buffers())

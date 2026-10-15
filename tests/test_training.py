import copy
import numpy as np
import torch
from deltamask.model import make_model, mask_scores, set_mask_scores, train_local, probe
from deltamask.data import load_data, partitions
from deltamask.config import defaults


def test_frozen_backbone_masks_train_and_probe_head(image_check_config):
    cfg = image_check_config
    ds, _, classes = load_data(cfg)
    model = make_model(cfg, classes)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    probe(model, ds, cfg, seed=42)
    assert not torch.equal(before["head.weight"], model.head.weight)
    for n, p in model.named_parameters():
        if not n.startswith("head."):
            assert torch.equal(before[n], p)
    model.enable_masks()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    scores = mask_scores(model)
    train_local(model, ds, cfg, seed=43)
    assert not np.array_equal(scores, mask_scores(model))
    for n, p in model.named_parameters():
        if not n.endswith("_mask"):
            assert torch.equal(before[n], p)
    assert all(
        n.endswith("_mask") for n, p in model.named_parameters() if p.requires_grad
    )


def test_partition_is_seeded_disjoint_and_complete():
    labels = np.repeat(np.arange(3), 40)
    a = partitions(labels, 4, 0.1, 42)
    b = partitions(labels, 4, 0.1, 42)
    assert [v.tolist() for v in a] == [v.tolist() for v in b]
    assert sorted(np.concatenate(a).tolist()) == list(range(len(labels)))
    assert min(map(len, a)) > 0


def test_full_clip_last_five_blocks_and_gradients():
    from deltamask.model import MaskedLayer

    cfg = defaults() | {"pretrained": False}
    model = make_model(cfg, 3)
    model.configure_method("finetune")
    assert model.backbone.visual_projection.weight.requires_grad
    assert model.head.weight.requires_grad
    assert model.backbone.vision_model.encoder.layers[7].self_attn.q_proj.weight.requires_grad
    assert not model.backbone.vision_model.encoder.layers[6].self_attn.q_proj.weight.requires_grad
    assert not model.backbone.vision_model.encoder.layers[7].layer_norm1.weight.requires_grad
    model.enable_masks()
    layers = model.backbone.vision_model.encoder.layers
    assert len(layers) == 12
    assert not any(
        isinstance(m, MaskedLayer) for layer in layers[:7] for m in layer.modules()
    )
    assert all(
        any(isinstance(m, MaskedLayer) for m in layer.modules()) for layer in layers[7:]
    )
    model(torch.randn(1, 3, 224, 224)).sum().backward()
    assert all(
        p.grad is None for n, p in model.named_parameters() if not n.endswith("_mask")
    )
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad) > 0
        for n, p in model.named_parameters()
        if n.endswith("_mask")
    )


def test_convolution_masks_preserve_depthwise_shape_and_backbone():
    from deltamask.model import MaskedLayer

    layer = torch.nn.Conv2d(4, 4, 3, padding=1, groups=4)
    masked = MaskedLayer(layer)
    masked(torch.randn(2, 4, 8, 8)).square().mean().backward()
    assert layer.weight.grad is None and masked.weight_mask.grad is not None


def test_default_backbone_uses_full_paper_clip_without_weights():
    torch.set_num_threads(1)
    cfg = defaults() | {"pretrained": False}
    model = make_model(cfg, 3)
    assert sum(p.numel() for p in model.backbone.parameters()) > 85_000_000
    config = model.backbone.config
    assert (config.hidden_size, config.num_hidden_layers, config.patch_size) == (
        768,
        12,
        32,
    )


def test_nonpaper_backbones_are_rejected():
    import pytest
    from deltamask.config import validate

    with pytest.raises(ValueError, match="paper"):
        validate(defaults() | {"model": "mlp"})

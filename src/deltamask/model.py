"""Frozen pretrained weights with trainable stochastic masks.

The recovered implementation scales the Bernoulli surrogate gradient by p;
sigmoid supplies its own derivative as well. Scores start as seeded N(0,1).
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class Bernoulli(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p):
        ctx.save_for_backward(p)
        return torch.bernoulli(p)

    @staticmethod
    def backward(ctx, gradient):
        return ctx.saved_tensors[0] * gradient


class Threshold(torch.autograd.Function):
    """Recovered FedMask threshold-training variant, including masked gradient."""

    @staticmethod
    def forward(ctx, p):
        active = p > .5
        ctx.save_for_backward(active)
        return active.to(p.dtype)

    @staticmethod
    def backward(ctx, gradient):
        return ctx.saved_tensors[0] * gradient


class MaskedLayer(nn.Module):
    def __init__(self, layer, generator=None, threshold=False):
        super().__init__()
        self.layer = layer
        for p in layer.parameters():
            p.requires_grad_(False)
        self.threshold = threshold
        def scores(p):
            # CPU generator makes initialization independent of device/global RNG.
            return torch.randn(p.shape, generator=generator, dtype=p.dtype).to(p.device)
        self.weight_mask = nn.Parameter(scores(layer.weight))
        if layer.bias is not None:
            self.bias_mask = nn.Parameter(scores(layer.bias))
        else:
            self.register_parameter("bias_mask", None)

    def forward(self, x):
        sample = Threshold.apply if self.threshold else Bernoulli.apply
        weight = self.layer.weight * sample(self.weight_mask.sigmoid())
        bias = (
            None
            if self.bias_mask is None
            else self.layer.bias * sample(self.bias_mask.sigmoid())
        )
        if isinstance(self.layer, nn.Conv2d):
            return self.layer._conv_forward(x, weight, bias)
        return nn.functional.linear(x, weight, bias)


def mask_children(module, generator=None, threshold=False):
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.Linear, nn.Conv2d)):
            setattr(module, name, MaskedLayer(child, generator, threshold))
        elif not isinstance(child, MaskedLayer):
            mask_children(child, generator, threshold)


class Model(nn.Module):
    def __init__(self, backbone, features, classes, blocks, kind):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(features, classes)
        self.blocks = blocks
        self.kind = kind
        self.mask_seed = 42
        for p in backbone.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        if self.kind == "clip":
            x = self.backbone(pixel_values=x).image_embeds
        elif self.kind == "dino":
            x = self.backbone(pixel_values=x).last_hidden_state[:, 0]
        else:
            x = self.backbone(x)
        return self.head(x)

    def enable_masks(self, threshold=False):
        if isinstance(self.head, MaskedLayer):
            return
        generator = torch.Generator().manual_seed(self.mask_seed)
        for block in self.blocks:
            if self.kind == "convmixer":
                # Recovered ConvMixer mask selector targets block.1 (pointwise)
                # only; the depthwise convolution in block.0.fn.0 stays frozen.
                block[1] = MaskedLayer(block[1], generator, threshold)
            else:
                mask_children(block, generator, threshold)
        # Apply masks to the CLIP projection and classifier as well.
        if self.kind == "clip":
            self.backbone.visual_projection = MaskedLayer(
                self.backbone.visual_projection, generator, threshold
            )
        self.head = MaskedLayer(self.head, generator, threshold)
        for n, p in self.named_parameters():
            p.requires_grad_(n.endswith("_mask"))

    def configure_method(self, method):
        from .config import MASK_METHODS
        if method in MASK_METHODS:
            self.enable_masks(threshold=method == "fedmask")
        elif method in {"finetune", "linear-probe"}:
            for p in self.parameters():
                p.requires_grad_(False)
            modules = [self.head]
            if method == "finetune":
                modules += [layer for block in self.blocks for layer in block.modules()
                            if isinstance(layer, (nn.Linear, nn.Conv2d))]
                if self.kind == "clip":
                    modules.append(self.backbone.visual_projection)
            for module in modules:
                for p in module.parameters():
                    p.requires_grad_(True)
        else:
            raise ValueError("Unknown training method")


def make_model(cfg, classes):
    from .config import defaults, validate
    from .model_assets import resolve_model

    cfg = resolve_model(validate(defaults() | cfg))
    torch.set_num_threads(cfg["torch-threads"])
    revisions = {
        "clip-vit-b32": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        "clip-vit-l14": "32bd64288804d66eefd0ccbe215aa642df71cc41",
        "dinov2-base": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
        "dinov2-small": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
    }
    revision = cfg["model-revision"] or revisions.get(cfg["model"])
    with torch.random.fork_rng():
        torch.manual_seed(cfg["seed"])
        name = cfg["model"]
        if name.startswith("clip"):
            from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

            identifier = {
                "clip-vit-b32": "openai/clip-vit-base-patch32",
                "clip-vit-l14": "openai/clip-vit-large-patch14",
            }[name]
            dimensions = {
                "clip-vit-b32": dict(
                    hidden_size=768,
                    intermediate_size=3072,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    patch_size=32,
                    projection_dim=512,
                ),
                "clip-vit-l14": dict(
                    hidden_size=1024,
                    intermediate_size=4096,
                    num_hidden_layers=24,
                    num_attention_heads=16,
                    patch_size=14,
                    projection_dim=768,
                ),
            }[name]
            if cfg["pretrained"]:
                backbone = CLIPVisionModelWithProjection.from_pretrained(
                    cfg["model-checkpoint"] or identifier,
                    revision=None if cfg["model-checkpoint"] else revision,
                    local_files_only=not cfg["allow-download"],
                )
            else:
                backbone = CLIPVisionModelWithProjection(CLIPVisionConfig(**dimensions))
            if any(
                getattr(backbone.config, key) != value
                for key, value in dimensions.items()
            ):
                raise ValueError(
                    "Checkpoint architecture does not match the paper model"
                )
            model = Model(
                backbone,
                backbone.config.projection_dim,
                classes,
                list(backbone.vision_model.encoder.layers)[-5:],
                "clip",
            )
        elif name.startswith("dinov2"):
            from transformers import AutoModel, Dinov2Config, Dinov2Model

            identifier = {
                "dinov2-base": "facebook/dinov2-base",
                "dinov2-small": "facebook/dinov2-small",
            }[name]
            dimensions = {
                "dinov2-base": dict(hidden_size=768, num_attention_heads=12),
                "dinov2-small": dict(hidden_size=384, num_attention_heads=6),
            }[name] | dict(
                num_hidden_layers=12, patch_size=14, image_size=518, mlp_ratio=4
            )
            if cfg["pretrained"]:
                backbone = AutoModel.from_pretrained(
                    cfg["model-checkpoint"] or identifier,
                    revision=None if cfg["model-checkpoint"] else revision,
                    local_files_only=not cfg["allow-download"],
                )
            else:
                backbone = Dinov2Model(Dinov2Config(**dimensions))
            if any(
                getattr(backbone.config, key) != value
                for key, value in dimensions.items()
            ):
                raise ValueError(
                    "Checkpoint architecture does not match the paper model"
                )
            model = Model(
                backbone,
                backbone.config.hidden_size,
                classes,
                list(backbone.encoder.layer)[-5:],
                "dino",
            )
        elif name == "convmixer-768-32":
            import timm

            checkpoint = cfg.get("model-checkpoint", "")
            backbone = timm.create_model(
                "convmixer_768_32.in1k",
                pretrained=False,
                checkpoint_path=checkpoint,
            )
            features = backbone.num_features
            backbone.reset_classifier(0)
            model = Model(
                backbone, features, classes, list(backbone.blocks)[-5:], "convmixer"
            )
        else:
            raise ValueError(f"Unsupported model: {name}")
    model.mask_seed = cfg["seed"]
    return model.to(cfg.get("device", "cpu"))


def mask_scores(model):
    return np.concatenate(
        [
            p.detach().cpu().numpy().ravel()
            for n, p in model.named_parameters()
            if n.endswith("_mask")
        ]
    )


def set_mask_scores(model, values):
    values = np.asarray(values)
    params = [p for n, p in model.named_parameters() if n.endswith("_mask")]
    if (values.ndim != 1 or values.size != sum(p.numel() for p in params)
            or not np.issubdtype(values.dtype, np.floating)
            or not np.isfinite(values).all()
            or (params and np.any(np.abs(values) > min(torch.finfo(p.dtype).max for p in params)))):
        raise ValueError("Expected finite flat scores matching the mask layout")
    offset = 0
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("_mask"):
                count = p.numel()
                p.copy_(
                    torch.as_tensor(
                        values[offset : offset + count].reshape(p.shape),
                        device=p.device,
                    )
                )
                offset += count
    if offset != values.size:
        raise ValueError("Mask layout mismatch")


def _fit(model, ds, cfg, seed, head_only):
    torch.manual_seed(seed)
    # Match recovered local training: BatchNorm uses and updates local statistics.
    # Only learned parameters/masks leave the client, never these running buffers.
    model.train()
    params = (
        list(model.head.parameters())
        if head_only
        else [p for p in model.parameters() if p.requires_grad]
    )
    optimizer = torch.optim.Adam(
        params, lr=cfg["probe-learning-rate" if head_only else "learning-rate"]
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["batch-size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    loss_sum, seen = 0., 0
    device = next(model.parameters()).device
    for _ in range(1 if head_only else cfg["local-epochs"]):
        for x, y in loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            seen += len(y)
    if not seen:
        raise ValueError("Cannot train an empty client")
    return loss_sum / seen


def probe(model, ds, cfg, seed):
    return _fit(model, ds, cfg, seed, True)


def train_local(model, ds, cfg, seed):
    return _fit(model, ds, cfg, seed, False)


def evaluate(model, ds, cfg, seed):
    model.eval()
    torch.manual_seed(seed)
    correct = 0
    loss = 0.0
    device = next(model.parameters()).device
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=cfg["batch-size"]):
            y = y.to(device)
            pred = model(x.to(device))
            correct += int((pred.argmax(1) == y).sum())
            loss += nn.functional.cross_entropy(pred, y, reduction="sum").item()
    return {"accuracy": correct / len(ds), "loss": loss / len(ds)}

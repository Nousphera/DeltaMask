# Implementation and experiment settings

DeltaMask trains stochastic masks over frozen model weights, compresses changed mask positions and aggregates binary observations. See [the IEEE BigData paper](https://doi.org/10.1109/BigData62323.2024.10825398) and [accepted manuscript](https://pure.tue.nl/ws/portalfiles/portal/349909071/2024337517.pdf).

## Training and transport

| Stage | Code | Behavior |
|---|---|---|
| Classifier adaptation | `model.probe`, `server_app.run_server` | One local epoch per client, sample-weighted classifier averaging, frozen backbone |
| Stochastic masks | `model.Model.enable_masks`, `MaskedLayer` | Final five backbone blocks; Bernoulli forward with identity straight-through gradients in probability space |
| Shared reference | `codec.sample_mask` | NumPy PCG64, seed plus round, one flattened random stream |
| Changed-position selection | `codec.bernoulli_kl`, `selected_changes` | Full Bernoulli KL, floor(κ × changed positions), stable index ties |
| Four-wise BFuse8 | `native/fastfilter`, `_bfuse` | Four hash locations and eight-bit fingerprints; all inserted keys verified after construction |
| Payload | `codec.encode` / `decode` | PNG bytes, filter geometry, seed, dimension, version and context digest |
| Mask reconstruction | `server_app.run_server` | Query every coordinate and XOR matches with the shared reference |
| Bayesian aggregation | `aggregate.BayesianMask` | Unweighted binary observations, Beta(1,1) MAP and probability clipping |
| Flower application | `client_app`, `server_app` | Classifier arrays during setup; encoded bytes for mask updates |

Local and global transmission masks use common random uniforms within a round. Each mask is marginally Bernoulli; the coupling reduces changed positions. Each round uses a new public stream, while local minibatch sampling has a separate Torch client/round seed.

Mask logits initialize at zero, giving probability 0.5. Selected linear/convolution weights and biases, the CLIP projection and the classifier are masked; earlier blocks and normalization parameters remain frozen. The Bernoulli straight-through estimator passes the incoming gradient unchanged in probability space, and sigmoid supplies its derivative. Adam optimizes local scores.

The cosine selection schedule increases κ from 0.8 to 1.0 over `max(1, (R−1)//16)` rounds and stays at its endpoint. Bayesian priors reset before aggregation at rounds divisible by `floor(1/participation)`. Probabilities are clipped to `[1e−7, 1−1e−7]`. Evaluation samples seeded stochastic masks per forward batch.

Top-κ selection and filter false positives make mask reconstruction lossy; PNG compression is lossless. The payload carries visible filter fingerprints and public sampling metadata.

## Data and models

[DATASETS.md](DATASETS.md) defines preparation, acquisition and consumed paths. Image preprocessing converts to RGB, resizes/center-crops to 224×224 and applies CLIP or ImageNet normalization. CIFAR-10/100, SVHN, Fashion-MNIST, Food-101 and EMNIST use torchvision splits; EMNIST uses **balanced, 47 classes**. EuroSAT, Cars196 and general ImageFolder inputs use supplied fixed train/test class folders.

Client partitions use seeded per-class Dirichlet draws and multinomial allocation. Empty clients receive one sample from the largest partition, so every training sample is assigned once. Configurations include 30 clients, one local epoch, α=10 with full participation, or α=0.1 with 20% participation. Save the manifest and run multiple independent seeds when comparing results.

CLIP ViT-B/32 and ViT-L/14, DINOv2 Base and Small use pinned Hugging Face revisions in `model.py`; `model-revision` provides an explicit override. CLIP uses its visual projection and DINOv2 its CLS representation. ConvMixer-768/32 uses timm's ImageNet-pretrained `convmixer_768_32.in1k`. Model acquisition requires `allow-download=true`; cached weights or local checkpoints allow offline execution. Record the checkpoint revision or file SHA256 with each experiment.

The default synthetic setup uses deterministic eight-dimensional vectors, a fixed random linear teacher, three classes and disjoint client partitions. The tiny backbone is randomly initialized, making this a fast complete execution check without external assets.

## Flower execution and resources

[README.md](README.md#run) gives the primary `flwr run . local` command. It launches the `ClientApp` and `ServerApp` through Flower's Ray simulation backend. `num-clients` must match `num-supernodes`; configure per-client CPU/GPU resources and the available Ray CPU budget explicitly. Every selected client must reply; dropout handling is unsupported.

Use absolute data, checkpoint and output paths: Flower executes an installed application bundle with its own working directory. Its C++ extension is built from the bundled native source. The full vision stack is an application dependency so isolated runtimes have the same loaders as the host environment.

Flower 1.40 requires an actual virtual-environment path without spaces for its migration/runtime path handling; a symlink does not resolve that restriction. `FLWR_HOME` selects runtime state and `FLWR_LOCAL_SUPERLINK_HTTP_API_PORT` selects the local service's HTTP port. State-directory separation alone does not isolate listening addresses. Use one local service at a time or configure distinct SuperLink addresses.

`deltamask-train` executes the same application handlers sequentially through an in-process transport. Both entry points use the same codec, native filter and server aggregation.

## Outputs and byte accounting

JSON reports configuration, the input-manifest hash, package versions, per-round accuracy and actual mask-payload bytes, including metadata and PNG. Uplink bits per parameter divide those bytes by the participating client count and masked parameter count. Separate counters record classifier-probe uplink and score/head downlink record sizes.

These are application-payload measurements: network framing, metrics/control messages, initial pretrained-checkpoint downloads and deployment setup are excluded. Small models have proportionally larger headers.

The sibling `.npz` stores `scores`, `head.weight` and `head.bias` without pickle. To evaluate saved state, rebuild `make_model` with the JSON configuration and class count, load the head tensors, call `enable_masks()`, then `set_mask_scores(model, scores)`. Frozen pretrained weights are loaded from the recorded model revision. Optimizer state and automatic training resumption are not stored. Existing JSON or checkpoint files are protected from overwrite.

## Validation

```bash
python -m pip check
python -m pytest -q
```

Tests cover native four-wise construction and membership, false positives, PNG roundtrips and corruption, seed/round separation, Bernoulli KL, prior resets, disjoint partitions, frozen-weight gradients, exact native source packaging and both Flower handlers. A small six-block CLIP checks final-five-block masking and live mask gradients; a convolution fixture checks depthwise masking. Prepared-image integration exercises the same data path used by Flower.

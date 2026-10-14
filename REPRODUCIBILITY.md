# Reproducibility and code-to-paper mapping

This is a runnable implementation of the paper's mask-training and compression
pipeline, with corrections to recovered experimental code. It is not a verified
reproduction of the published benchmark tables.

| Paper component | Implementation | Validation / boundary |
| --- | --- | --- |
| Algorithm 1, initial classifier adaptation | `model.probe`, `server_app.run_server` | One local epoch at every client, sample-count-weighted classifier averaging; backbone frozen |
| Stochastic masking, final five backbone blocks | `model.Model.enable_masks`, `MaskedLayer` | Bernoulli forward and identity straight-through gradient in probability space; frozen originals checked |
| Shared reference mask | `codec.sample_mask` | NumPy PCG64, seed plus round, one flattened stream; no repeated seed per layer |
| Eq.4 importance selection | `codec.bernoulli_kl`, `selected_changes` | Full Bernoulli KL: p log(p/q)+(1−p)log((1−p)/(1−q)); floor(κ·changed), stable index ties |
| Eq.1–2 four-wise BFuse8 | `native/fastfilter`, `_bfuse` | Four hash locations, 8-bit fingerprints; all inserted keys queried after construction |
| PNG/DEFLATE transfer | `codec.encode` / `decode` | Actual PNG bytes plus version, filter geometry, seed, dimension, and context digest; empty update explicit |
| Algorithm 1 lines 13–16 | `server_app.run_server` | Server receives bytes, queries every coordinate, XORs the shared reference |
| Eq.3 aggregation | `aggregate.BayesianMask` | Unweighted binary observations; Beta(1,1) MAP, clip to [1e−7,1−1e−7]; reset at rounds divisible by floor(1/participation) |
| Flower execution | `client_app`, `server_app` | Current message API; initial head arrays then mask replies containing bytes only |

## Explicit implementation choices and corrections

- The recovered implementation used `abs(rel_entr(p,q))`, omitting the Bernoulli
  complement, Xor16 by default, a three-hash Fuse header, and simulated byte
  counts. This release uses the paper's full KL and four-wise BFuse8 with actual
  reconstruction metadata included in its payload count.
- Local and global transmission masks use **common random uniforms** per round,
  preserving the recovered implementation's useful coupling and reducing changes.
  Each remains marginally Bernoulli. Randomness is public and changes across
  rounds; local minibatch sampling uses a separate Torch client/round seed. The
  paper does not precisely specify this coupling. Do not interpret clients'
  transmitted samples as independent draws.
- The recovered custom backward multiplied its incoming gradient by probability
  before sigmoid's derivative. Here Bernoulli uses the conventional identity
  straight-through estimator; sigmoid supplies its own derivative. The paper
  does not specify this surrogate. Training results can change.
- Masks initialize at zero logits (probability 0.5), rather than random logits.
  All selected linear/conv weights and biases, the CLIP projection, and the
  classifier receive masks; earlier blocks and normalization parameters freeze.
  One federated probe round replaces the unavailable precomputed head file.
- Adam optimizes local scores; learning rates are explicit. The recovered
  cosine top-κ horizon is max(1,(R−1)//16), increasing κ from 0.8 to 1.0; it is
  clamped rather than cycling after that horizon. Periodic priors reset **before**
  aggregation at divisible rounds, matching the recovered implementation.
- Evaluation uses a seeded stochastic mask per forward batch. It does not silently
  replace Bernoulli masks with mean probabilities or thresholded masks.
- Top-κ truncation and filter false positives are lossy. PNG is lossless. This
  release makes no unbiasedness, Appendix B error-bound, encryption, or privacy
  claim. Shared randomness, truncation bias, and filter errors require separate
  theoretical analysis.

## Data and pretrained models

Synthetic data are deterministic 8-dimensional vectors with a fixed random
linear teacher, three classes, and disjoint client partitions; the tiny backbone
is randomly initialized. This fixture validates execution, not foundation-model
adaptation or paper accuracy. No external assets are needed for it.

Image data use 224×224 resize/center crop, RGB conversion and CLIP or ImageNet
normalization. CIFAR-10/100, SVHN (train/test), Fashion-MNIST, Food-101, and EMNIST
**balanced, 47 classes** use torchvision. The exact paper EMNIST variant was not
recovered; do not equate this choice with the reported benchmark without checking
its original data manifest. EuroSAT and Cars196 require locally prepared fixed
`data/<dataset>/train/<class>/*` and `test/<class>/*` directories with matching
class maps. No split is invented for these datasets. Data licenses are separate.

Client partitions use a seeded per-class Dirichlet draw followed by multinomial
allocation. Empty clients receive one sample from the largest partition. Every
sample is used once; published partition files were not recovered. The paper's
α=10/0.1 settings are provided, with one local epoch, 30 clients, and full/20%
participation. Three independent seeds are required for comparable averages;
only bounded correctness fixtures were run here.

CLIP ViT-B/32 and ViT-L/14, DINOv2 Base and Small use immutable Hugging Face
revision defaults recorded in `model.py` (queried from the model owners on
2026-10-15). `model-revision` can explicitly override them. CLIP uses its visual
projection; DINOv2 uses its CLS representation. ConvMixer-768/32 uses timm's
`convmixer_768_32.in1k`: supply `model-checkpoint` or explicitly enable downloads.
That checkpoint is ImageNet supervised pretraining, and its correspondence to
the paper's unspecified ConvMixer checkpoint is unverified. Record the file's
SHA-256 when running it. Full pretrained checkpoints were not downloaded for
release validation. Only the final five blocks are masked.

## Measurements and operational limits

JSON results report bytes actually present in each mask payload, including
metadata and PNG. Uplink bpp divides those bytes by participating clients and
masked parameter count. Separate fields report classifier-probe uplink and
per-round score/head downlinks (Flower record sizes). TLS/HTTP framing,
metrics/control records, initial pretrained checkpoint downloads and deployment
setup are not included. These are application-level counters, not packet-capture
measurements. Tiny fixtures can exceed 1 bpp because headers dominate.

The sequential command calls real Flower application handlers through an
in-process test transport. `flwr run` uses Flower's Ray simulation backend with
isolated clients. Neither is a secure production deployment: no adversarial
client validation, privacy mechanism, or authenticated network rollout is claimed.
Every selected client must reply successfully; dropout recovery is unsupported.

Flower 1.40's Alembic path construction fails when its installed site-packages
path contains spaces; use an actual virtual environment in a path without spaces
(a symlink is insufficient). Set `FLWR_HOME` to keep local simulation state inside
the project. Local Flower services may remain running after a CLI run; manage
them through Flower's own local-service controls when finished.

## Validation

`pytest -q` exercises real four-wise filter construction/membership, observed
false positives, PNG roundtrip/corruption detection, seed/round separation,
Bernoulli KL, prior resets, disjoint partitions, frozen weight updates, exact
native source packaging, and two deterministic federated rounds through both
Flower app handlers. With the vision extra installed, a small real six-block
CLIP transformer checks that only its last five blocks are masked and that
mask gradients propagate while backbone gradients remain absent. A convolution
fixture checks depthwise layer masking. These tests do not certify security or
reproduce large image benchmark results.

The 2026-10-15 validation used Python 3.12.14, Torch 2.8.0 and Flower 1.40.0 on
macOS arm64. All 17 tests passed with vision dependencies installed; a separate
non-editable wheel installation passed 16 tests and skipped the optional CLIP
fixture. The real `flwr run` Ray simulation completed two rounds with two clients,
matching the sequential handler run: 631 and 254 uplink payload bytes, accuracies
0.359375 and 0.328125, and final-score SHA-256
`6d5d36de0776b5030a84dd36865c7c5b03e3e418cbe2e76e9f2dbec60a8f2066`.
These are correctness-fixture measurements, not paper results.

The output `.npz` contains `scores`, `head.weight`, and `head.bias`. To evaluate a
saved run, rebuild `make_model` with its JSON configuration and class count, load
these head tensors, call `enable_masks()`, then `set_mask_scores(model, scores)`.
The frozen pretrained weights remain external and are identified by the model
revision; optimizer state and automatic training resumption are not provided.

## Dataset setup follow-up

[DATASETS.md](DATASETS.md) defines the maintained preparation contract. Training now consumes `ROOT/DATASET` roots created by `scripts/prepare_data.py`; older direct CIFAR roots can be imported with `--source`. The preparation manifest identifies class maps, split sizes, label order and file hashes. Reports include its SHA256. Generated source-format fixtures exercise import, validation and actual training; their metrics are correctness diagnostics. No newly implemented ingestion path implies recovered historical splits or reproduced benchmark results.

The supported CLIP/DINO loaders also accept a local Hugging Face `save_pretrained` directory through `model-checkpoint`. This permits offline integration checks with a small random CLIP model while retaining real CLIP modules, mask gradients, filter transport and Bayesian aggregation. Such a model is a test fixture, not the paper's pretrained checkpoint. Vision libraries are base application dependencies because native Flower installs an isolated app environment; host-only extras were insufficient.

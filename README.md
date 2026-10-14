# DeltaMask

**Communication-efficient federated fine-tuning through stochastic masks.**

[Paper](https://doi.org/10.1109/BigData62323.2024.10825398) · [Author manuscript](https://pure.tue.nl/ws/portalfiles/portal/349909071/2024337517.pdf) · [Project](https://nousphera.github.io/DeltaMask/) · [Poster](https://nousphera.github.io/DeltaMask/poster.pdf) · [Reproducibility](REPRODUCIBILITY.md)

DeltaMask learns binary subnetworks over a frozen pretrained backbone. Clients rank changed mask positions by Bernoulli KL divergence, encode the selected positions in a **four-wise BFuse8 filter**, and send its fingerprints as a losslessly compressed PNG. The server queries the received filter, flips a shared reference mask, and updates global probabilities by Bayesian aggregation.

This release recovers the authors’ training design and implements an actual client/server byte protocol. It corrects gaps in the recovered experimental scripts; it does **not** claim to reproduce the paper’s accuracy or communication tables. The compact offline example uses a small randomly initialized network to check correctness. [Code-to-paper mapping and differences →](REPRODUCIBILITY.md)

## Install and run

Python 3.12 and a C++17 compiler are tested. Linux and macOS are supported; no model or dataset downloads occur during installation or the smoke test.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints.txt -r requirements.txt
pytest -q
deltamask-train --config configs/smoke.toml
```

The two-round example runs the same Flower `ClientApp` and `ServerApp` handlers through a bounded in-process transport. It writes `results/smoke.json`, including configuration, package versions, accuracy, measured update bytes, downlink score bytes, classifier setup cost, and a deterministic final-score digest. The matching `.npz` stores global mask scores and the probed classifier without pickle; existing results are preserved, so use a new `--output` path for another run. Small masks have substantial message-header overhead.

## Flower simulation

The application uses Flower 1.40’s message API. Clients return `ConfigRecord` bytes during mask training; global score arrays are never returned to the server. The custom server loop performs DeltaMask reconstruction and Bayesian aggregation.

Install the optional simulation dependencies in an environment whose **absolute path contains no spaces** (a Flower 1.40 migration-path limitation). Activate it before running:

```bash
python -m pip install -c constraints.txt -e '.[test,simulation]'
export FLWR_HOME="$PWD/.flwr"
flwr run . local --federation-config 'num-supernodes=2 client-resources-num-cpus=1 client-resources-num-gpus=0.0 init-args-num-cpus=2' --stream
```

For larger experiments, set both `num-clients` and `num-supernodes` to the same value. Native filter sources are preserved verbatim in a generated JSON bundle because Flower bundles omit C++ file extensions. See [provenance](PROVENANCE.md).

## Image experiments

```bash
python -m pip install -c constraints.txt -e '.[vision]'
deltamask-train --config configs/clip-cifar10-iid.toml --download
```

That command explicitly downloads CIFAR-10 and the pinned CLIP checkpoint and runs the full configured experiment. Set `device = "cpu"` when CUDA is unavailable. `clip-cifar10-noniid.toml` provides the 30-client, 20% participation, Dirichlet α=0.1 configuration. Use `--rounds 2` for a bounded integration check; full runs are expensive and were not executed for this release.

Available backbones: CLIP ViT-B/32 and ViT-L/14, DINOv2 Base and Small, and ConvMixer-768/32. Available datasets: CIFAR-10/100, SVHN, balanced EMNIST, Fashion-MNIST, Food-101, and explicit ImageFolder splits for EuroSAT/Cars196. Dataset splits, checkpoint revisions, initialization, sampling, and evaluation choices are documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Citation and provenance

Vasileios Tsouvalas, Yuki M. Asano, and Aaqib Saeed. **DeltaMask: Minimizing Communication in Federated Fine-Tuning of Vision Foundation Models.** IEEE Big Data 2024, pp. 8042–8051; published January 2025. DOI: [10.1109/BigData62323.2024.10825398](https://doi.org/10.1109/BigData62323.2024.10825398). See [CITATION.cff](CITATION.cff).

The inherited repository and recovered first-party source contain no license grant; this release does not invent one. The vendored FastFilter implementation retains its Apache-2.0 license. See [PROVENANCE.md](PROVENANCE.md) for source hashes and attribution. Mask compression provides no encryption or differential-privacy guarantee.

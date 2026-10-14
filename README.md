# DeltaMask

**DeltaMask: Minimizing Communication in Federated Fine-Tuning of Vision Foundation Models**

[Paper](https://doi.org/10.1109/BigData62323.2024.10825398) · [Accepted manuscript](https://pure.tue.nl/ws/portalfiles/portal/349909071/2024337517.pdf) · [Project](https://nousphera.github.io/DeltaMask/) · [Citation](CITATION.cff)

## Abstract

Foundation Models (FMs) have revolutionized machine learning with their adaptability and high performance across tasks; yet, their integration into Federated Learning (FL) is challenging due to substantial communication overhead from their extensive parameterization. Current communication-efficient FL strategies, such as gradient compression, reduce bitrates to around 1 bit-per-parameter (bpp). However, these approaches fail to harness the characteristics of FMs, with their large number of parameters still posing a challenge to communication efficiency, even at these bitrate regimes. In this work, we present DeltaMask, a novel method that efficiently fine-tunes FMs in FL at an ultra-low bitrate, well below 1 bpp. DeltaMask employs stochastic masking to detect highly effective subnetworks within FMs and leverage stochasticity and sparsity in client masks to compress updates into a compact grayscale image using probabilistic filters, deviating from traditional weight training approaches. Our comprehensive evaluations across various datasets and architectures demonstrate DeltaMask efficiently achieves bitrates as low as 0.09 bpp, enhancing communication efficiency while maintaining FMs performance, as measured on 8 datasets and 5 pre-trained models of various network architectures.

*The abstract describes the paper's results. This release implements stochastic masks, four-wise BFuse8 filters, PNG transport and Bayesian aggregation, but does not claim to reproduce the paper's accuracy or communication tables.* See [implementation scope](REPRODUCIBILITY.md).

## Setup

Python 3.12 and a C++17 compiler are tested. Package metadata permits Python 3.11–3.13; Linux/macOS are supported targets and native Windows is untested (use WSL2). Linux needs a compiler and Python development headers, for example `build-essential python3-dev`. macOS needs Python 3.12 and installed Command Line Tools. The real four-wise Binary Fuse extension builds from the bundled C++ source; no prebuilt Linux binary is redistributed.

Use a dedicated virtual environment whose **absolute path has no spaces**, required by Flower 1.40's migration/runtime path handling:

```bash
git clone https://github.com/Nousphera/DeltaMask.git
cd DeltaMask
bash scripts/setup.sh "$HOME/.venvs/deltamask"
source "$HOME/.venvs/deltamask/bin/activate"
python -m pytest -q
```

The setup script creates the venv, installs the constraints-pinned requirements including Flower simulation and vision dependencies, and runs `pip check`. It uses installed standalone macOS Command Line Tools without accepting an Xcode license. Manual equivalent:

```bash
python3.12 -m venv "$HOME/.venvs/deltamask"
source "$HOME/.venvs/deltamask/bin/activate"
# macOS with standalone Command Line Tools:
# export DEVELOPER_DIR=/Library/Developer/CommandLineTools
python -m pip install -c constraints.txt -r requirements.txt
```

Installation and the smoke test download no datasets or model weights. Vision libraries are application dependencies so Flower's isolated runtime also installs them. After pulling changes, rerun installation to replace any older installed package.

## Datasets

Prepare official CIFAR data separately from model acquisition:

```bash
python scripts/prepare_data.py --dataset cifar10 --root data --download
python scripts/prepare_data.py --dataset cifar10 --root data --validate
```

The output is `data/cifar10/`, containing torchvision's original 50,000/10,000 train/test split and `manifest.json` with SHA256 hashes, class mapping and split sizes. Imports use `--source` instead of `--download`; existing dataset roots are protected. [DATASETS.md](DATASETS.md) documents every consumed layout and provider/access requirement.

| Dataset option | Acquisition and split |
|---|---|
| `cifar10`, `cifar100` | Official torchvision download; original train/test split |
| `svhn`, `fashion-mnist`, `emnist`, `food101` | Explicit torchvision download; EMNIST balanced variant |
| `eurosat`, `cars196`, `imagefolder` | Import acquired `train/CLASS/*` and `test/CLASS/*` folders with fixed splits |

## Run

Launch the actual Flower `ClientApp` and `ServerApp` with two local SuperNodes:

```bash
mkdir -p results
export FLWR_HOME="$HOME/.flwr-deltamask"
flwr run . local \
  --federation-config 'num-supernodes=2 client-resources-num-cpus=1 client-resources-num-gpus=0.0 init-args-num-cpus=2' \
  --run-config "output=\"$PWD/results/flower.json\"" --stream
```

This offline, two-round CPU smoke run trains a tiny random network, sends encoded mask bytes and applies the method's server aggregation. The default models/data are diagnostic fixtures, not pretrained benchmark models. `num-clients` and `num-supernodes` must agree. `FLWR_HOME` stores runtime state; changing it alone does not isolate a previously running local SuperLink's ports. Run one local simulation service at a time or configure distinct SuperLink addresses.

A prepared CIFAR experiment uses the pinned CLIP checkpoint. `allow-download=true` permits its initial model download. The following is a bounded integration configuration; it still requires downloading a large pretrained model and was not benchmarked:

```bash
flwr run . local \
  --federation-config 'num-supernodes=2 client-resources-num-cpus=1 client-resources-num-gpus=0.0 init-args-num-cpus=2' \
  --run-config "dataset=\"cifar10\" model=\"clip-vit-b32\" data-root=\"$PWD/data\" num-clients=2 num-rounds=2 device=\"cpu\" allow-download=true output=\"$PWD/results/clip-cifar10-check.json\"" --stream
```

For the full configured experiment, use `--run-config configs/clip-cifar10-iid.toml` (30 clients) or `configs/clip-cifar10-noniid.toml` (30 clients, 20% participation, Dirichlet α=0.1), followed by overrides for absolute data/output paths and available device. Set `num-supernodes=30`; CPU/GPU allocations must match your machine. A GPU configuration needs an appropriate CUDA PyTorch build and explicit GPU client resources. Full configured experiments have not been executed for this release.

The offline handler runner is also available:

```bash
deltamask-train --config configs/smoke.toml --output results/offline.json
deltamask-train --help
```

It uses the same app handlers through an in-process transport. Supported pretrained backbones are CLIP ViT-B/32 and ViT-L/14, DINOv2 Base and Small, and ConvMixer-768/32. [DATASETS.md](DATASETS.md) explains cached/local checkpoints. Use absolute paths with native Flower because its installed bundle has its own working directory.

## Outputs and reproducibility

The JSON report saves configuration, input-manifest hash, package versions, accuracy, update bytes, score downlink bytes, classifier setup traffic and a final-score digest. A sibling `.npz` contains mask scores and the probed classifier without pickle. Existing output/checkpoint paths are preserved; choose another name to rerun. Run data validation before experiments; training checks split and label identity while full file hashes are verified explicitly by preparation.

The native protocol sends four-wise BFuse8 fingerprints compressed as PNG, not client score arrays. Small fixtures have substantial protocol-header overhead. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the paper/code mapping, initialization, sampling, model revisions and evaluation differences. Mask compression provides no encryption or differential-privacy guarantee.

## Citation

```bibtex
@inproceedings{tsouvalas2025deltamask,
  title = {DeltaMask: Minimizing Communication in Federated Fine-Tuning of Vision Foundation Models},
  author = {Vasileios Tsouvalas and Yuki M. Asano and Aaqib Saeed},
  booktitle = {2024 IEEE International Conference on Big Data (BigData)},
  year = {2025},
  pages = {8042--8051},
  publisher = {IEEE},
  doi = {10.1109/BigData62323.2024.10825398},
  url = {https://doi.org/10.1109/BigData62323.2024.10825398}
}
```

## License and attribution

The inherited repository and recovered first-party source contain no license grant; this release does not invent one. Vendored FastFilter retains its Apache-2.0 license. [PROVENANCE.md](PROVENANCE.md) records source hashes and attribution. Native source is also preserved in a generated JSON bundle because Flower bundles omit C++ extensions; the package reconstructs those files for compilation.

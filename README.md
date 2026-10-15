# DeltaMask

**DeltaMask: Minimizing Communication in Federated Fine-Tuning of Vision Foundation Models**

[<img src="docs/assets/icons/paper.svg" width="16" height="16" alt=""> Paper](https://doi.org/10.1109/BigData62323.2024.10825398) · [<img src="docs/assets/icons/project.svg" width="16" height="16" alt=""> Project](https://nousphera.github.io/DeltaMask/) · [<img src="docs/assets/icons/code.svg" width="16" height="16" alt=""> Code](https://github.com/Nousphera/DeltaMask) · [Citation](CITATION.cff)

Vasileios Tsouvalas · Yuki M. Asano · Aaqib Saeed

IEEE BigData 2024 · Published January 2025

## Abstract

Foundation Models (FMs) have revolutionized machine learning with their adaptability and high performance across tasks; yet, their integration into Federated Learning (FL) is challenging due to substantial communication overhead from their extensive parameterization. Current communication-efficient FL strategies, such as gradient compression, reduce bitrates to around 1 bit-per-parameter (bpp). However, these approaches fail to harness the characteristics of FMs, with their large number of parameters still posing a challenge to communication efficiency, even at these bitrate regimes. In this work, we present DeltaMask, a novel method that efficiently fine-tunes FMs in FL at an ultra-low bitrate, well below 1 bpp. DeltaMask employs stochastic masking to detect highly effective subnetworks within FMs and leverage stochasticity and sparsity in client masks to compress updates into a compact grayscale image using probabilistic filters, deviating from traditional weight training approaches. Our comprehensive evaluations across various datasets and architectures demonstrate DeltaMask efficiently achieves bitrates as low as 0.09 bpp, enhancing communication efficiency while maintaining FMs performance, as measured on 8 datasets and 5 pre-trained models of various network architectures.

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

The setup script creates the venv, installs the constraints-pinned requirements including Flower simulation and vision dependencies, and runs `pip check`. On macOS it selects the installed standalone Command Line Tools. Manual equivalent:

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
export FLWR_LOCAL_SUPERLINK_HTTP_API_PORT=39134
flwr run . local \
  --federation-config 'num-supernodes=2 client-resources-num-cpus=1 client-resources-num-gpus=0.0 init-args-num-cpus=2' \
  --run-config "output=\"$PWD/results/flower.json\"" --stream
```

This offline, two-round CPU smoke run trains a tiny random network, sends encoded mask bytes and applies the method's server aggregation. The default configuration uses generated vectors and a randomly initialized tiny model; no external assets are required. `num-clients` and `num-supernodes` must agree. `FLWR_HOME` stores runtime state; changing it alone does not isolate a previously running local SuperLink's ports. Run one local simulation service at a time or configure distinct SuperLink addresses.

A prepared CIFAR experiment uses the pinned CLIP checkpoint. `allow-download=true` permits its initial model download. Start with two clients and two rounds; the pretrained model download requires additional disk space:

```bash
flwr run . local \
  --federation-config 'num-supernodes=2 client-resources-num-cpus=1 client-resources-num-gpus=0.0 init-args-num-cpus=2' \
  --run-config "dataset=\"cifar10\" model=\"clip-vit-b32\" data-root=\"$PWD/data\" num-clients=2 num-rounds=2 device=\"cpu\" allow-download=true output=\"$PWD/results/clip-cifar10-check.json\"" --stream
```

For the full configured experiment, use `--run-config configs/clip-cifar10-iid.toml` (30 clients) or `configs/clip-cifar10-noniid.toml` (30 clients, 20% participation, Dirichlet α=0.1), followed by overrides for absolute data/output paths and available device. Set `num-supernodes=30`; CPU/GPU allocations must match your machine. A GPU configuration needs an appropriate CUDA PyTorch build and explicit GPU client resources.

The offline handler runner is also available:

```bash
deltamask-train --config configs/smoke.toml --output results/offline.json
deltamask-train --help
```

It uses the same app handlers through an in-process transport. Supported pretrained backbones are CLIP ViT-B/32 and ViT-L/14, DINOv2 Base and Small, and ConvMixer-768/32. [DATASETS.md](DATASETS.md) explains cached/local checkpoints. Use absolute paths with native Flower because its installed bundle has its own working directory.

## Outputs

The JSON report saves configuration, input-manifest hash, package versions, accuracy, update bytes, score downlink bytes, classifier setup traffic and a final-score digest. A sibling `.npz` contains mask scores and the probed classifier without pickle. Existing output/checkpoint paths are preserved; choose another name to rerun. Run data validation before experiments; training checks split and label identity while full file hashes are verified explicitly by preparation.

The native protocol sends four-wise BFuse8 fingerprints compressed as PNG, not client score arrays. Small fixtures have substantial protocol-header overhead. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for initialization, sampling, model revisions, evaluation and byte accounting. Payloads contain visible filter fingerprints and public sampling metadata.

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

First-party code has no repository-wide license grant. Vendored FastFilter is Apache-2.0; its license applies to that component. See [attribution and licenses](PROVENANCE.md). Native C++ source is included in a generated JSON bundle for compilation inside Flower application bundles.

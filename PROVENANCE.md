# Provenance

This repository is an independent Nousphera copy of [DeltaMask](https://github.com/FederatedML/DeltaMask). The original FederatedML repository remains in place.

- Source commit: [`ca719d327124d4f31b2f351046d76535b5178a20`](https://github.com/FederatedML/DeltaMask/commit/ca719d327124d4f31b2f351046d76535b5178a20)
- Source branch: `main`
- Baseline tag in this repository: `federatedml-baseline`
- Original Git history and author attribution are retained.

Maintenance changes after this baseline improve documentation, setup, and validation. They do not constitute a new reproduction of the published experiments. For the original code and configuration, inspect the baseline tag.

GitHub issues, pull requests, stars, releases, and Pages configuration are not part of Git history and have not been transferred. Existing discussions remain at the source repository.

## Implementation recovery, 15 October 2026

The original baseline was a project reference without executable code. The new
implementation was prepared from an author-supplied archive, principally
`code/fedmft/src/{compress,client,server,main,data,utils}.py` and
`code/fedmft/src/model/{network,vit}.py`. Original SHA-256 hashes are recorded in
[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json); the private archive, datasets,
checkpoints, machine paths, and logs are not redistributed.

The recovered code supplied the stochastic-masking design, CLIP layer selection,
classifier masking, Bayesian reset rule, and cosine top-k schedule. It sent full
mask scores and simulated compression on the server. The new package implements
client-produced messages and corrects full Bernoulli KL, sampling stream reuse,
filter arity, and malformed-message handling. Original scripts are not exposed
as alternative runnable methods. Numerical trajectories therefore differ.

### Four-wise filter dependency

The recovered `pyfilters` package metadata names Vasileios Tsouvalas and declares
Apache-2.0, but no standalone wrapper license was recovered. Its binary-fuse
header allocates using arity four while construction/query use three hashes.
This release does not redistribute or depend on that wrapper.

Instead, native sources come from
[FastFilter/fastfilter_cpp](https://github.com/FastFilter/fastfilter_cpp), commit
`7c5bf187d5096ccbcec6a16e02975d0af3328b01`, with its original
[Apache-2.0 license](src/deltamask/native/fastfilter/LICENSE), copyright 2018 the
fastfilter_cpp authors. The four-wise low-memory implementation is preserved,
with marked changes for bounded retries, explicit overflow errors, and safe
small-size/include handling. New bindings use the same MurmurHash3 mixer with an
explicit deterministic seed and validate serialized filter geometry. No false
negatives are accepted after construction.

`native/sources.json` is a generated, exact text copy of the native sources and
license for Flower's restricted FAB file formats. `scripts/bundle_native.py`
regenerates it; a test checks every bundled file against its canonical source.
Builds reconstruct the C++ files from that bundle only when FAB omitted them.
No prebuilt platform binary is committed.

No first-party license was found in the baseline or recovered source; the
third-party Apache license is not a grant over the whole project. Attribution
and paper copyright are separate from a repository reuse license.

## Dataset/setup follow-up (2026-10-15)

The maintained release adds `scripts/setup.sh`, `scripts/prepare_data.py`, namespaced dataset preparation/validation, manifests and generated-source integration tests. These modules are new integration code rather than recovered historical experiment scripts. Official dataset loading remains delegated to torchvision; fixed-split imports remain user-supplied. See `DATASETS.md` for split conventions and `REPRODUCIBILITY.md` for method/benchmark limits.

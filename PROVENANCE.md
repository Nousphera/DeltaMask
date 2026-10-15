# Attribution and licenses

## Source attribution

DeltaMask is the work of Vasileios Tsouvalas, Yuki M. Asano and Aaqib Saeed. This repository retains the history of [FederatedML/DeltaMask](https://github.com/FederatedML/DeltaMask), including commit `ca719d327124d4f31b2f351046d76535b5178a20` under the `federatedml-baseline` tag.

The implementation draws on author-supplied DeltaMask source for stochastic masking, CLIP layer selection, classifier masking, Bayesian aggregation and the cosine selection schedule. [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) records original source-file hashes. The Python package provides Flower applications, client-produced filter payloads, native bindings, dataset preparation and tests.

## Native filter component

The native filter source is [FastFilter/fastfilter_cpp](https://github.com/FastFilter/fastfilter_cpp), commit `7c5bf187d5096ccbcec6a16e02975d0af3328b01`, copyright 2018 the fastfilter_cpp authors, under its [Apache-2.0 license](src/deltamask/native/fastfilter/LICENSE). Modification notices identify bounded construction retries, explicit overflow errors and safe small-size/include handling. Bindings use deterministic-seed MurmurHash3 mixing and validate serialized geometry; every inserted key is checked after construction.

`src/deltamask/native/sources.json` is an exact text bundle of the C++ source and license for Flower application packaging. Regenerate it with `python scripts/bundle_native.py`. A test checks every bundled file against its canonical source. Builds extract the source bundle when Flower packaging omits the C++ files; compilation produces the platform-specific extension.

## License records

First-party DeltaMask code has no repository-wide license grant. The Apache-2.0 license applies to the vendored FastFilter component. Paper copyright, dataset access and pretrained-model terms are separate from code licensing.

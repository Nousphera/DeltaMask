# Contributing

Include the command, configuration, Python version and relevant error output when reporting an issue. For dataset issues, include the preparation command and manifest summary without sharing private records.

Install the environment with `scripts/setup.sh`, make focused changes and run:

```bash
python -m pip check
python -m pytest -q
```

Keep dataset files, credentials, generated results and model checkpoints out of Git. Include regressions for changes to training, partitioning, aggregation or serialization, and update the corresponding setup or experiment instructions.

Preserve source attribution and component licenses in [PROVENANCE.md](PROVENANCE.md). When changing native code, verify the source build as well as Python tests.

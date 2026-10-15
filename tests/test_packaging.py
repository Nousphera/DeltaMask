import json
from pathlib import Path


def test_flower_source_bundle_exactly_matches_vendored_sources_and_terms():
    project = Path(__file__).resolve().parents[1]
    root = project / "src/deltamask/native"
    sources = json.loads((root / "sources.json").read_text())
    for name in ("LICENSE", "THIRD_PARTY_LICENSES"):
        assert sources[name] == (project / name).read_text()
    for name, source in sources.items():
        if name not in {"LICENSE", "THIRD_PARTY_LICENSES"}:
            assert (root / name).read_text() == source

import json
from pathlib import Path

def test_flower_source_bundle_exactly_matches_vendored_sources():
    root=Path(__file__).resolve().parents[1]/'src/deltamask/native'
    sources=json.loads((root/'sources.json').read_text())
    assert 'fastfilter/LICENSE' in sources
    assert all((root/name).read_text()==source for name,source in sources.items())

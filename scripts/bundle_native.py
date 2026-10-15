"""Refresh Flower's text-only copy of native sources and required license text."""

import json
from pathlib import Path

project = Path(__file__).resolve().parents[1]
root = project / "src/deltamask/native"
files = {
    str(path.relative_to(root)): path.read_text()
    for path in sorted(root.rglob("*"))
    if path.is_file() and path.suffix in {".h", ".cpp"}
}
for name in ("LICENSE",):
    files[name] = (project / name).read_text()
(root / "sources.json").write_text(json.dumps(files, indent=2) + "\n")

"""Refresh Flower's text-only FAB copy of the native sources and their license."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'src/deltamask/native'
files={str(p.relative_to(root)):p.read_text() for p in sorted(root.rglob('*')) if p.is_file() and p.name!='sources.json'}
(root/'sources.json').write_text(json.dumps(files,indent=2)+'\n')

"""Build the native four-wise BFuse extension, including Flower text-only bundles."""

import json
from pathlib import Path
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

root = Path("src/deltamask/native")
# Flower FAB omits native file extensions; extract the canonical text bundle.
if not (root / "bindings.cpp").exists() or not Path("THIRD_PARTY_LICENSES").exists():
    for name, source in json.loads((root / "sources.json").read_text()).items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid native source path")
        target = Path(name) if name == "THIRD_PARTY_LICENSES" else root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
setup(
    ext_modules=[
        Pybind11Extension("deltamask._bfuse", [str(root / "bindings.cpp")], cxx_std=17)
    ],
    cmdclass={"build_ext": build_ext},
)

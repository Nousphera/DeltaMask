"""Build the native four-wise BFuse extension, including Flower text-only bundles."""
import json
from pathlib import Path
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
root=Path('src/deltamask/native')
# Flower 1.40 FAB accepts JSON/Python but excludes C++/header file extensions.
# Keep an auditable, generated text bundle, including the upstream Apache license.
if not (root/'bindings.cpp').exists():
    for name,source in json.loads((root/'sources.json').read_text()).items():
        target=root/name
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Invalid native source path')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_text(source)
setup(ext_modules=[Pybind11Extension('deltamask._bfuse',[str(root/'bindings.cpp')],cxx_std=17)],cmdclass={'build_ext':build_ext})

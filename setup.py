"""Standard setuptools build, including the native four-wise BFuse extension."""

import json
import os
from pathlib import Path
import sys
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

# Prefer an installed standalone macOS compiler; do not accept SDK licenses.
if sys.platform == "darwin" and Path("/Library/Developer/CommandLineTools").is_dir():
    os.environ.setdefault("DEVELOPER_DIR", "/Library/Developer/CommandLineTools")

native = Path("src/deltamask/native")
# Flower's text-only FAB carries the canonical C++ sources and terms as JSON.
if not (native / "bindings.cpp").exists() or not Path("LICENSE").exists():
    for name, source in json.loads((native / "sources.json").read_text()).items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid native source path")
        target = Path("LICENSE") if name == "LICENSE" else native / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)


class BuildExt(build_ext):
    def build_extensions(self):
        # Imported only after setuptools has installed setup_requires in its
        # isolated build environment; metadata preparation needs no pybind11.
        import pybind11

        flags = ["/std:c++17"] if self.compiler.compiler_type == "msvc" else ["-std=c++17", "-fvisibility=hidden"]
        for extension in self.extensions:
            extension.include_dirs.append(pybind11.get_include())
            extension.extra_compile_args.extend(flags)
        super().build_extensions()


setup(
    name="deltamask",
    version="0.2.0",
    description="DeltaMask stochastic mask learning and four-wise binary fuse transport",
    long_description=Path("README.md").read_text() if Path("README.md").is_file() else "DeltaMask federated training.",
    long_description_content_type="text/markdown",
    license="MIT AND Apache-2.0",
    license_files=("LICENSE",),
    python_requires=">=3.11,<3.14",
    install_requires=[
        "numpy>=2,<3", "scipy>=1.14,<2", "pillow>=11,<13",
        "torch==2.8.0", "torchvision==0.23.0", "flwr[simulation]==1.40.0",
        "transformers==4.57.6", "timm==1.0.30",
    ],
    setup_requires=["pybind11==3.0.2"],
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"deltamask": ["native/*.cpp", "native/*.json", "native/fastfilter/*.h"]},
    data_files=[
        ("share/deltamask", ["setup.py"]),
        ("share/deltamask/configs", [str(p) for p in Path("configs").glob("*.toml")]),
        ("share/deltamask/scripts", [str(p) for p in Path("scripts").glob("*.py")]),
    ],
    entry_points={"console_scripts": [
        "deltamask-train=deltamask.train:main",
        "deltamask-experiment=deltamask.experiment:main",
    ]},
    ext_modules=[Extension("deltamask._bfuse", [str(native / "bindings.cpp")], language="c++")],
    cmdclass={"build_ext": BuildExt},
)

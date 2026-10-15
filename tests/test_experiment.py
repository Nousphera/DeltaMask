"""Launcher options must preserve paper defaults and native node counts."""

from pathlib import Path
import pytest


def test_paper_launcher_resolves_paths_and_native_node_count(tmp_path):
    from deltamask.experiment import parser, build_config, flower_command
    args = parser().parse_args(["--method", "fedpm", "--dataset", "cifar10", "--setting", "noniid",
                               "--device", "cpu", "--clients", "7", "--output", str(tmp_path / "out.json")])
    cfg = build_config(args)
    assert cfg["num-rounds"] == 300 and cfg["num-clients"] == 7
    assert cfg["participation"] == .2 and cfg["alpha"] == .1
    assert cfg["model"] == "clip-vit-b32" and cfg["pretrained"]
    assert all(Path(cfg[k]).is_absolute() for k in ["data-root", "model-root", "output"])
    command = flower_command(args, cfg, tmp_path / "run.toml")
    assert "num-supernodes=7" in command[command.index("--federation-config") + 1]
    assert "--stream" in command and "local" in command


def test_iid_default_and_explicit_cpu_requirement(monkeypatch):
    from deltamask.experiment import parser, build_config
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="device"):
        build_config(parser().parse_args([]))
    cfg = build_config(parser().parse_args(["--device", "cpu"]))
    assert (cfg["num-rounds"], cfg["num-clients"], cfg["participation"], cfg["alpha"]) == (100, 30, 1., 10.)


def test_native_flower_accepts_generated_federation_config(tmp_path):
    from deltamask.experiment import parser, build_config, flower_command
    from flwr.common.config import parse_config_args
    from flwr.supercore.utils import simulation_config_from_json
    args = parser().parse_args(["--device", "cpu"])
    command = flower_command(args, build_config(args), tmp_path / "run.toml")
    text = command[command.index("--federation-config") + 1]
    values = parse_config_args([text])
    native = simulation_config_from_json({k.replace("-", "_"): v for k, v in values.items()})
    assert native.num_supernodes == 30
    assert native.client_resources_num_cpus == 1

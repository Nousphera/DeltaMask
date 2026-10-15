import json
from deltamask.config import defaults
from deltamask.train import run


def test_two_round_actual_clientapp_serverapp_byte_flow(tmp_path, image_check_config):
    cfg = image_check_config
    cfg["output"] = str(tmp_path / "run.json")
    result = run(cfg)
    assert len(result["rounds"]) == 2
    assert all(r["payload_bytes"] > 0 for r in result["rounds"])
    assert result["mask_dimension"] > 0
    assert result["probe_uplink_bytes"] > 0
    assert json.loads((tmp_path / "run.json").read_text()) == result
    cfg["output"] = str(tmp_path / "repeat.json")
    repeated = run(cfg)
    assert result["rounds"] == repeated["rounds"]
    assert result["final_score_sha256"] == repeated["final_score_sha256"]


def test_flower_uint64_node_ids_are_not_rounded_to_floats(tmp_path, image_check_config):
    from flwr.app import Context, RecordDict
    from flwr.supercore.task_identity import TaskIdentity
    from deltamask.client_app import app as client_app
    from deltamask.server_app import run_server

    cfg = image_check_config
    cfg["output"] = str(tmp_path / "largeids.json")
    nodes = [2**63 - 19, 2**64 - 123]

    class Grid:
        def get_node_ids(self):
            return nodes

        def send_and_receive(self, messages, timeout=None):
            for msg in messages:
                cid = nodes.index(msg.metadata.dst_node_id)
                yield client_app(
                    msg,
                    Context(
                        run_id=1,
                        node_id=nodes[cid],
                        node_config={"partition-id": cid},
                        state=RecordDict(),
                        run_config=cfg,
                    ),
                )

    TaskIdentity.run_id = 1
    TaskIdentity.node_id = 0
    TaskIdentity.task_id = 0
    result = run_server(Grid(), cfg)
    assert len(result["rounds"]) == 2


def test_existing_outputs_are_preserved_and_checkpoint_is_reusable(
    tmp_path, image_check_config
):
    import numpy as np
    import pytest

    cfg = image_check_config
    cfg["output"] = str(tmp_path / "result.json")
    result = run(cfg)
    with np.load(tmp_path / "result.npz", allow_pickle=False) as state:
        assert state["scores"].shape == (result["mask_dimension"],)
        assert "head.weight" in state and "head.bias" in state
    with pytest.raises(FileExistsError):
        run(cfg)

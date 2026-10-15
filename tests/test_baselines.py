"""Training and transport regressions, using full official model architectures."""

import json
import numpy as np
import pytest
import torch

from deltamask.aggregate import BayesianMask
from deltamask.model import Bernoulli, make_model, mask_scores, set_mask_scores


def test_recovered_bernoulli_backward_scales_by_probability():
    p = torch.tensor([.2, .8], requires_grad=True)
    Bernoulli.apply(p).sum().backward()
    torch.testing.assert_close(p.grad, p.detach())


def test_bayesian_observations_do_not_saturate_after_one_round():
    state = BayesianMask(3, participation=.5)
    np.testing.assert_allclose(state.update([[1, 0, 1]], 1), [.99, .01, .99])
    np.testing.assert_allclose(state.update([[0, 1, 1]], 2), [.01, .99, .99])
    np.testing.assert_allclose(state.update([[1, 0, 1]], 3), [.5, .5, .99])


def test_gaussian_masks_are_seeded_independently_and_reject_partial_mutation(image_check_config):
    model = make_model(image_check_config, 3)
    torch.manual_seed(999)
    model.enable_masks()
    before = mask_scores(model).copy()
    assert .9 < before.std() < 1.1
    other = make_model(image_check_config, 3)
    torch.manual_seed(1000)
    other.enable_masks()
    np.testing.assert_array_equal(before, mask_scores(other))
    del other
    for bad in (np.ones(len(before) + 1), np.ones(len(before) - 1),
                np.full(len(before), np.nan), before.reshape(1, -1)):
        with pytest.raises(ValueError):
            set_mask_scores(model, bad)
        np.testing.assert_array_equal(before, mask_scores(model))


def test_threshold_forward_and_backward():
    from deltamask.model import Threshold
    p = torch.tensor([.2, .5, .8], requires_grad=True)
    out = Threshold.apply(p)
    torch.testing.assert_close(out, torch.tensor([0., 0., 1.]))
    out.sum().backward()
    torch.testing.assert_close(p.grad, out)


def test_finetune_updates_last_five_linear_blocks_and_preserves_frozen(image_check_config):
    from deltamask.data import load_data
    from deltamask.model import train_local
    cfg = image_check_config | {"method": "finetune", "learning-rate": .001}
    model = make_model(cfg, 3)
    model.configure_method(cfg["method"])
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert "backbone.encoder.layer.7.attention.attention.query.weight" in trainable
    assert "backbone.encoder.layer.6.attention.attention.query.weight" not in trainable
    assert all("norm" not in n for n in trainable)
    assert "head.weight" in trainable
    ds, _, _ = load_data(cfg)
    train_local(model, ds, cfg, 42)
    assert any(not torch.equal(p, before[n]) for n, p in model.named_parameters()
               if n.startswith("backbone.") and n in trainable)
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters()
               if n not in trainable)


def test_independent_packed_bernoulli_transport_and_validation():
    from deltamask.codec import encode_binary, decode_binary, sample_client_mask
    p = np.full(10001, .5)
    a = sample_client_mask(p, 42, 1, 0)
    b = sample_client_mask(p, 42, 1, 1)
    assert .45 < np.mean(a != b) < .55
    wire, stats = encode_binary(a, seed=42, round_id=1, client_id=0)
    np.testing.assert_array_equal(decode_binary(wire, len(p), seed=42,
                                              round_id=1, client_id=0), a)
    assert stats["payload_bytes"] == len(wire)
    assert len(wire) < len(p) / 8 + 512
    for invalid in (wire[:-1], wire + b"x", b"bad"):
        with pytest.raises(ValueError):
            decode_binary(invalid, len(p), seed=42, round_id=1, client_id=0)
    with pytest.raises(ValueError):
        decode_binary(wire, len(p), seed=42, round_id=1, client_id=1)


def test_registration_wait_accepts_delayed_nodes_and_is_bounded():
    from deltamask.server_app import wait_for_nodes
    class Grid:
        def __init__(self):
            self.calls = 0
        def get_node_ids(self):
            self.calls += 1
            return [] if self.calls == 1 else [2**63 + 1, 5]
    assert wait_for_nodes(Grid(), 2, timeout=.1, poll_interval=.001) == [5, 2**63 + 1]
    with pytest.raises(TimeoutError):
        wait_for_nodes(Grid(), 3, timeout=.005, poll_interval=.001)


@pytest.mark.parametrize("method", ["deltamask", "fedpm", "fedmask", "finetune", "linear-probe"])
def test_all_methods_save_restore_evaluate_and_record_bytes(tmp_path, image_check_config, method):
    from deltamask.train import run
    from deltamask.evaluate import evaluate_result, restore_checkpoint
    cfg = image_check_config | {"method": method, "num-rounds": 1, "save-updates": True,
                               "output": str(tmp_path / (method + ".json"))}
    result = run(cfg)
    assert result["method"] == method
    assert result["rounds"][0]["payload_bytes"] > 0
    assert np.isfinite(result["rounds"][0]["train_loss"])
    assert result["communication_dimension"] > 0
    assert result["model_identity"]
    restored = evaluate_result(cfg["output"])
    for key in ("loss", "accuracy"):
        assert restored[key] == result["rounds"][-1][key]
    assert list((tmp_path / (method + ".updates")).glob("round-*/client-*"))
    checkpoint = tmp_path / (method + ".npz")
    with np.load(checkpoint, allow_pickle=False) as src:
        state = {k: src[k] for k in src.files}
    target = "scores" if "scores" in state else next(k for k in state if k.startswith("parameter."))
    state[target] = np.full_like(state[target], np.nan)
    np.savez(checkpoint, **state)
    with pytest.raises(ValueError):
        evaluate_result(cfg["output"])
    # Exercise shape/finiteness validation separately from the checksum gate.
    with pytest.raises(ValueError, match="finiteness"):
        restore_checkpoint(make_model(cfg, 3), checkpoint, result)


def test_score_overflow_is_rejected_before_mutation(image_check_config):
    model = make_model(image_check_config, 3)
    model.enable_masks()
    before = mask_scores(model).copy()
    invalid = before.astype(np.float64)
    invalid[-1] = 1e300
    with pytest.raises(ValueError):
        set_mask_scores(model, invalid)
    np.testing.assert_array_equal(mask_scores(model), before)


def test_parameter_transport_weighted_mean_and_validation():
    from deltamask.codec import encode_parameters, decode_parameters
    from deltamask.baselines import WeightedMean
    expected = {"weight": torch.zeros(2, 3), "bias": torch.zeros(2)}
    first = {n: v + 1 for n, v in expected.items()}
    second = {n: v + 3 for n, v in expected.items()}
    mean = WeightedMean(expected)
    for state, count in ((first, 1), (second, 3)):
        wire = encode_parameters(state)
        restored = decode_parameters(wire, expected)
        mean.add(restored, count)
    for value in mean.result().values():
        torch.testing.assert_close(value, torch.full_like(value, 2.5))
    for wrong in (expected | {"weight": torch.zeros(1, 3)},
                  expected | {"weight": torch.full((2, 3), float("nan"))},
                  {"weight": expected["weight"]}):
        with pytest.raises(ValueError):
            decode_parameters(encode_parameters(wrong), expected)


def test_weighted_mean_is_stable_when_replies_arrive_in_different_orders():
    from deltamask.baselines import WeightedMean
    states = [{"weight": torch.tensor([v], dtype=torch.float32)} for v in (1e8, 1., -1e8)]
    results = []
    for order in ((0, 1, 2), (2, 0, 1), (1, 2, 0)):
        mean = WeightedMean({"weight": torch.zeros(1)})
        for i in order:
            mean.add(states[i], 1)
        results.append(mean.result()["weight"])
    for result in results:
        torch.testing.assert_close(result, torch.tensor([1 / 3]))

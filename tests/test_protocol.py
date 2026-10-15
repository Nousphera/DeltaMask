import numpy as np
import pytest
from deltamask.codec import encode, decode, selected_changes, sample_mask, bernoulli_kl
from deltamask.aggregate import BayesianMask
from deltamask import _bfuse


def test_filter_four_hashes_roundtrip_and_false_positives():
    keys = np.arange(0, 20000, 10, dtype=np.uint64)
    meta, fingerprints = _bfuse.build(keys, 42)
    assert meta['arity'] == 4
    queries = np.arange(20000, dtype=np.uint64)
    found = np.asarray(_bfuse.query(queries, meta, fingerprints))
    assert found[keys.astype(int)].all()
    negatives = np.ones(20000, bool); negatives[keys.astype(int)] = False
    assert 0.001 < found[negatives].mean() < 0.01
    assert _bfuse.build(keys, 42) == (meta, fingerprints)


def test_full_bernoulli_kl_and_stable_tie_ranking():
    p = np.array([.1, .9, .5]); q = np.full(3, .5)
    expected = p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))
    np.testing.assert_allclose(bernoulli_kl(p,q), expected)
    assert selected_changes(p,q,np.array([1,1,1],bool), np.zeros(3,bool),2/3).tolist() == [0,1]


def test_bytes_transport_seeded_reference_and_empty_changes():
    q = np.full(10000,.5); p = np.linspace(.01,.99,10000)
    ref = sample_mask(q,42,1); local = sample_mask(p,42,1)
    chosen = selected_changes(p,q,local,ref,1.)
    wire, stats = encode(p,q,seed=42,round_id=1,kappa=1.)
    decoded = decode(wire,q,seed=42,round_id=1)
    assert isinstance(wire,bytes) and len(wire) == stats['payload_bytes']
    assert np.all(decoded[chosen] == local[chosen])
    assert np.mean(decoded != local) < .01
    empty,_ = encode(q,q,seed=42,round_id=1,kappa=1.)
    assert np.array_equal(decode(empty,q,seed=42,round_id=1),ref)
    with pytest.raises(ValueError): decode(wire,q,seed=42,round_id=2)
    with pytest.raises(ValueError): decode(wire[:-8],q,seed=42,round_id=1)
    assert not np.array_equal(sample_mask(q,42,1),sample_mask(q,42,2))
    assert not np.array_equal(ref[:5000],ref[5000:])


def test_bayesian_counts_reset_every_inverse_participation():
    state=BayesianMask(3,participation=.5)
    np.testing.assert_allclose(state.update([np.array([1,0,1])],1),[.99,.01,.99])
    np.testing.assert_allclose(state.update([np.array([0,1,1])],2),[.01,.99,.99])
    np.testing.assert_allclose(state.update([np.array([1,0,1])],3),[.5,.5,.99])

@pytest.mark.parametrize('size',[1,2,3,17,100])
def test_small_filters(size):
    keys=np.arange(size,dtype=np.uint64)
    geometry,raw=_bfuse.build(keys,8)
    assert np.all(_bfuse.query(keys,geometry,raw))
    with pytest.raises(ValueError): _bfuse.query(keys,dict(geometry,array_length=0),raw)

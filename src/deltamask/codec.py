"""Paper Algorithm 1, lines 6–16: actual BFuse8/PNG message transport."""

import hashlib
import io
import json
import math
import struct
import numpy as np
from PIL import Image
from scipy.special import rel_entr
from . import _bfuse

MAGIC = b"DMK1"


def probabilities(value):
    value = np.asarray(value, dtype=np.float64)
    if (
        value.ndim != 1
        or not value.size
        or not np.isfinite(value).all()
        or np.any((value < 0) | (value > 1))
    ):
        raise ValueError("Expected finite flat probabilities in [0,1]")
    return value


def sample_mask(p, seed, round_id):
    p = probabilities(p)
    rng = np.random.default_rng(np.random.SeedSequence([seed, round_id]))
    return rng.random(p.size) < p


def bernoulli_kl(p, q):
    p, q = probabilities(p), probabilities(q)
    if p.shape != q.shape:
        raise ValueError("Probability shapes differ")
    return rel_entr(p, q) + rel_entr(1 - p, 1 - q)


def selected_changes(p, q, local, reference, kappa):
    if not 0 <= kappa <= 1:
        raise ValueError("kappa must lie in [0,1]")
    changed = np.flatnonzero(local != reference)
    # Stable index order breaks exactly equal importance ties.
    ranks = bernoulli_kl(p, q)[changed]
    order = np.lexsort((changed, -np.round(ranks, 15)))
    return changed[order[: math.floor(kappa * len(changed))]]


def context_hash(q, seed, round_id):
    h = hashlib.sha256(np.asarray(q, dtype="<f8").tobytes())
    h.update(struct.pack("<QQ", seed, round_id))
    return h.hexdigest()


def encode(p, q, *, seed, round_id, kappa):
    p, q = probabilities(p), probabilities(q)
    reference = sample_mask(q, seed, round_id)
    # Common random numbers reproduce the recovered implementation's coupling,
    # with one flat stream (no repeated per-layer seed) and a new stream per round.
    local = sample_mask(p, seed, round_id)
    selected = selected_changes(p, q, local, reference, kappa)
    header = {
        "version": 1,
        "dimension": q.size,
        "context": context_hash(q, seed, round_id),
        "selected": len(selected),
    }
    png = b""
    if len(selected):
        geometry, fingerprints = _bfuse.build(
            selected.astype(np.uint64), seed + round_id
        )
        header["filter"] = geometry
        width = min(1024, max(1, math.ceil(math.sqrt(len(fingerprints)))))
        height = math.ceil(len(fingerprints) / width)
        pixels = np.frombuffer(
            fingerprints + bytes(width * height - len(fingerprints)), dtype=np.uint8
        ).reshape(height, width)
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream, format="PNG", optimize=True)
        png = stream.getvalue()
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    wire = MAGIC + struct.pack("<I", len(raw)) + raw + png
    return wire, {
        "payload_bytes": len(wire),
        "png_bytes": len(png),
        "changed": int(np.count_nonzero(local != reference)),
        "selected": len(selected),
    }


def decode(wire, q, *, seed, round_id):
    q = probabilities(q)
    if not isinstance(wire, bytes) or len(wire) < 8 or wire[:4] != MAGIC:
        raise ValueError("Invalid DeltaMask message")
    length = struct.unpack("<I", wire[4:8])[0]
    if length > 8192 or length > len(wire) - 8:
        raise ValueError("Invalid header length")
    try:
        header = json.loads(wire[8 : 8 + length])
        if (
            header["version"] != 1
            or header["dimension"] != q.size
            or header["context"] != context_hash(q, seed, round_id)
        ):
            raise ValueError("Mask context mismatch")
        selected = header["selected"]
        if type(selected) != int or not 0 <= selected <= q.size:
            raise ValueError("Invalid selection count")
        result = sample_mask(q, seed, round_id)
        raw = wire[8 + length :]
        if selected == 0:
            if raw or "filter" in header:
                raise ValueError("Unexpected empty-filter payload")
            return result
        geometry = header["filter"]
        count = geometry["array_length"]
        if not 1 <= count <= max(1024, 20 * q.size):
            raise ValueError("Invalid filter size")
        with Image.open(io.BytesIO(raw)) as image:
            if (
                image.format != "PNG"
                or image.mode != "L"
                or image.width * image.height > count + 1024
                or image.width * image.height < count
            ):
                raise ValueError("Invalid fingerprint image")
            # verify checks PNG integrity including IEND; reopen to decode pixels.
            image.verify()
        with Image.open(io.BytesIO(raw)) as image:
            fingerprints = image.tobytes()[:count]
        flips = _bfuse.query(np.arange(q.size, dtype=np.uint64), geometry, fingerprints)
        return np.logical_xor(result, flips)
    except (
        KeyError,
        TypeError,
        OverflowError,
        OSError,
        SyntaxError,
        struct.error,
    ) as error:
        raise ValueError("Malformed DeltaMask message") from error

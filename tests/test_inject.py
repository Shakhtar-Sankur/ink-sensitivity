"""Offline tests: glyphs, box filter, contrast, and the score/aggregate chain on synthetic data."""

import json
import os
import sys
from argparse import Namespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import inject as I  # noqa: E402


def test_letter_mask_is_bounded_and_cropped():
    m = I.letter_mask("Ω", 120)
    assert m.min() >= 0 and m.max() <= 1
    assert m[0].max() > 0.02 and m[-1].max() > 0.02  # cropped to its ink
    assert 60 < m.shape[0] < 200


def test_box_mean_matches_naive():
    rng = np.random.default_rng(1)
    a = rng.random((37, 41)).astype(np.float64)
    k = 7
    got = I.box_mean(a, k)
    for y in (0, 5, 18, 36):
        for x in (0, 9, 40):
            y0, x0 = max(0, y - k // 2), max(0, x - k // 2)
            want = a[y0:min(37, y0 + k), x0:min(41, x0 + k)].mean()
            assert abs(got[y, x] - want) < 1e-12


def test_contrast_sees_a_planted_bump():
    pred = np.zeros((200, 200), np.float32)
    m = np.zeros((40, 30), bool)
    m[5:35, 10:20] = True
    pred[60:100, 80:110][m] = 0.5
    assert I.contrast(pred, m, 60, 80, 10) > 0.4
    assert abs(I.contrast(pred, m, 120, 20, 10)) < 1e-6


def _synthetic_dir(tmp, seed, strength_gain):
    """Letters planted into a noise map with a response that grows with strength."""
    rng = np.random.default_rng(seed)
    cap, cell = 40, 52
    H = W = cell * 8
    pred = (0.25 + 0.02 * rng.standard_normal((H, W))).astype(np.float32)
    slots = [(y, x) for y in range(0, H - cell, cell) for x in range(0, W - cell, cell)]
    rng.shuffle(slots)
    letters, i = [], 0
    for a in I.STRENGTHS:
        for _ in range(4):
            cy, cx = slots[i]
            m = I.letter_mask(I.LETTERS[i % len(I.LETTERS)], cap) > 0.5
            oy, ox = cy + (cell - m.shape[0]) // 2, cx + (cell - m.shape[1]) // 2
            pred[oy:oy + m.shape[0], ox:ox + m.shape[1]][m] += strength_gain * a
            np.save(os.path.join(tmp, f"mask_{i:03d}.npy"), m)
            letters.append({"char": "x", "alpha": a, "y": oy, "x": ox, "h": m.shape[0], "w": m.shape[1]})
            i += 1
    null_slots = [{"y": y, "x": x} for y, x in slots[i:i + 12]]
    meta = {"target": "synthetic", "shape": [21, H, W], "cap_px": cap, "cell_px": cell,
            "letters": letters, "null_slots": null_slots}
    json.dump(meta, open(os.path.join(tmp, "letters.json"), "w"))
    np.save(os.path.join(tmp, "papyrus.npy"), np.ones((H, W), bool))
    np.save(os.path.join(tmp, "ck0.npy"), np.clip(pred * 255, 0, 255).astype(np.uint8))


def test_score_and_aggregate_recover_strong_letters(tmp_path):
    dirs = []
    for seed in range(3):
        d = tmp_path / f"v{seed}"
        d.mkdir()
        _synthetic_dir(str(d), seed, strength_gain=0.02)
        I.score(Namespace(dir=str(d), pred=[str(d / "ck0.npy")]))
        dirs.append(str(d))
    out = tmp_path / "agg.json"
    I.aggregate(Namespace(dirs=dirs, label="synthetic", out=str(out)))
    agg = json.load(open(out))["mean of all"]
    assert agg["false_positive_rate"] <= 0.10
    rec = [agg[str(a)]["recovered"] for a in I.STRENGTHS]
    assert rec[-1] >= 0.9  # strength 4 is far above the noise
    assert rec[-1] >= rec[0]

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


def test_donor_windows_split_text_from_bare():
    ref = np.zeros((400, 400), np.float32)
    ref[20:120, 20:120] = 0.2  # a text-like block, a fifth of it labelled ink
    ref[40:60, 30:110] = 1.0
    pap = np.ones_like(ref, bool)
    text, bare = I.donor_windows(ref, pap, 0.9, 100)
    assert (20, 20) in text or any(abs(y - 20) <= 50 and abs(x - 20) <= 50 for y, x in text)
    assert bare and all(ref[y:y + 100, x:x + 100].max() < 0.9 for y, x in bare)
    for a in text:  # windows do not overlap
        for b in text:
            assert a == b or abs(a[0] - b[0]) >= 100 or abs(a[1] - b[1]) >= 100


def test_transplant_matches_target_statistics_under_the_mask_only():
    rng = np.random.default_rng(0)
    out = (100 + 10 * rng.standard_normal((5, 60, 60))).astype(np.float32)
    before = out.copy()
    donor = (50 + 2 * rng.standard_normal((3, 20, 20))).astype(np.float32)
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    I.transplant(out, donor, mask, 10, 10, 1, np.full(5, 100.0), np.full(5, 10.0), np.full(3, 50.0), np.full(3, 2.0))
    np.testing.assert_allclose(out[1:4, 15:25, 15:25], 100 + (donor[:, 5:15, 5:15] - 50) * 5, rtol=1e-5)
    changed = out != before
    assert not changed[0].any() and not changed[4].any()  # layers outside the donor's depth
    assert changed[1].sum() == mask.sum()


def test_transplant_scoring_separates_ink_from_bare(tmp_path):
    rng = np.random.default_rng(3)
    dirs = []
    for v in range(4):
        d = tmp_path / f"t{v}"
        d.mkdir()
        size, H = 40, 400
        pred = (0.2 + 0.02 * rng.standard_normal((H, H))).astype(np.float32)
        entries = []
        for i in range(30):
            y, x = 50 + (i // 6) * 60, 20 + (i % 6) * 60
            m = np.zeros((size, size), bool)
            m[10:30, 15:25] = True
            kind = ("ink", "bare", "empty")[i % 3]
            if kind == "ink":
                pred[y:y + size, x:x + size][m] += 0.1
            np.save(d / f"mask_{i:03d}.npy", m)
            entries.append({"kind": kind, "y": y, "x": x, "mask": i})
        meta = {"mode": "transplant", "target": "synthetic", "shape": [21, H, H], "window_px": size, "entries": entries}
        json.dump(meta, open(d / "letters.json", "w"))
        np.save(d / "ck0.npy", np.clip(pred * 255, 0, 255).astype(np.uint8))
        I.score(Namespace(dir=str(d), pred=[str(d / "ck0.npy")]))
        dirs.append(str(d))
    out = tmp_path / "agg.json"
    I.aggregate(Namespace(dirs=dirs, label="synthetic", out=str(out)))
    agg = json.load(open(out))["mean of all"]
    assert agg["recovered"] >= 0.9 and agg["auc_ink_vs_bare"] >= 0.95
    assert agg["empty_above_threshold"] <= 0.2
    assert agg["auc_ink_vs_bare_ci95"][0] <= agg["auc_ink_vs_bare"] <= agg["auc_ink_vs_bare_ci95"][1]

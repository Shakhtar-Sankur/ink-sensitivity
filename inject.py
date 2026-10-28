"""Injection-recovery test for Herculaneum ink detection.

Measures the ink signature on a scroll with known text, plants synthetic
Greek letters of controlled strength into the CT surface volume of another
scroll, and scores how often an ink model recovers them.

  python inject.py signature  --out sig.json
  python inject.py make       --sig sig.json --target 0139|0800 --out DIR
  python inject.py score      --dir DIR --pred a.npy [b.npy ...]

Only NumPy, zarr (v2 API), fsspec and Pillow are needed.
"""

import argparse
import json
import os
import urllib.request

import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont

B = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
W045 = "PHerc0139/segments/20260126000000-w045_2026012619"
W045_VOL = f"{B}/{W045}/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr"
W045_REF = (f"{B}/{W045}/ink-detection/downsampled/PHerc0139-20260126000000-2.399um-0.22m-78keV-volume-"
            "20260102150214-20260417190342-new_canon_autoresearch_recipe-tile256-stride128-ds8.jpg")
# PHerc. 800: eligible for First Letters and the Grand Prize; six published auto-grown segments
P800_SEGS = ["20251028213516-auto_grown_20251028213516907", "20251028220042-auto_grown_20251028220042762",
             "20251028220955-auto_grown_20251028220955262", "20251028222030-auto_grown_20251028222030940",
             "20251028225813-auto_grown_20251028225813045", "20251029010146-auto_grown_20251029010146642"]
P800_VOL = B + "/PHerc0800/segments/{}/surface-volumes/8.64um-1.2m-116keV-volume-20250521135224.zarr"
TARGETS = {"0139": {"vol": W045_VOL, "voxel_um": 9.362, "ref": W045_REF}}
for i, seg in enumerate(P800_SEGS):
    TARGETS[f"0800-{i}"] = {"vol": P800_VOL.format(seg), "voxel_um": 8.64, "ref": None}
STRENGTHS = (0.5, 1.0, 2.0, 4.0)  # strength 0 is the reserved empty slots
NULL_SHARE = 0.4
LETTERS = "ΑΕΗΙΚΛΜΝΟΠΡΣΤΥΩ"
LETTER_MM = 2.5  # cap height of a typical Herculaneum book hand
CROP = 2048  # signature window
CROP_0139 = 4096  # injection window on PHerc. 139: the signature window plus ink-free papyrus around it


def open_array(url):
    import fsspec

    root = zarr.open(fsspec.get_mapper(url), mode="r")
    return root if isinstance(root, zarr.Array) else root["0"]


def load_ref(url, shape):
    """Published 2.4 um ink map (downsampled JPEG), resized onto the 9 um grid."""
    path = "/tmp/" + os.path.basename(url)
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    im = Image.open(path).convert("L").resize((shape[1], shape[0]), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def box_mean(a, k):
    """Mean over a k x k box, same shape, via an integral image."""
    c = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    h, w = a.shape
    y0, x0 = np.clip(np.arange(h) - k // 2, 0, h), np.clip(np.arange(w) - k // 2, 0, w)
    y1, x1 = np.clip(y0 + k, 0, h), np.clip(x0 + k, 0, w)
    s = c[y1][:, x1] - c[y0][:, x1] - c[y1][:, x0] + c[y0][:, x0]
    return s / ((y1 - y0)[:, None] * (x1 - x0)[None, :])


def coarse_papyrus(arr, step=512, tile=128):
    """Papyrus coverage from one 128-px chunk every `step` px, upsampled to full size.
    Reads ~1/16 of the chunks instead of the whole volume."""
    from concurrent.futures import ThreadPoolExecutor

    depth, h, w = arr.shape
    ys, xs = list(range(0, h, step)), list(range(0, w, step))

    def frac(yx):
        y, x = yx
        return float((np.asarray(arr[depth // 2, y:y + tile, x:x + tile]) > 0).mean())

    with ThreadPoolExecutor(16) as ex:
        v = np.array(list(ex.map(frac, [(y, x) for y in ys for x in xs])), np.float32)
    coarse = (v.reshape(len(ys), len(xs)) > 0.5).astype(np.uint8) * 255
    return np.asarray(Image.fromarray(coarse).resize((w, h), Image.NEAREST)) > 127


def pick_crop(ref, papyrus, size):
    """Crop with the most published ink that is still mostly papyrus."""
    step = size // 4
    best, at = -1.0, (0, 0)
    h, w = ref.shape
    for y in range(0, max(1, h - size + 1), step):
        for x in range(0, max(1, w - size + 1), step):
            if papyrus[y:y + size, x:x + size].mean() < 0.85:
                continue
            s = ref[y:y + size, x:x + size].mean()
            if s > best:
                best, at = s, (y, x)
    return at


# ------------------------------------------------------------------ signature
def signature(args):
    arr = open_array(W045_VOL)
    depth, h, w = arr.shape
    ref = load_ref(W045_REF, (h, w))
    papyrus = coarse_papyrus(arr)
    y, x = pick_crop(ref, papyrus, CROP)
    vol = np.asarray(arr[:, y:y + CROP, x:x + CROP], dtype=np.float32)
    r = ref[y:y + CROP, x:x + CROP]
    pap = (vol > 0).all(0)
    inside = r[pap]
    hi, lo = np.quantile(inside, 0.95), np.quantile(inside, 0.50)
    ink, bg = pap & (r >= hi), pap & (r <= lo)
    d, s = [], []
    for z in range(depth):
        a, b = vol[z][ink], vol[z][bg]
        d.append(float((a.mean() - b.mean()) / b.std()))
        s.append(float(a.std() / b.std()))
    out = {"crop_yx": [int(y), int(x)], "crop": CROP, "depth": depth, "voxel_um": 9.362,
           "ink_px": int(ink.sum()), "bg_px": int(bg.sum()), "d": d, "contrast_ratio": s}
    json.dump(out, open(args.out, "w"), indent=1)
    peak = int(np.argmax(np.abs(d)))
    print(f"signature from {ink.sum()} ink and {bg.sum()} background pixels; "
          f"peak d = {d[peak]:+.3f} sd at layer {peak} of {depth}")
    print("d per layer:", " ".join(f"{v:+.2f}" for v in d))


# ---------------------------------------------------------------- injection
def font(px):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, px)
    return None


def letter_mask(ch, cap_px):
    """Anti-aliased glyph mask in [0, 1], cropped to its ink."""
    f = font(int(cap_px * 1.4))
    size = int(cap_px * 2)
    im = Image.new("L", (size, size), 0)
    dr = ImageDraw.Draw(im)
    if f is not None:
        dr.text((size // 2, size // 2), ch, fill=255, font=f, anchor="mm")
    else:  # no font: a ring of comparable stroke area
        r, wd = cap_px // 2, max(2, cap_px // 8)
        dr.ellipse((size // 2 - r, size // 2 - r, size // 2 + r, size // 2 + r), outline=255, width=wd)
    a = np.asarray(im, dtype=np.float32) / 255.0
    ys, xs = np.nonzero(a > 0.02)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def make(args):
    sig = json.load(open(args.sig))
    t = TARGETS[args.target]
    arr = open_array(t["vol"])
    depth, h, w = arr.shape
    if args.target == "0139":
        cy, cx = sig["crop_yx"][0] + CROP // 2, sig["crop_yx"][1] + CROP // 2
        y = int(np.clip(cy - CROP_0139 // 2, 0, max(0, h - CROP_0139)))
        x = int(np.clip(cx - CROP_0139 // 2, 0, max(0, w - CROP_0139)))
        ch, cw = min(h, CROP_0139), min(w, CROP_0139)
        ref = load_ref(t["ref"], (h, w))[y:y + ch, x:x + cw]
    else:  # a whole PHerc. 800 segment
        y, x, ch, cw = 0, 0, h, w
        ref = None
    vol = np.asarray(arr[:, y:y + ch, x:x + cw])
    H, Wd = vol.shape[1:]
    pap = (vol > 0).all(0)
    # Signature in units of each target layer's background sd, centred on the middle layer
    d = np.zeros(depth, np.float32)
    src = np.asarray(sig["d"], np.float32)
    off = (depth - len(src)) // 2
    for i, v in enumerate(src):
        if 0 <= i + off < depth:
            d[i + off] = v
    sd = np.array([vol[z][pap].std() for z in range(depth)], np.float32)

    cap = int(round(LETTER_MM * 1000 / t["voxel_um"]))
    rng = np.random.default_rng(args.seed)
    cell = int(cap * 1.3)
    if ref is None:
        free = pap
    else:  # plant only where the published map is in its cleaner half, letter-sized box average
        bm = box_mean(ref, cap)
        free = pap & (bm < np.quantile(bm[pap], 0.5))
    slots = []
    for cy in range(0, H - cell, cell):
        for cx in range(0, Wd - cell, cell):
            if free[cy:cy + cell, cx:cx + cell].mean() > 0.9:
                slots.append((cy, cx))
    rng.shuffle(slots)
    n_null = int(round(len(slots) * NULL_SHARE))
    n = ((len(slots) - n_null) // len(STRENGTHS)) * len(STRENGTHS)
    if n == 0 or n_null == 0:
        raise SystemExit(f"too few free {cell}px slots in this crop ({len(slots)})")
    null_slots = [{"y": int(cy), "x": int(cx)} for cy, cx in slots[n:]]
    out = vol.astype(np.float32)
    letters = []
    for i, (cy, cx) in enumerate(slots[:n]):
        alpha = STRENGTHS[i % len(STRENGTHS)]
        ch = LETTERS[rng.integers(len(LETTERS))]
        m = letter_mask(ch, cap)
        oy, ox = cy + (cell - m.shape[0]) // 2, cx + (cell - m.shape[1]) // 2
        if alpha > 0:
            for z in range(depth):
                out[z, oy:oy + m.shape[0], ox:ox + m.shape[1]] += alpha * d[z] * sd[z] * m
        letters.append({"char": ch, "alpha": alpha, "y": int(oy), "x": int(ox),
                        "h": int(m.shape[0]), "w": int(m.shape[1])})
        np.save(os.path.join(args.out, f"mask_{i:03d}.npy"), (m > 0.5))
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    os.makedirs(args.out, exist_ok=True)
    z = zarr.open(os.path.join(args.out, "volume.zarr"), mode="w", shape=out.shape,
                  chunks=(depth, 128, 128), dtype="u1", compressor=None)
    z[:] = out
    meta = {"target": args.target, "source_url": t["vol"], "crop_yx": [int(y), int(x)], "shape": list(out.shape),
            "voxel_um": t["voxel_um"], "cap_px": cap, "cell_px": cell, "strengths": list(STRENGTHS),
            "letters": letters, "null_slots": null_slots,
            "papyrus_fraction": float(pap.mean()), "sd": sd.tolist(), "d": d.tolist()}
    json.dump(meta, open(os.path.join(args.out, "letters.json"), "w"), indent=1)
    if ref is not None:
        np.save(os.path.join(args.out, "ref.npy"), ref.astype(np.float16))
    np.save(os.path.join(args.out, "papyrus.npy"), pap)
    per = {a: sum(1 for L in letters if L["alpha"] == a) for a in STRENGTHS}
    print(f"{args.target}: {len(letters)} letters of {cap}px cap height, per strength {per}, "
          f"{len(null_slots)} empty control slots; "
          f"crop {out.shape} at {y},{x}; papyrus {pap.mean():.2f}")


# ------------------------------------------------------------------- scoring
def contrast(pred, m, y, x, ring):
    """Mean prediction on the strokes minus the mean in a surrounding ring."""
    h, w = m.shape
    H, W = pred.shape
    y0, x0, y1, x1 = max(0, y - ring), max(0, x - ring), min(H, y + h + ring), min(W, x + w + ring)
    win = pred[y0:y1, x0:x1]
    strokes = np.zeros(win.shape, bool)
    strokes[y - y0:y - y0 + h, x - x0:x - x0 + w] = m
    near = np.zeros(win.shape, bool)
    near[y - y0:y - y0 + h, x - x0:x - x0 + w] = True
    return float(win[strokes].mean() - win[~near].mean())


def score(args):
    meta = json.load(open(os.path.join(args.dir, "letters.json")))
    preds = [np.load(p).astype(np.float32) / 255.0 for p in args.pred]
    shape = tuple(meta["shape"][1:])
    preds = [p[:shape[0], :shape[1]] for p in preds]
    pap = np.load(os.path.join(args.dir, "papyrus.npy"))
    ring = meta["cap_px"] // 3
    names = [os.path.basename(p) for p in args.pred]
    combos = {n: [p] for n, p in zip(names, preds)}
    combos["mean of all"] = preds
    results = {}
    for cname, ps in combos.items():
        pred = np.mean(ps, axis=0)
        cs = {}
        for i, L in enumerate(meta["letters"]):
            m = np.load(os.path.join(args.dir, f"mask_{i:03d}.npy"))
            cs.setdefault(L["alpha"], []).append(contrast(pred, m, L["y"], L["x"], ring))
        # Null (strength 0): every planted glyph's mask, centred in each reserved empty slot
        null = []
        cell = meta["cell_px"]
        for slot in meta["null_slots"]:
            for i in range(len(meta["letters"])):
                m = np.load(os.path.join(args.dir, f"mask_{i:03d}.npy"))
                yy = slot["y"] + (cell - m.shape[0]) // 2
                xx = slot["x"] + (cell - m.shape[1]) // 2
                if yy - ring < 0 or xx - ring < 0 or yy + m.shape[0] + ring > shape[0] or xx + m.shape[1] + ring > shape[1]:
                    continue
                null.append(contrast(pred, m, yy, xx, ring))
        thr = float(np.quantile(null, 0.95)) if len(null) >= 20 else float("nan")
        rec = {0.0: float(np.mean(np.asarray(null) > thr))}  # 5% by construction
        rec.update({a: float(np.mean(np.asarray(v) > thr)) for a, v in sorted(cs.items())})
        med = {0.0: float(np.median(null))}
        med.update({a: float(np.median(v)) for a, v in sorted(cs.items())})
        results[cname] = {"threshold_95": thr, "null_n": len(null), "recovered": rec, "median_contrast": med,
                          "null": [float(v) for v in null], "letters": {str(a): v for a, v in cs.items()}}
        if "ref.npy" in os.listdir(args.dir):
            busy = np.zeros(pred.shape, bool)  # planted letters, excluded from the comparison with real ink
            for L in meta["letters"]:
                busy[max(0, L["y"] - ring):L["y"] + L["h"] + ring, max(0, L["x"] - ring):L["x"] + L["w"] + ring] = True
            ref = np.load(os.path.join(args.dir, "ref.npy")).astype(np.float32)
            ok = pap & ~busy
            results[cname]["r_vs_published_map"] = float(np.corrcoef(pred[ok], ref[ok])[0, 1])
            hi = ok & (ref >= np.quantile(ref[ok], 0.95))
            lo = ok & (ref <= np.quantile(ref[ok], 0.50))
            results[cname]["real_ink_contrast"] = float(pred[hi].mean() - pred[lo].mean())
    json.dump(results, open(os.path.join(args.dir, "scores.json"), "w"), indent=1)
    print(f"== {meta['target']}: {len(meta['letters'])} letters, cap {meta['cap_px']} px")
    for cname, r in results.items():
        line = "  ".join(f"a={a:g}: {v:.2f}" for a, v in r["recovered"].items())
        extra = ""
        if "r_vs_published_map" in r:
            extra = f"  | r vs published {r['r_vs_published_map']:.3f}, real-ink contrast {r['real_ink_contrast']:+.3f}"
        print(f"  {cname:28s} recovered  {line}  (null n={r['null_n']}){extra}")
        print(f"  {'':28s} median contrast " + "  ".join(f"a={a:g}: {v:+.3f}" for a, v in r["median_contrast"].items()))


def aggregate(args):
    """Pool letters across directories; each letter is judged against its own directory's null."""
    pooled = {}
    for d in args.dirs:
        res = json.load(open(os.path.join(d, "scores.json")))
        for cname, r in res.items():
            null = np.asarray(r["null"])
            if len(null) < 20:
                continue
            thr, mu, sd = np.quantile(null, 0.95), null.mean(), null.std() + 1e-9
            p = pooled.setdefault(cname, {"n_null": 0, "fp": 0, "alpha": {}})
            p["n_null"] += len(null)
            p["fp"] += int((null > thr).sum())
            for a, v in r["letters"].items():
                q = p["alpha"].setdefault(float(a), {"n": 0, "hit": 0, "z": []})
                v = np.asarray(v)
                q["n"] += len(v)
                q["hit"] += int((v > thr).sum())
                q["z"] += ((v - mu) / sd).tolist()
    out = {}
    print(f"== {args.label}: {len(args.dirs)} volumes")
    for cname, p in pooled.items():
        row = {"false_positive_rate": p["fp"] / max(1, p["n_null"])}
        parts = [f"null {row['false_positive_rate']:.2f}"]
        for a in sorted(p["alpha"]):
            q = p["alpha"][a]
            row[str(a)] = {"recovered": q["hit"] / q["n"], "n": q["n"], "median_z": float(np.median(q["z"]))}
            parts.append(f"a={a:g}: {q['hit']}/{q['n']} (z {np.median(q['z']):+.1f})")
        out[cname] = row
        print(f"  {cname:22s} " + "   ".join(parts))
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("signature")
    s.add_argument("--out", required=True)
    m = sub.add_parser("make")
    m.add_argument("--sig", required=True)
    m.add_argument("--target", choices=sorted(TARGETS), required=True,
                   help="0139 (calibration, known ink) or 0800-0 ... 0800-5 (eligible scroll segments)")
    m.add_argument("--out", required=True)
    m.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("score")
    c.add_argument("--dir", required=True)
    c.add_argument("--pred", nargs="+", required=True)
    g = sub.add_parser("aggregate")
    g.add_argument("--dirs", nargs="+", required=True)
    g.add_argument("--label", required=True)
    g.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "make":
        os.makedirs(a.out, exist_ok=True)
    {"signature": signature, "make": make, "score": score, "aggregate": aggregate}[a.cmd](a)

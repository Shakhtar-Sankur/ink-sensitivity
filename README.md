# ink-sensitivity

**Injection-recovery tests for Herculaneum ink detection: how strong does ink have to be before today's
models can see it on an unread scroll?**

The [Vesuvius Challenge](https://scrollprize.org) offers $50,000 for the first letters found on each of 21 eligible
Herculaneum scrolls. A community survey of those scrolls, run with the official 9 µm ink models, covered 971 cm² and found
no letters ([first-letters-survey](https://github.com/Bullo27/first-letters-survey)). Its own README explains why that
null is hard to read: the models were trained on other scrolls, and it is "not yet known whether they work on the
eligible ones". The challenge's [2026 open problems](https://scrollprize.org/2026_open_problems) ask for exactly this:
diagnostics that "reliably tell 'no ink' apart from 'no ink recovered yet'".

Astronomers answer the same question with injection-recovery tests: plant a known signal of measured strength into real
data, run the real pipeline, and count how often it comes back. This repository does that for scroll ink.

## Method

1. **Measure the ink signature** on a scroll with known text. PHerc. 139 segment w045 is published at 9.362 µm, the
   scan setting of the unread scrolls, and at 2.4 µm, where its letters are visible. Using the published 2.4 µm ink map
   as the label, compare voxels under ink with bare papyrus, layer by layer, in units of the papyrus's own spread.
2. **Plant synthetic Greek capitals** (2.5 mm cap height, a typical Herculaneum book hand) into real CT surface volumes,
   at 0.5×, 1×, 2× and 4× the measured signature, scaled by each target layer's own spread. 40% of the free positions are
   left empty as controls.
3. **Run the official model** ([`scrollprize/ink_9um`](https://huggingface.co/scrollprize/ink_9um), two checkpoints, both
   depth directions) on the injected volumes, unchanged.
4. **Score every letter**: mean prediction on its strokes minus a surrounding ring. A letter counts as recovered when it
   beats the 95th percentile of the same score on the empty controls, so the false-positive rate is 5% by construction.
   Results are pooled over segments and independent placements.

Targets: **PHerc. 139** itself, next to its real text (calibration), and **PHerc. 800**, which is eligible for the First
Letters prizes and the 2027 Grand Prize and has five published surface segments.

## Results so far

**Ink signature, PHerc. 139 w045 at 9.362 µm**
([data](results/signature_PHerc0139_w045_9.362um.json)): from 212,496 ink and 2,192,751 papyrus pixels, ink is brighter
by up to **+0.23 standard deviations**, a smooth profile peaking at layers 8–13 of 28. That is far below what an eye can
see in a single slice.

**Stage 0, baseline** ([output](results/stage0_kaggle_2026-10-28.txt), Kaggle Tesla T4, 28 Oct 2026): on the held-out
segment w045 at 9.362 µm, the official model's map correlates with the published 2.4 µm map at **r = 0.368**, against
**0.068** for the same map shifted (the null). It sees real ink at the unread scrolls' resolution. This r is a lower bound:
the reference is a downsampled JPEG, the two resolutions come from separate meshes, and both depth directions are averaged.

**Stage 1, injection-recovery**: running. Recovery fractions per strength for PHerc. 139 and PHerc. 800 will be added
here with the raw outputs.

## Reproduce

Everything streams from the [open data](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/index.html);
nothing is downloaded whole.

- `notebooks/stage0_baseline.ipynb` and `notebooks/stage1_injection.ipynb` run on a free Kaggle GPU (T4, Internet on).
  They install the official inference code from
  [ScrollPrize/villa](https://github.com/ScrollPrize/villa) (branch `merge-ink-pipelines`) with `uv`.
- `inject.py` holds the method and runs anywhere with NumPy, zarr 2, fsspec and Pillow:

```bash
pip install -r requirements.txt
python inject.py signature --out sig.json
python inject.py make --sig sig.json --target 0800-3 --out inj/0800-3 --seed 0
# ... run the ink model on inj/0800-3/volume.zarr, save its maps as inj/0800-3/ck*.npy ...
python inject.py score --dir inj/0800-3 --pred inj/0800-3/ck*.npy
python inject.py aggregate --dirs inj/0800-* --label "PHerc. 800" --out agg.json
python -m pytest -q tests        # offline tests on synthetic data
```

## Limits

- The injected signal is the measured mean brightness profile of ink. Real ink also changes texture. Stage 1 tests the
  injection itself on PHerc. 139: letters planted at 1× should score like the real text next to them.
- The signature is measured on one segment of one scroll. Ink chemistry may differ between scrolls, so a recovery curve
  says what the model *could* see on a scroll, not what ink that scroll carries.
- The labels are the published 2.4 µm predictions, registered by scaling, not hand-traced letters.

## Credits

Data: [Vesuvius Challenge](https://scrollprize.org) open data. Model and inference code:
[ScrollPrize/villa](https://github.com/ScrollPrize/villa) and [`scrollprize/ink_9um`](https://huggingface.co/scrollprize/ink_9um)
(MIT). Survey that motivated this work: [first-letters-survey](https://github.com/Bullo27/first-letters-survey) by
Matteo Bulloni. This repository contains code and derived measurements only, no scan data.

MIT licence. Copyright 2026 Sankur Kundu.

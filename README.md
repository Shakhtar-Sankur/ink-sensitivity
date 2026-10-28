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

**Stage 1, brightness injection** ([raw scores and output](results/stage1_kaggle_2026-10-28/), Kaggle Tesla T4,
28 Oct 2026, 18 volumes, 47.7 min of inference): **a brightness profile alone is not ink to the model.** On PHerc. 139,
the model's map rises by **0.11** on the real text, but by only **0.01–0.05** on planted letters, even at **4×** the
measured brightness. Pooled over all four model runs, at most 4 of 6 letters on PHerc. 139 and 9 of 18 on PHerc. 800
cleared the threshold at 4×. The ink the model detects must carry more than a mean shift: its texture across layers
and within strokes. Two caveats from the run itself: Kaggle's container had no font with Greek capitals, so every
"letter" was drawn as a ring of comparable stroke area (now an error, not a fallback), and each volume had only 5–8
empty control slots, so the 95% threshold is the maximum of a few values and the false-positive rate reads 0.

**Stage 2, real ink transplanted** (`notebooks/stage2_transplant.ipynb`, `inject.py make --mode transplant`): since
synthetic brightness is not ink, move real ink. From 2.2 mm windows of PHerc. 139's text, the voxel columns under the
labelled ink are written into the target's papyrus through the ink's own stroke shapes, matched layer by layer to the
target's brightness and spread. The control writes bare PHerc. 139 papyrus through the same shapes, so ink and control
differ only in the ink; a fifth of the windows stay untouched. On PHerc. 139 the same text is also scored in place.
Checked before inference: transplanted strokes are **+0.14 to +0.23 sd** brighter than their window, bare controls about
0, which is the signature carried across intact. Scores: recovery above the bare controls' 95th percentile and the AUC
of ink against bare, with confidence intervals that resample whole volumes.

## Reproduce

Everything streams from the [open data](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/index.html);
nothing is downloaded whole.

- `notebooks/stage0_baseline.ipynb`, `stage1_injection.ipynb` and `stage2_transplant.ipynb` run on a free Kaggle GPU
  (T4, Internet on).
  They install the official inference code from
  [ScrollPrize/villa](https://github.com/ScrollPrize/villa) (branch `merge-ink-pipelines`) with `uv`.
- `inject.py` holds the method and runs anywhere with NumPy, zarr 2, fsspec and Pillow:

```bash
pip install -r requirements.txt
python inject.py signature --out sig.json
python inject.py make --sig sig.json --target 0800-3 --out inj/0800-3 --seed 0 --mode transplant
# ... run the ink model on inj/0800-3/volume.zarr, save its maps as inj/0800-3/ck*.npy ...
python inject.py score --dir inj/0800-3 --pred inj/0800-3/ck*.npy
python inject.py aggregate --dirs inj/0800-* --label "PHerc. 800" --out agg.json
python -m pytest -q tests        # offline tests on synthetic data
```

## Limits

- Stage 1's injected signal is the measured mean brightness profile of ink, and it failed its own calibration: letters
  planted at 1× on PHerc. 139 score far below the real text next to them. Stage 2 transplants real ink instead.
- Stage 2's ink comes from one scroll, through 16 windows of its text (7 bare windows for the control), reused across
  placements. It asks whether PHerc. 139's ink would be seen on another scroll's papyrus, not what ink that scroll has.
- The signature is measured on one segment of one scroll. Ink chemistry may differ between scrolls, so a recovery curve
  says what the model *could* see on a scroll, not what ink that scroll carries.
- The labels are the published 2.4 µm predictions, registered by scaling, not hand-traced letters.

## Credits

Data: [Vesuvius Challenge](https://scrollprize.org) open data. Model and inference code:
[ScrollPrize/villa](https://github.com/ScrollPrize/villa) and [`scrollprize/ink_9um`](https://huggingface.co/scrollprize/ink_9um)
(MIT). Survey that motivated this work: [first-letters-survey](https://github.com/Bullo27/first-letters-survey) by
Matteo Bulloni. This repository contains code and derived measurements only, no scan data.

MIT licence. Copyright 2026 Sankur Kundu.

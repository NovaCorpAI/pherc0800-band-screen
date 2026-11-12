# Screening an unscreened band of PHerc. 0800 with the PHerc. 306B lane for about $2 of GPU (negative result + reusable tooling)

Anael Casanova (GitHub: NovaCorpAI), November 2026.

## Summary

We ran the public First Letters lane of the PHerc. 306B winners (Warner and Bodill, MIT) together with parts of the
PHerc. 343 recipe (Nieuwlaar, MIT) on a band of PHerc. 0800 that, as far as we can tell, nobody had screened:
z 5700 to 7200 (8.64 um scan). The result: no letters. We report it, with the numbers, so the next person can
skip this band or screen it differently.

The second deliverable is the tooling. The only step that needs a 24 GB GPU (the spiral fit and the snap) runs on a
rented RTX 3090 for **$2.01**. Everything after that (flatten, 31-layer render, Reader V3 ink at two depths on both
faces, window scoring, glyph CNN, row ranking, line finder, review cards) runs on **free Kaggle kernels** and is
validated end to end on 40 windings. Anyone with a Kaggle account can screen a band of any scroll with published
tracks this way.

The third is a **calibration of the 306B triage for Reader V3 maps** against known text on three scrolls and our 506
negative windows: on V3 the glyph threshold must rise from 9.06 to 11, which cuts the review queue tenfold for one
lost text window (section below).

## What was screened

| Item | Value |
|---|---|
| Scroll / volume | PHerc. 0800, 20250521135224-8.640um-1.2m-116keV-masked |
| Band | z 5700 to 7200 (SQM quality 0.36; tracks are published only for z 4500 to 17500) |
| Windings | 40 (w050 to w089), about 11 cm2 each, about 440 cm2 in total |
| Fit | villa 0fb38c45 spiral fitter, 30 000 steps, both senses (CW and ACW) |
| Snap to m7 surface | 99.96 % of vertices snapped, 21.4 % rejected, median offset 2.17 voxels |
| Ink | Reader V3, depth 0 and +3, both faces of every winding |
| Triage | 306B window_score, tri_rank, glyph_win (gA.pt), line_find; 33 review cards |

## Results

- tri_rank: 50 REVIEW, 107 REVIEW-2, 349 auto-rejected windows.
- Row score above the 306B text threshold (4) in three windows: w082 rev x2600 (7.26), w085 fwd x7800 (4.21),
  w063 rev x5200 (4.16).
- line_find best chains: w068 fwd 8.94, w075 rev 7.89, w089 rev 7.85, w087 rev 7.63, w074 rev 7.56. For reference, the
  306B calibration gives 8.2 to 11.2 on known text and up to 6.8 on nulls.
- Visual review of the top cards (CT middle layer, V3 map, 306B overlay; 3 best by rows, best by glyph CNN,
  2 best line_find chains): round 2 to 5 mm blobs, no strokes, no rows of letter-like shapes. Several of the
  high-scoring blobs sit on cracks or dark gaps in the CT.
- Verdict under the cut criterion we set before screening (at least one window with 2 or more rows of 5 or more
  letter-like shapes, or a line_find chain with high glyph probability): **negative**.

## Findings that may be useful beyond this band

1. **Spiral sense could not be decided.** CW and ACW fits are practically identical on every metric we tried, at
   4000 steps and at 30 000 steps: final loss 268 vs 262, snapped 99.96 % both, rejected 21.4 % both, median
   offset 2.17 vs 2.20, on-sheet 63 % both. Three independent sense tests (seam test, track slope, m7
   sheet-following) did not discriminate either. On this band the snap is insensitive to the sense, so the sense
   should not be inferred from snap quality.
2. **The 306B glyph CNN threshold does not transfer to Reader V3 maps.** glyph_win scores 9.3 to 13.9 in every one
   of the 40 windings, above its 9.06 threshold, including windings with nothing on them. Before using gA.pt as a
   gate on V3 maps it needs recalibration on V3 output of known text and of blank papyrus. Done below: with V3
   the threshold that keeps false alarms near 1 % is 11, not 9.06.
3. **The row score is fooled by blobs along cracks, but a crack veto is not the fix.** The best row window (w082,
   7.26) is a set of blobs aligned along a crack. We first suggested a hard crack veto; the calibration below
   refutes it: the crack score separates nothing on V3 maps (AUC 0.53) and known-text windows of PHerc. 0841 reach
   crack 1.29 to 1.58, so a hard veto would drop real text. The glyph CNN with a V3 threshold is the useful filter.
4. **Earlier negative hunts (same month, reported for completeness):** Reader V3 over 1 385 cm2 of public meshes
   (single depth, no sheet screening) and a depth-swept V3 hunt over 117 cm2, both negative. On positive controls
   (PHerc. 0841 and PHerc. 1447) a V3 x v8-in product reached AUC 0.877 [0.859 to 0.894] on 1447 w058 against
   0.850 for V3 alone, with overlapping intervals; a wide depth sweep (-15 to +15) made V3 worse on all five
   controls, while +3 layers helped a little on 4 of 5. That is why this band used depths 0 and +3 only.

## Calibration of the 306B triage for Reader V3 maps

The 306B triage (row score, glyph CNN gA, gates, line_find) was calibrated on the 306B team's own fused maps. Most
hunters run it on Reader V3 maps. We measured it on V3 maps with known text against our 40 visually negative
PHerc. 0800 windings, all through the same worker and triage (kernel `kaggle/calib`, public run
[band0800-calib](https://www.kaggle.com/code/anaelcasanova/band0800-calib), 2 h 53 min on free Kaggle GPUs).

- **Text (17 windows, 6 surfaces, 3 scrolls):** PHerc. 1447 w058 and w060 (8.64 um, labels from
  YoussefMoNader/ink-8um-pherc1447-surfaces), PHerc. 0841 w00, ag144 and ag174 (9.37 um, organisers' labels), and
  the PHerc. 0306B claim surface (the render Warner and Bodill published; positive windows are those over the claimed
  line). A window counts as text when at least 2 % of its labelled pixels are ink (5 % of the line box for 0306B), on
  the face where the V3 map matches the labels best.
- **Nulls (506 windows):** every window of the 40 PHerc. 0800 windings.

| feature on V3 maps | AUC text vs nulls | null p99 |
|---|---|---|
| glyph (gA) | **0.853** | 12.83 |
| row direction (row_off) | 0.767 | |
| row score | 0.716 | 7.42 |
| shape | 0.715 | |
| fibre direction (fib_off) | 0.672 | |
| crack | 0.526 | |
| papyrus texture (rel_tex) | 0.504 | |

| rule | text kept | null windows to review |
|---|---|---|
| shipped admission (row score >= 4 or glyph >= 9.06) | 16 of 17 | 310 of 506 (61 %) |
| shipped REVIEW queue | 10 of 17 | 50 of 506 (9.9 %) |
| **REVIEW queue and glyph >= 11 (recommended for V3)** | **9 of 17** | **5 of 506 (1.0 %)** |
| REVIEW queue and glyph >= 11.5 | 7 of 17 | 1 of 506 |

What this means for anyone running the 306B lane on Reader V3 maps:

1. **Raise the glyph threshold from 9.06 to 11 and use it inside the REVIEW queue.** It cuts the windows to look at
   tenfold (50 to 5 here) for one lost text window out of 10 that the queue had kept.
2. **The hard row-direction gate drops real text on V3 maps:** 5 of the 17 text windows (4 of 8 on PHerc. 1447) are
   auto-rejected for "rows not along winding" (row_off 25 to 85 degrees). It also removes most nulls, so we keep it,
   but a lane that cares about recall should demote it to a soft gate and rank by glyph.
3. **Crack and papyrus-texture gates carry no signal on V3 maps** (AUC 0.53 and 0.50); keep them soft.
4. **The faint case is the hard case.** On the PHerc. 0306B claim surface V3 alone does not show the line at pixel level
   (AUC 0.52 against the line box), yet the window over the line has a row score of 13.98, higher than every
   null window but one (14.22); the shape gate demotes it to REVIEW-2 as "round blobs". Only 1 of its 4 line windows passes the
   recommended rule. Reader V3 is not enough for First Letters-grade faint text; 306B needed their own fine-tuned
   models for it.
5. **line_find does not separate on V3 maps:** on PHerc. 1447 the best chains on text score 9.05 and 6.49 while the
   PHerc. 0800 nulls reach 8.94 (p99 7.50). (line_find skipped the 0841 and 0306B surfaces because their names do not
   follow the `wNNN` winding pattern; to be fixed.)

The 5 null windows the recommended rule keeps were checked by eye: 3 had review cards (cards 06, 11 and 19 of the
[band0800-cards](https://www.kaggle.com/code/anaelcasanova/band0800-cards) run: w064 rev x1300, w078 rev x7800, w075 rev
x7800) and show the same blobs on gaps; w067 rev x3900 and x5200 have no card yet.

Limits: 17 text windows on 3 scrolls is a small sample, the 1447 and 0841 text is denser than a First Letters target,
recur (needs neighbouring windings) was not available for the text surfaces, and the 0306B labels are a box around the
line, not ink tracings. Per-window features and labels are in `results/calib/windows.csv`.

## Tooling (what is reusable)

Layout of this repository (code MIT, see Licenses):

- **`pod/`, the pod package** (RunPod, any 24 GB GPU): `pod_setup.sh`, `pod_data.sh`, `pod_lane.sh` (stages 00_disk to
  06_pack), `pod_acw.sh` (second sense), `native/` (the vc spiral kernels built standalone with CMake and gcc 13),
  `spiral_old/` (fitter at villa 0fb38c45), `build_umb.py` / `smooth_umb.py` (umbilicus), `snap_meshes.py`,
  `patch_fitter.py`, sense tests (`sense_geo.py`, `sense_tracks.py`, `sense_m7.py`, `seam_test.py`). One file,
  `lane.env`, picks scroll, band, voxel size, sense and number of windings.
  Measured: setup + umbilicus + data 30 min; 30k-step fit 4 h 19 min at 0.41 s/step, 16 GB VRAM; snap 22 s.
- **`kaggle/`, the Kaggle chain** (free): `kaggle/flatten` (40 windings in 42 min on a T4), `kaggle/render-0..3` (31-layer
  surface volumes streamed by chunks from the open-data bucket, 4 CPU kernels in parallel, about 3 h each),
  `kaggle/ink` (Reader V3 at two depths and both faces, then the 306B triage, 6 h 52 min on GPU),
  `kaggle/cards` (review cards). `kaggle/code-ds` is the source of the `band0800-code` dataset the kernels read. Kernels chain through `kernel_sources`, so no large file ever leaves Kaggle.
- **Cost:** $2.01 of RunPod in total (8 h 36 min of an RTX 3090 Community at $0.22/h), $0 on Kaggle.

- **`results/`**: `fit/` (snap reports for both senses, timing, umbilicus, winding list and the 40 snapped CW
  windings as tifxyz in `fit/snap/`), `triage/` (tri_rank, line_find, run log of the ink kernel) and `cards/` (the six
  review cards discussed above plus the index of all 33).

## Kaggle (public runs)

The runs behind every number above are public, with logs and outputs: kernels
[flatten](https://www.kaggle.com/code/anaelcasanova/band0800-flatten),
[render-0](https://www.kaggle.com/code/anaelcasanova/band0800-render-0),
[render-1](https://www.kaggle.com/code/anaelcasanova/band0800-render-1),
[render-2](https://www.kaggle.com/code/anaelcasanova/band0800-render-2),
[render-3](https://www.kaggle.com/code/anaelcasanova/band0800-render-3),
[ink](https://www.kaggle.com/code/anaelcasanova/band0800-ink) and
[cards](https://www.kaggle.com/code/anaelcasanova/band0800-cards); datasets
[band0800-code](https://www.kaggle.com/datasets/anaelcasanova/band0800-code),
[band0800-snap](https://www.kaggle.com/datasets/anaelcasanova/band0800-snap) and
[band0800-flat](https://www.kaggle.com/datasets/anaelcasanova/band0800-flat).
"Copy & Edit" on a kernel reruns it on your own account.

## How to reproduce

1. Edit `pod/lane.env` (scroll, band z0/z1, voxel size, sense, number of windings). Start a 24 GB pod with
   `runpod/pytorch:2.1.0-cu1281-torch280-ubuntu2404` and an 80 GB volume, copy `pod/` to it and run
   `pod_setup.sh`, `pod_data.sh`, then `pod_lane.sh` (it writes `results.tar`; `pod_acw.sh` fits the other sense).
2. Upload the `snap/` windings as a Kaggle dataset, and `kaggle/code-ds` as another one.
3. Push the kernels in order with `kaggle kernels push`: `flatten`, then `render-0..3` (in parallel), then `ink`,
   then `cards`. Each kernel's `kernel-metadata.json` lists its inputs; change the owner prefix to your user.
4. Read `tri_rank.txt` and `line_find.json` from the ink kernel and look at the cards.

## Licenses

Our code: MIT (`LICENSE`). Vendored code: villa (`pod/spiral_old`, `pod/native/src`, `pod/vesuvius`) MIT, see
`third_party/LICENSE-villa`; PHerc. 306B scripts in `kaggle/code-ds/fl` and `kaggle/code-ds/x_glyph` MIT, see
`third_party/LICENSE-306B`; ideas from the PHerc. 343 recipe, see `third_party/LICENSE-343`. The glyph CNN weights
`kaggle/code-ds/x_glyph/models/gA.pt` are CC BY-NC 4.0 (`third_party/LICENSE-306B-DATA`). Everything in `results/`
and `img/` derives from the Vesuvius Challenge open data and is CC BY-NC 4.0; please credit the Vesuvius Challenge.

## Credits

Lane, triage scripts and gA.pt: Liam Warner and Nicola Bodill (PHerc. 306B, MIT). Recipe ideas (spiral fit + snap +
lasagna flatten): Erwin Nieuwlaar (PHerc. 343, MIT). Spiral fitter, vc kernels, vesuvius package and data: the
ScrollPrize/villa contributors and the open-data release. Ink model: Reader V3 (MIT).

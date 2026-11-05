# Real-frame data

Two manifests; the images themselves are not in the repo.

| file | what | how to fetch |
|---|---|---|
| `clean_frames.csv` | 326 free-licence screenshots of 20 open-source games from Wikimedia Commons, with title, licence, author and source page per file; 4 titles marked `unseen` are held out of training | `python realdata/fetch.py realdata/clean_frames.csv data/real_frames` |
| `real_glitches.csv` | screenshots attached to public issues of graphics projects (emulators, translation layers, engines), labelled `glitch` / `clean` / `excluded` | `python realdata/fetch.py realdata/real_glitches.csv data/real_glitches` |

`build_manifest.py` rebuilds the Commons manifest; `find_glitch_reports.py` collects bug-report
candidates through the GitHub search API.

## How `real_glitches.csv` was labelled

**Pass 1 (Oct 30).** Candidate screenshots were collected from issues whose title or body
mentions rendering problems, in three batches (random per repository, then keyword-filtered,
then glitch-titled issues). Each image was viewed on numbered contact sheets and marked: a
visible rendering glitch (with best-fit Glitchify classes and a one-line description), a clean
in-game frame (often the "expected" screenshot in the same report), or not usable (UI, logs,
photos of monitors, unclear). Unusable images were not kept.

**Pass 2 (Nov 4–5).** Every row was re-checked at higher resolution (700 px per image, full
resolution where needed):

- all 117 pass-1 glitch rows, by the same reviewer as pass 1;
- all 382 pass-1 clean rows, each by one of three further reviewers, plus an independent
  60-image sample by the pass-1 reviewer to measure agreement. Every clean→glitch change was
  confirmed by the pass-1 reviewer before it was applied; a flagged row that reviewer could not
  confirm was excluded rather than kept as clean.

The `review` column records the outcome: `confirmed`, `relabelled` (label or classes changed),
or `excluded` (row kept for the record, label set to `excluded`, never scored). `review_note`
says why. `realdata/apply_review.py` applies a decisions file and is the only thing that
changes labels.

| pass 1 | pass 2 | rows |
|---|---|---|
| glitch | glitch (class labels unchanged) | 101 |
| glitch | glitch, classes corrected | 12 |
| glitch | clean (the "expected" render of the same scene) | 1 |
| glitch | excluded (2 phone photos of a screen, 1 not a fault) | 3 |
| clean | clean | 175 |
| clean | glitch (fault confirmed at higher resolution) | 87 |
| clean | excluded (flagged by a reviewer, not confirmable) | 120 |
| | **final: 200 glitch, 176 clean, 123 excluded** | 499 |

Pass 1 missed many subtle faults: the second pass found visible glitches in 87 of the 382
screenshots pass 1 had called clean. On the pass-1 reviewer's own 60-image re-check, the
second reviewers agreed on all 5 glitches but flagged 16 of the 52 images the pass-1 reviewer
still saw as clean, so the "clean" class is the noisier one. Excluding every disputed clean row
keeps the scored clean set conservative. The decision files are in `review_2026-11/`.

## Caveats

- Labels come from visual review of screenshots, not from the bug fixes. Subtle faults (a wrong
  shadow, a missing particle) are probably still labelled clean, so real-glitch recall is
  measured on visible faults.
- The class labels are best-fit mappings onto the ten procedural Glitchify types; many real
  faults (missing geometry, garbage textures) only fit loosely. The binary label is the
  reliable one.
- Emulators and translation layers dominate the sources, and screenshots often include UI
  overlays (FPS counters, emulator chrome).

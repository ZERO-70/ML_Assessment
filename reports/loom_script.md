# Loom recording plan — 3 minutes

**Do not read this aloud.** These are talking points, not a script. Reading from
a page is audible — the pace flattens and the pauses land in the wrong places.
Read the whole thing twice, then record while glancing only at the bold phrases.
The numbers are the only part worth getting exactly right.

Spotter asked for five things. Every one is covered below, in an order that
tells a story rather than working down their list:

1. Key findings from exploring the data → 0:15
2. Data-quality issues and how you addressed them → 0:45
3. Training and validation approach, including the split → 1:20
4. Reasoning behind the chosen model → 1:50
5. Brief walkthrough of the most important code → woven through, and 2:25

---

## Before you hit record

- Open these five tabs, **in this order**, so you never hunt for one on camera:
  1. `notebooks/01_exploratory_analysis.ipynb` — scrolled to the corruption chart (section 6)
  2. `notebooks/04_evaluation.ipynb` — scrolled to the fold table
  3. `notebooks/06_final_model.ipynb` — scrolled to the extrapolation chart (section 5)
  4. `src/model.py` — scrolled to `class HybridModel`
  5. `scorer_results/candidate_december.png`
- Zoom the browser to ~125%. Notebook text is small on replay.
- Close Slack, mail, anything that can pop a notification.
- Do one throwaway take to warm up. Nobody's first take is their best.

---

## 0:00 – 0:15 — Open on the constraint, not on yourself

**Screen:** notebook 01, top.

Skip "hi, my name is, I'm going to walk you through". Start with the thing that
shaped every decision:

> The training data stops on the 31st of October, and every load I have to
> predict comes after it. So this isn't really a regression problem, it's a
> forecasting problem — and that changes how you validate and which model you
> can use.

Saying this first tells them you understood the assessment rather than just
completed it.

---

## 0:15 – 0:45 — What the data showed

**Screen:** notebook 01, scroll slowly through the first two scatter plots, then
stop on the market-signals chart.

Three points, quickly:

- Price is **almost linear in distance** — freight is priced per mile. The band
  widens with distance, so error is proportional, which is why the model fits
  **log of the rate**, not raw dollars.
- **Weight barely matters** — correlation of 0.09. Worth saying out loud because
  it's the kind of thing people assume matters and never check.
- `market_index` is the strongest single driver, and it carries **a seasonal
  hump plus a clean seven-day cycle**.

Don't linger. This section exists to set up the next two.

---

## 0:45 – 1:20 — Data quality: lead with the corruption

**Screen:** notebook 01, the corruption chart (three separated blobs). Let it sit
on screen — this is your strongest single visual.

> About 1.4% of the labels are corrupted. Not noisy — corrupted.

Then the evidence, in this order:

- I compared each load to **its own peer group** — same equipment, similar
  distance — rather than to the whole dataset, because a short reefer and a long
  dry van aren't priced the same per mile.
- That gives **three separated populations**: 99% of loads within ±16% of their
  peer group, then **340 loads at about 3x** and **337 at about a third**.
  Near-symmetric, reciprocal multipliers, spread evenly across all ten months.
- The clincher: **the flagged rows look identical to everything else on every
  input feature.** Only the label differs. So it isn't premium freight or a real
  segment — it's injected noise.

Then the decision, which is the part that shows judgement:

> I drop them from training, where they'd teach a pattern that doesn't exist —
> but I keep them when scoring. The data you grade me on is presumably
> corrupted the same way, so removing them would just flatter my own number.

Mention the other three defects in one breath — **437 sign-flipped weights
recovered by magnitude, missing market index filled from the same day's other
loads, and eight cities that only appear in the validation set** — and say the
last one is why every geographic feature is built from coordinates rather than
city names.

---

## 1:20 – 1:50 — Validation and the split

**Screen:** switch to notebook 04, the fold table.

> A random train/test split would have scored much better and meant nothing.

- Three folds, **cut-off walking forward**, training window expanding, testing
  on the two months after each cut-off.
- **61-day test windows on purpose** — same horizon as November and December.
  On one-month windows the same model scores $49 instead of $62, so a shorter
  window would have flattered me.
- I **check for overlap directly** rather than trusting the date arithmetic —
  point at the leakage-guard output on screen.

If you only have time for one sentence here, make it the 61-day one. It's the
detail most candidates won't have thought about.

---

## 1:50 – 2:25 — The model, and the finding behind it

**Screen:** notebook 06, the extrapolation chart. This is the moment of the whole
video — slow down.

> Gradient boosting was the more accurate model on every fold. I didn't submit
> it.

Let that land, then explain:

- To test the trend properly I held one load fixed, **froze the market signals**,
  and moved only the date. That isolates what the model believes about time.
- Point at the chart: **boosting learns the trend inside the training window,
  then goes completely flat past October.** Ridge keeps climbing.
- Why: **a tree splits on thresholds it saw in training.** Any later date falls
  on the same side of every split as the last day it knows, so it gets the same
  answer. No hyperparameter fixes that.

**Screen:** switch to `src/model.py`, `class HybridModel`.

> So the model is two stages. Ridge fits the log rate including the time trend —
> that's the part that extrapolates. Then gradient boosting fits ridge's
> residual, which recovers the detail a straight line misses. The prediction is
> the two added together.

One honest aside, if the pace allows — it reads as genuine and it's true:

> I originally hid the date column from the second stage, assuming it would
> re-break extrapolation. I measured it, and I was wrong — stage one has already
> removed the trend, so there's nothing left for stage two to get wrong. Hiding
> it cost about seven dollars of error for nothing.

---

## 2:25 – 3:00 — The chart, and close

**Screen:** `scorer_results/candidate_december.png`.

- Fixed load, only the date changes, so **the chart is a pure readout of what
  the model thinks about time**.
- The market signals aren't in that file, but **validation.csv has about 200
  real loads on every one of those 31 December days** — so I averaged their
  market index per day. That's a lookup of recorded conditions, not a forecast.
- Point at the shape: **five weekly peaks from the market cycle, plus a 3.2%
  rise across the month from the trend.** Boosting alone would have given the
  peaks and no rise.

Close on the number and stop:

> $61.80 mean absolute error, 2.06% median error on clean labels, under
> rolling-origin validation. Everything's in the repo — the notebooks have their
> outputs saved, so you can read the whole chain of reasoning without running
> anything.

Don't add "thanks for watching, hope to hear from you". End on the work.

---

## Numbers worth memorising

Getting one wrong on camera is worse than not saying it.

| | |
|---|---|
| Corrupted labels | **1.4%** — 340 inflated ~3x, 337 deflated ~⅓ |
| Peer-group spread | 99% within **±16%** |
| Sign-flipped weights | **437** recovered |
| Unseen cities / lanes | **8** cities, **736** lanes |
| Trend the signals miss | **~7%** January to October |
| Folds | **3**, **61-day** test windows |
| Headline | **$61.80** MAE, **2.06%** median APE (clean labels) |
| Boosting vs hybrid past October | **−0.0%** vs **+2.0%** |
| December chart | **+3.2%** across the month |

## If you run long

Cut in this order — first to go first:

1. The `weight` correlation point (0.09)
2. The three minor data defects, keeping only the corruption
3. The honest aside about hiding the date column

Never cut: the corruption evidence, the 61-day window, or the extrapolation
chart. Those three are what separate this from a competent-but-ordinary
submission.

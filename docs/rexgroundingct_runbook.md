# ReXGroundingCT runbook

## Objective

Maximize the official ReXrank instance-level grounding score: text-conditioned
3D masks for every finding in a scan. The released scorer reports global Dice
and connected-component instance precision/recall/F1. The primary selection
metric is instance F1; global Dice and hit rate remain mandatory guardrails.

No clinical-use claim is made. This is a research challenge route.

## Data contract

- `../RexGroundingData` contains the gated `rajpurkarlab/ReXGroundingCT`
  snapshot: metadata, report findings, released masks, and the official scorer.
- Current challenge metadata has 2,992 train scans, 200 public validation scans,
  and 300 test scans. Test masks are intentionally absent.
- The mask tensor is `[F,H,W,D]`; each finding key selects one channel and each
  nonzero value is a labeled entity. Predictions must use the same filename and
  `[F,H,W,D]` shape.
- ReXGroundingCT does not contain the CT voxel volumes. The matching volumes are
  resolved by basename from `ibrahimhamamci/CT-RATE`, using its fixed snapshot
  hierarchy (`train_fixed` or `valid_fixed`). `train_1741_b_2.nii.gz` was
  verified as `(512,512,238)` int16 and shape-aligned with its released mask.
  The mask affine is identity while the CT affine carries physical spacing; the
  training path therefore uses the shared voxel index grid and does not silently
  resample either array.
- The challenge train set contains 7,687 finding queries; the public validation
  set contains 381. Split by scan, never by finding, to prevent volume leakage.

## Model route

`train_rexgrounding.py` uses arjun's `FPNDecoder3D` and `DiceBCELoss` around the
released CT-FM SegResEncoder (`project-lighter/ct_fm_feature_extractor`,
77,760,992 backbone parameters). A frozen PubMedBERT encoder supplies semantic
query embeddings. Text-dependent channel gates condition every CT-FM pyramid
level before the 3D FPN head. The model receives one normalized CT channel:
HU clip `[-1024, 2048]`, map to `[0,1]`.

The default input is a `96^3` patch. Positive patches are lesion-centered with
probability 0.75; the remaining patches are random negatives. No left/right,
apex/base, or anatomical-axis flip is used because it would corrupt the
radiology language-to-location relationship. Intensity gain/bias jitter is
small and deterministic per sample.

### Staged optimization

1. **Stage A:** freeze CT-FM and PubMedBERT; train text projection, text gates,
   and FPN decoder with AdamW (`1e-4`, weight decay `1e-2`). Use a scan-level
   internal development split carved deterministically from official train.
2. **Stage B:** unfreeze CT-FM with a lower LR (`1e-5`); retain the head LR.
   Select only if internal development instance F1 improves without degrading
   global Dice or small-lesion recall.
3. **Optional text adaptation:** unfreeze only after Stage B is stable. It is
   lower priority than visual adaptation because the released findings are
   already richly descriptive and the text encoder is a useful semantic prior.

For inference, use sliding windows with overlap. Tune threshold, stride,
component filtering, and any global-to-local refinement only on internal
training-derived development. The public validation set is a single final
comparison, not a tuning set. The private test set is never used for fitting.

### TotalSegmentator anatomical-prior fusion

TotalSegmentator is an offline preprocessing stage, not a training-time
subprocess. This avoids running a second 3D model once per patch and makes
priors cacheable, auditable, and reusable for train/validation/test inference.
The optional `totalsegmentator` extra installs the open `total` task; the
default prior channels are the five lung lobes, heart, aorta, trachea,
esophagus, and pulmonary vein. Licensed high-resolution subtasks are not used.

```bash
# Optional; pulls the nnUNet/TotalSegmentator inference stack.
UV_PROJECT_ENVIRONMENT=.venv-rex uv sync --frozen \
  --extra medical --extra hf --extra rexgrounding --extra cuda \
  --extra totalsegmentator --python /home/siddhesh/miniconda3/bin/python3.13

# Requires the matching CT-RATE fixed volumes to be staged locally.
.venv-rex/bin/python scripts/prepare_totalsegmentator_priors.py \
  --data-dir ../RexGroundingData \
  --volume-root ../RexGroundingData/ct_rate_fixed \
  --prior-dir ../RexGroundingData/totalsegmentator_priors \
  --splits train val test --device gpu
```

Each case is written under
`../RexGroundingData/totalsegmentator_priors/<volume-stem>/` as one binary
NIfTI per label, plus a manifest. The loader rejects missing labels, shape
changes, or affine changes; it never silently resamples an anatomical prior.
Use `--dry-run --max-cases 1` to verify the staged path and exact CLI command
without invoking TotalSegmentator.

The fusion model encodes the prior stack with a small strided 3D stem, projects
it into every CT-FM pyramid level, adds it to the CT features, and then applies
the existing text-dependent gates. Projection layers are zero-initialized so a
baseline CT-FM checkpoint has an exact safe starting point while the prior
path learns. Start Stage A from scratch or initialize from the unfused
checkpoint; the loader permits only the new prior keys to be absent and resets
the optimizer when parameter-group sizes change:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-rex/bin/python scripts/train_rexgrounding.py \
  --mode train --data-dir ../RexGroundingData \
  --volume-root ../RexGroundingData/ct_rate_fixed \
  --totalseg-prior-dir ../RexGroundingData/totalsegmentator_priors \
  --checkpoint artifacts/runs/rexgroundingct/stage_a/best.pt \
  --patch-shape 96 96 96 --patches-per-finding 2 \
  --positive-ratio 0.75 --batch-size 1 --gradient-accumulation 8 \
  --max-steps 4000 --epochs 2 --eval-every 100 --dev-batches 64 \
  --output-dir artifacts/runs/rexgroundingct/stage_a_totalseg --device cuda
```

Pass the same `--totalseg-prior-dir` and label list to `--mode predict`.
Compare the fused and unfused checkpoints on the scan-level internal
development split with the official evaluator before promoting either route.

### Report-selected lobe guidance

The report-conditioned route keeps the full CT input but selects only the
TotalSegmentator channels named by the current finding. The deterministic
resolver maps side/lobe phrases, lobe abbreviations, lingula, bilateral lung
phrases, and explicit lung fallback phrases to the five lobe channels. The
selected channels are confidence-scaled; unrelated anatomical channels are
zeroed before the prior stem sees them.

This is a soft anatomical hypothesis, not a lesion ground truth. Training adds
the normal Dice/BCE objective plus a small configurable outside-region penalty
(`--region-guidance-weight`, default `0.02`) only when a selected region is
present in the patch. Inference applies conservative connected-component
filtering only for resolver confidence at least `0.6`; components are retained
when they overlap a 15 mm physical halo around the selected lobe by at least
`0.02`. The model never hard-masks CT voxels, and unsupported phrases such as
isolated pleural thickening fall back to the unfiltered prediction.

Run the same prior directory and labels for training and prediction. Tune the
guidance weight, post-processing confidence, halo, and overlap only on the
internal scan-level development split. Add specialized
`pleural_pericard_effusion` inference later only if the anatomy coverage audit
shows that lobe guidance is insufficient.

## Commands

From `arjun/`:

```bash
UV_PROJECT_ENVIRONMENT=.venv-rex uv sync --frozen \
  --extra medical --extra hf --extra rexgrounding --extra cuda \
  --python /home/siddhesh/miniconda3/bin/python3.13

# Confirm the targeted list without downloading the 12-TB CT-RATE repository.
.venv-rex/bin/python scripts/download_rexgrounding_ct.py \
  --data-dir ../RexGroundingData --splits train val test --dry-run

# Stage only the basenames referenced by this challenge. This is resumable.
.venv-rex/bin/python scripts/download_rexgrounding_ct.py \
  --data-dir ../RexGroundingData --splits train val test --workers 16
```

A real-CT one-step smoke run already completed on the RTX 3090 using the same
model and patch geometry. It used 7.739 GiB peak allocated VRAM and produced
finite logits and a checkpoint. Reproduce it with:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-rex/bin/python scripts/train_rexgrounding.py \
  --mode train --data-dir ../RexGroundingData \
  --volume-root ../RexGroundingData/ct_rate_fixed \
  --allow-remote-download --max-cases-for-smoke 1 \
  --patches-per-finding 1 --positive-ratio 1.0 \
  --batch-size 1 --gradient-accumulation 1 --max-steps 1 --epochs 1 \
  --eval-every 0 --dev-batches 1 \
  --output-dir artifacts/runs/rexgroundingct/smoke --device cuda
```

Full Stage A:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-rex/bin/python scripts/train_rexgrounding.py \
  --mode train --data-dir ../RexGroundingData \
  --volume-root ../RexGroundingData/ct_rate_fixed \
  --patch-shape 96 96 96 --patches-per-finding 2 \
  --positive-ratio 0.75 --batch-size 1 --gradient-accumulation 8 \
  --max-steps 2000 --epochs 1 --eval-every 100 --dev-batches 64 \
  --output-dir artifacts/runs/rexgroundingct/stage_a --device cuda
```

Stage B starts from `stage_a/best.pt` and must be run as a separate artifact:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-rex/bin/python scripts/train_rexgrounding.py \
  --mode train --data-dir ../RexGroundingData \
  --volume-root ../RexGroundingData/ct_rate_fixed \
  --train-visual --checkpoint artifacts/runs/rexgroundingct/stage_a/best.pt \
  --patch-shape 96 96 96 --patches-per-finding 2 \
  --positive-ratio 0.75 --learning-rate 1e-4 --visual-learning-rate 1e-5 \
  --batch-size 1 --gradient-accumulation 8 --max-steps 4000 --epochs 2 \
  --eval-every 100 --dev-batches 64 \
  --output-dir artifacts/runs/rexgroundingct/stage_b --device cuda
```

Generate public-validation masks from the selected checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-rex/bin/python scripts/train_rexgrounding.py \
  --mode predict --data-dir ../RexGroundingData \
  --volume-root ../RexGroundingData/ct_rate_fixed \
  --split val --checkpoint artifacts/runs/rexgroundingct/stage_b/best.pt \
  --inference-stride 64 64 64 --inference-batch-size 1 \
  --threshold 0.5 --output-dir artifacts/runs/rexgroundingct/stage_b \
  --device cuda

.venv-rex/bin/python scripts/evaluate_rexgrounding.py \
  --data-dir ../RexGroundingData \
  --pred-dir artifacts/runs/rexgroundingct/stage_b/predictions/val \
  --split val \
  --output-json artifacts/runs/rexgroundingct/stage_b/official_val.json
```

The evaluator imports the released `rexrank_eval_fast.py` by path and records
its SHA-256. Defaults match the official scorer: global hit threshold `0.1`,
instance match Dice threshold `0.2`, minimum predicted component size `10`,
and 18-neighbor 3D connectivity.

## Promotion and ablations

Keep a manifest for every candidate: metadata hash, checkpoint hash, CT-FM and
text revisions, preprocessing, seed, split membership, threshold, stride,
component filter, and hardware peak memory. Promote only from internal
scan-level development results. Recommended order:

1. Stage A vs Stage B.
2. Threshold grid around 0.35–0.65 and minimum-component grid around 10–100.
3. Stride 48/64/80 with fixed checkpoint.
4. Hard-negative mining from high-scoring false-positive components.
5. Multi-window CT input or a global-low-resolution plus local-refinement route,
   only if the internal development improvement survives all guardrails.
6. An equal-weight ensemble of genuinely complementary retained checkpoints,
   with averaging and post-processing frozen before public validation.

Do not use public validation answers, private test masks, or leaderboard feedback
for any parameter selection. Store test predictions only after the route is
frozen.

## Prerequisites and risks

- The CT-RATE account gate must be accepted for the same Hugging Face token used
  by the downloader. The ReXGroundingCT token alone does not authorize CT-RATE
  voxel files.
- The targeted fixed CT set is approximately 369 GiB for train/validation/test
  basenames. The current filesystem has over 1 TiB free, so full staging is
  feasible; the downloader is resumable and writes a manifest. Do not snapshot
  the entire CT-RATE repository (about 12 TB in the API inventory).
- Keep raw CT and masks outside git. Checkpoints and run artifacts stay under
  ignored `artifacts/`.
- A single smoke loss or patch Dice is not a leaderboard claim. The official
  ReXrank evaluator on complete public validation predictions is the required
  quality evidence.

Validate mask metadata and every staged train/validation CT header before a
training run:

```bash
.venv-rex/bin/python scripts/validate_rexgrounding.py \
  --data-dir ../RexGroundingData --splits train val --require-volumes
```

After the resumable downloader reports all 3,492 files, repeat with
`--splits train val test --require-volumes`.

# Task: Improve Mahalanobis AUROC for failure prediction

## Background

We use Mahalanobis distance from VLM embeddings as an OOD detector to predict
episode failures in a robotic imitation learning system. The current AUROC is
**0.7579** on 50 episodes (36 success, 14 failure) from
`reece-omahoney/libero-10`. The goal is to improve this.

### How the pipeline works

1. **Fit stats** (`distal/compute_maha_stats.py`): A base policy is used to
   embed all frames of a success-only dataset (`lerobot/libero`) via
   `embed_prefix_pooled()`. A Gaussian is fit (mean + covariance inverse) and
   saved to `reece-omahoney/maha-stats` on HuggingFace Hub.

1. **Evaluate AUROC** (`distal/maha_auroc.py`): The same policy embeds frames
   from a separate rollout dataset (`reece-omahoney/libero-10`) which contains
   both successes and failures. Per-frame Mahalanobis distances are computed,
   aggregated per episode (currently max), and AUROC is computed with failure as
   the positive class.

The stats are fit on a **different dataset** (`lerobot/libero`, all successes)
than the evaluation dataset (`reece-omahoney/libero-10`, mixed success/failure).
This is correct -- the Gaussian represents "normal" behavior from
demonstrations, and we're testing whether OOD distance from that distribution
predicts failures during rollouts.

### Current baseline results

```
Episodes: 50 (36 success, 14 failure)
Max Maha dist -- success: 39.60 +/- 1.52
Max Maha dist -- failure: 41.62 +/- 2.15
AUROC (max Maha -> failure): 0.7579
```

## Goal

Improve the AUROC. You may modify `distal/compute_maha_stats.py` and/or
`distal/maha_auroc.py`. Read the existing code, understand the pipeline, and
decide what to try. Report AUROC after each change so we can track what helped.

## How to run

- Fit stats: `uv run python -m distal.compute_maha_stats`
- Evaluate AUROC: `uv run python -m distal.maha_auroc`
- Save new stats to `reece-omahoney/maha-stats-test` (not the original
  `maha-stats`) by passing `--hub_repo_id reece-omahoney/maha-stats-test`.
  Update `maha_auroc.py`'s default `maha_stats_repo_id` accordingly if the stats
  repo changes.

## Important constraints

- Read `CLAUDE.md` for project conventions before writing code.
- Run `uv run pre-commit run --files <file>` on every file you modify and fix
  any issues before considering the change done.
- The project uses **UV** as package manager. Run scripts with `uv run`.
- `sklearn` is already available (used in `maha_auroc.py`).
- Do not modify `distal/embedding.py` -- the embedding function is correct.
- After modifying `compute_maha_stats.py`, you must re-run it to regenerate
  stats before re-evaluating AUROC. This requires a GPU.

## Approaches tried

Record each approach here with its AUROC result so we don't repeat work. Update the log for a result immediately after observing it and before running the next experiment.

- **Baseline** (max per episode, naive cov): AUROC = 0.7579
- **Mean aggregation** (mean per episode, naive cov): AUROC = 0.8849 (+0.127)
- **Ledoit-Wolf covariance** (mean per episode, Ledoit-Wolf cov): AUROC = 0.8869
  (+0.002 over mean alone — marginal; main gain was from mean aggregation)
  Stats pushed to `reece-omahoney/maha-stats-test`. Default `maha_stats_repo_id` updated.
- Other aggregations tested (all with naive cov): p90=0.8214, p95=0.8115,
  mean_top10pct=0.7976, max=0.7659
- **Temporal aggregations** (Ledoit-Wolf stats): weighting later frames more
  heavily improves AUROC significantly — failures drift OOD toward episode end.
  - weighted_mean (linear ramp): AUROC = 0.9266 (+0.040 over mean)
  - mean_last50pct: AUROC = 0.9246
  - mean_last25pct: AUROC = 0.8968
- **PCA-256**: hurts badly AUROC = 0.7956

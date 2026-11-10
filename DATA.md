# Data format

A dataset lives in `public/` with three splits: `train/`, `validation/` and `test/`. Splits contain separate episodes.

| File | Splits | Contents |
| --- | --- | --- |
| `queries.csv` | all | `query_id`, `episode_id`, `machine_id`, `machine_type`, `signal`, `cutoff` |
| `history.csv` | all | `episode_id`, `machine_id`, `tick`, `payout` |
| `labels.csv` | train, validation | Observed next payout (`outcome`) for each query |
| `conditions.csv` | train, validation | Condition of each episode: stable, short history, drift or shared change |
| `targets.csv` | validation | True probabilities and the sampled outcome |
| `reference.csv` | validation | Pooled-baseline forecasts used for normalization |
| `sample_submission.csv` | all | Template with the required query IDs |
| `metadata.json` | root | Data version, dataset name, season ID and sizes |

## Fields

- A **query** asks for one machine's next spin at its `cutoff` tick (192 to 256).
- An **episode** is a group of related machines observed at the same cutoff. Group histories by `episode_id` and `machine_id`.
- `machine_type` is `steady`, `balanced` or `jackpot`. `signal` is -1, 0 or 1.
- `payout` is the gross return on a one-chip wager: 0, 1, 2, 5 or 20.

## Histories

- Use only ticks below the cutoff, including other machines in the same episode.
- Long histories contain every earlier tick. Short histories contain 40 recent consecutive ticks and 24 older sampled ticks.
- A missing tick is unknown, not a loss.
- Past payouts interact at lags of up to eight ticks. Win chances and payout sizes can change, sometimes across several machines together.

## Forecast file

```
query_id,p_0,p_1,p_2,p_5,p_20
```

One row per query, with finite probabilities in the order 0, 1, 2, 5, 20 that sum to 1. `check_submission.py` validates the file.

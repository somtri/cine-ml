# Model Card

## Model Details

- **Target:** IMDb rating
- **Champion:** Extra Trees regressor
- **Tuning:** 5-fold GridSearchCV on the training partition
- **Evaluation:** fixed 80/20 holdout split with random state 42
- **Metrics:** MAE 0.239, MSE 0.148, R2 0.835

## Intended Use

The champion estimates IMDb ratings after a film has accumulated TMDB audience
signals. It can support comparative analysis, missing-rating estimation, and
cross-platform consistency checks.

It should not be presented as a pre-release quality forecast because
`vote_average`, `vote_count`, popularity, and revenue are post-release signals.
For pre-release analysis, use the separately reported engineered model
(MAE 0.630, R2 0.298).

## Features

The full model includes:

- Log-transformed budget, revenue, popularity, and vote count
- Runtime and release date features
- Genre multi-hot indicators
- Language and metadata-count features
- TMDB vote average for the post-release champion

## Limitations

- The current OMDb-labeled sample contains 491 films and overrepresents
  comparatively visible, higher-budget titles.
- IMDb and TMDB scores are both audience aggregates, so the champion is a
  cross-platform estimator rather than an independent artistic-quality model.
- A random holdout does not measure performance under temporal distribution
  shift.
- Ratings and popularity evolve after release.

## Responsible Reporting

Always report the feature-availability stage beside a metric. Do not quote the
champion R2 as evidence that budget, runtime, and genres alone explain IMDb
ratings; those pre-release features produce R2 0.298 in this dataset.

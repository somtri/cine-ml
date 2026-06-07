# Data Card

## Sources

### TMDB 5000 Movie Dataset

- 4,803 movie rows
- 20 original columns
- Includes budget, runtime, genres, release date, language, popularity,
  revenue, vote average, and vote count

### OMDb Labels

- 491 cached films with valid IMDb ratings
- Titles are normalized and joined to TMDB metadata
- Missing and `N/A` ratings are removed
- Duplicate normalized titles are removed

## Collection Pipeline

`cine_ml.collect_omdb` is a resumable API collector. It:

1. Loads the existing OMDb cache.
2. Selects unmatched TMDB movie titles.
3. Queries OMDb concurrently with timeouts.
4. Parses numeric ratings and vote counts.
5. Deduplicates normalized titles.
6. Saves the merged cache to `data/data/omdb_movies.csv`.

The current repository does not claim 1,000 OMDb API labels because that run
requires a fresh user-owned API key. The collector target is 1,100 records.

## Quality Checks

- IMDb rating is numeric and non-null.
- TMDB title joins are one-to-one after normalization.
- Feature engineering handles malformed JSON-like metadata safely.
- Models are evaluated only on rows with valid targets.

## Known Biases

- Selection favors films represented in the TMDB 5000 dataset.
- Existing labels skew toward more prominent and higher-budget films.
- Audience ratings reflect platform demographics and voting behavior.

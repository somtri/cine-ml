from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "data"
TMDB_PATH = DATA_DIR / "tmdb_5000_movies.csv"
LEGACY_OMDB_PATH = DATA_DIR / "cleaned_movies.csv"
OMDB_PATH = DATA_DIR / "omdb_movies.csv"

BASIC_FEATURES = ["budget_log", "runtime"]
PRE_RELEASE_NUMERIC_FEATURES = [
    "budget_log",
    "runtime",
    "release_year",
    "release_month",
    "overview_length",
    "tagline_length",
    "keyword_count",
    "company_count",
    "country_count",
    "language_count",
]
POST_RELEASE_NUMERIC_FEATURES = PRE_RELEASE_NUMERIC_FEATURES + [
    "revenue_log",
    "popularity_log",
    "vote_count_log",
]
AUDIENCE_NUMERIC_FEATURES = POST_RELEASE_NUMERIC_FEATURES + ["vote_average"]
CATEGORICAL_FEATURES = ["original_language"]


def normalize_title(value: object) -> str:
    text = str(value or "").casefold()
    return re.sub(r"[^a-z0-9]+", "", text)


def parse_names(value: object) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    return [
        str(item.get("name", "")).strip()
        for item in parsed
        if isinstance(item, dict) and item.get("name")
    ]


def load_omdb_data(path: Path = OMDB_PATH) -> pd.DataFrame:
    if path.exists():
        omdb = pd.read_csv(path)
    else:
        omdb = pd.read_csv(LEGACY_OMDB_PATH)

    if "imdb_rating" not in omdb.columns and "imdbRating" in omdb.columns:
        omdb["imdb_rating"] = pd.to_numeric(omdb["imdbRating"], errors="coerce")

    omdb["imdb_rating"] = pd.to_numeric(omdb["imdb_rating"], errors="coerce")
    omdb["title_key"] = omdb["title"].map(normalize_title)
    return omdb.dropna(subset=["imdb_rating"]).drop_duplicates("title_key")


def load_modeling_data(
    tmdb_path: Path = TMDB_PATH,
    omdb_path: Path = OMDB_PATH,
) -> pd.DataFrame:
    tmdb = pd.read_csv(tmdb_path)
    tmdb["title_key"] = tmdb["title"].map(normalize_title)
    tmdb = tmdb.drop_duplicates("title_key")

    omdb = load_omdb_data(omdb_path)
    merged = omdb[["title_key", "imdb_rating"]].merge(
        tmdb,
        on="title_key",
        how="inner",
        validate="one_to_one",
    )
    return merged.dropna(subset=["imdb_rating"]).reset_index(drop=True)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    release_date = pd.to_datetime(df["release_date"], errors="coerce")

    features["title"] = df["title"]
    features["imdb_rating"] = pd.to_numeric(df["imdb_rating"], errors="coerce")
    features["budget_log"] = np.log1p(pd.to_numeric(df["budget"], errors="coerce"))
    features["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    features["release_year"] = release_date.dt.year
    features["release_month"] = release_date.dt.month
    features["overview_length"] = df["overview"].fillna("").str.len()
    features["tagline_length"] = df["tagline"].fillna("").str.len()
    features["original_language"] = df["original_language"].fillna("unknown")
    features["revenue_log"] = np.log1p(
        pd.to_numeric(df["revenue"], errors="coerce").clip(lower=0)
    )
    features["popularity_log"] = np.log1p(
        pd.to_numeric(df["popularity"], errors="coerce").clip(lower=0)
    )
    features["vote_count_log"] = np.log1p(
        pd.to_numeric(df["vote_count"], errors="coerce").clip(lower=0)
    )
    features["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")

    parsed_columns = {
        "genres": "genre",
        "keywords": "keyword",
        "production_companies": "company",
        "production_countries": "country",
        "spoken_languages": "language",
    }
    parsed: dict[str, pd.Series] = {}
    for source, prefix in parsed_columns.items():
        parsed[source] = df[source].map(parse_names)
        features[f"{prefix}_count"] = parsed[source].str.len()

    all_genres = sorted({genre for values in parsed["genres"] for genre in values})
    for genre in all_genres:
        column = f"genre_{re.sub(r'[^a-z0-9]+', '_', genre.casefold()).strip('_')}"
        features[column] = parsed["genres"].map(lambda values, g=genre: int(g in values))

    return features


def feature_columns(
    frame: pd.DataFrame,
    stage: str,
) -> tuple[list[str], list[str]]:
    genre_columns = sorted(column for column in frame if column.startswith("genre_"))
    numeric_by_stage = {
        "basic": BASIC_FEATURES,
        "pre_release": PRE_RELEASE_NUMERIC_FEATURES + genre_columns,
        "post_release": POST_RELEASE_NUMERIC_FEATURES + genre_columns,
        "audience_signal": AUDIENCE_NUMERIC_FEATURES + genre_columns,
    }
    if stage not in numeric_by_stage:
        raise ValueError(f"Unknown feature stage: {stage}")

    categorical = [] if stage == "basic" else CATEGORICAL_FEATURES
    return numeric_by_stage[stage], categorical

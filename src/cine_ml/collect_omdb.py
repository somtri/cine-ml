from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from .features import LEGACY_OMDB_PATH, OMDB_PATH, TMDB_PATH, normalize_title

OMDB_URL = "https://www.omdbapi.com/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a resumable OMDb rating cache.")
    parser.add_argument("--target", type=int, default=1100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OMDB_PATH)
    return parser.parse_args()


def fetch_movie(
    title: str,
    api_key: str,
) -> dict[str, object] | None:
    response = requests.get(
        OMDB_URL,
        params={"apikey": api_key, "t": title, "type": "movie"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("Response") != "True" or payload.get("imdbRating") == "N/A":
        return None

    return {
        "title": title,
        "omdb_title": payload.get("Title"),
        "imdb_id": payload.get("imdbID"),
        "imdb_rating": pd.to_numeric(payload.get("imdbRating"), errors="coerce"),
        "imdb_votes": pd.to_numeric(
            str(payload.get("imdbVotes", "")).replace(",", ""),
            errors="coerce",
        ),
        "metascore": pd.to_numeric(payload.get("Metascore"), errors="coerce"),
        "omdb_year": payload.get("Year"),
        "rated": payload.get("Rated"),
        "director": payload.get("Director"),
        "source": "omdb_api",
    }


def load_cache(path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if LEGACY_OMDB_PATH.exists():
        legacy = pd.read_csv(LEGACY_OMDB_PATH)
        legacy["source"] = "legacy_omdb_cache"
        frames.append(legacy)
    if path.exists():
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame(columns=["title", "imdb_rating"])

    cache = pd.concat(frames, ignore_index=True)
    cache["title_key"] = cache["title"].map(normalize_title)
    cache["imdb_rating"] = pd.to_numeric(cache["imdb_rating"], errors="coerce")
    return (
        cache.dropna(subset=["imdb_rating"])
        .drop_duplicates("title_key", keep="last")
        .drop(columns="title_key")
    )


def collect(target: int, workers: int, output: Path) -> pd.DataFrame:
    api_key = os.getenv("OMDB_API_KEY")
    if not api_key:
        raise RuntimeError("Set OMDB_API_KEY before collecting OMDb records.")

    cache = load_cache(output)
    if len(cache) >= target:
        return cache

    tmdb = pd.read_csv(TMDB_PATH)
    tmdb = tmdb[
        (tmdb["runtime"].fillna(0) > 0)
        & (tmdb["vote_count"].fillna(0) >= 20)
    ].copy()
    tmdb["title_key"] = tmdb["title"].map(normalize_title)
    cached_keys = set(cache["title"].map(normalize_title))

    candidates = tmdb[~tmdb["title_key"].isin(cached_keys)].copy()
    candidates = candidates.sort_values(
        ["vote_count", "popularity"],
        ascending=False,
    )
    requested = max(target - len(cache), 0)
    titles = candidates["title"].head(requested + 250).tolist()

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_movie, title, api_key): title
            for title in titles
        }
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                row = future.result()
            except requests.RequestException:
                row = None
            if row:
                rows.append(row)
            if index % 50 == 0:
                print(f"Processed {index}/{len(titles)} requests")
            if len(cache) + len(rows) >= target:
                break
            time.sleep(0.02)

    collected = pd.DataFrame(rows)
    combined = pd.concat([cache, collected], ignore_index=True)
    combined["title_key"] = combined["title"].map(normalize_title)
    combined = (
        combined.dropna(subset=["imdb_rating"])
        .drop_duplicates("title_key", keep="last")
        .drop(columns="title_key")
        .sort_values("title")
        .reset_index(drop=True)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    return combined


def main() -> None:
    args = parse_args()
    data = collect(args.target, args.workers, args.output)
    print(f"Saved {len(data):,} OMDb-labeled films to {args.output}")


if __name__ == "__main__":
    main()

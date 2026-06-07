import pandas as pd

from cine_ml.features import build_feature_frame, normalize_title, parse_names


def test_normalize_title_removes_punctuation() -> None:
    assert normalize_title("Spider-Man: Homecoming") == "spidermanhomecoming"


def test_parse_names_handles_invalid_values() -> None:
    assert parse_names("not valid") == []
    assert parse_names(None) == []


def test_build_feature_frame_encodes_genres() -> None:
    raw = pd.DataFrame(
        {
            "title": ["Example"],
            "imdb_rating": [7.2],
            "budget": [10_000_000],
            "runtime": [105],
            "release_date": ["2020-05-01"],
            "overview": ["A test movie"],
            "tagline": ["Testing"],
            "original_language": ["en"],
            "revenue": [30_000_000],
            "popularity": [12.0],
            "vote_count": [500],
            "vote_average": [7.0],
            "genres": ['[{"id": 18, "name": "Drama"}]'],
            "keywords": ["[]"],
            "production_companies": ["[]"],
            "production_countries": ["[]"],
            "spoken_languages": ["[]"],
        }
    )
    features = build_feature_frame(raw)
    assert features.loc[0, "genre_drama"] == 1
    assert features.loc[0, "release_year"] == 2020
    assert features.loc[0, "budget_log"] > 0

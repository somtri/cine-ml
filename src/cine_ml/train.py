from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import (
    BASIC_FEATURES,
    PROJECT_ROOT,
    build_feature_frame,
    feature_columns,
    load_modeling_data,
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


@dataclass
class Evaluation:
    name: str
    stage: str
    mae: float
    mse: float
    r2: float


def make_preprocessor(
    numeric: list[str],
    categorical: list[str],
    scale: bool,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers: list[tuple[str, object, list[str]]] = [
        ("numeric", Pipeline(numeric_steps), numeric)
    ]
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=3,
                    sparse_output=False,
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def evaluate(name: str, stage: str, y_true: pd.Series, prediction: np.ndarray) -> Evaluation:
    return Evaluation(
        name=name,
        stage=stage,
        mae=float(mean_absolute_error(y_true, prediction)),
        mse=float(mean_squared_error(y_true, prediction)),
        r2=float(r2_score(y_true, prediction)),
    )


def save_figures(
    leaderboard: pd.DataFrame,
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.barplot(data=leaderboard, x="mae", y="model", ax=axes[0], color="#10b981")
    axes[0].set_title("Holdout MAE (lower is better)")
    sns.barplot(data=leaderboard, x="r2", y="model", ax=axes[1], color="#6366f1")
    axes[1].set_title("Holdout R2 (higher is better)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_performance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        data=predictions,
        x="actual",
        y="predicted",
        alpha=0.7,
        color="#6366f1",
        ax=ax,
    )
    low = min(predictions["actual"].min(), predictions["predicted"].min())
    high = max(predictions["actual"].max(), predictions["predicted"].max())
    ax.plot([low, high], [low, high], "--", color="#ef4444")
    ax.set_title("Champion model: actual vs predicted")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "actual_vs_predicted.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    top = importance.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"], color="#f59e0b")
    ax.set_title("Top feature importances")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_modeling_data()
    frame = build_feature_frame(raw).dropna(subset=["imdb_rating"]).reset_index(drop=True)
    train, test = train_test_split(frame, test_size=0.2, random_state=42)
    y_train = train["imdb_rating"]
    y_test = test["imdb_rating"]

    evaluations: list[Evaluation] = []
    fitted: dict[str, Pipeline] = {}

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(train[BASIC_FEATURES], y_train)
    baseline_prediction = baseline.predict(test[BASIC_FEATURES])
    evaluations.append(evaluate("Mean baseline", "basic", y_test, baseline_prediction))

    basic_numeric, _ = feature_columns(frame, "basic")
    basic_model = Pipeline(
        [
            ("preprocessor", make_preprocessor(basic_numeric, [], scale=True)),
            ("model", Ridge(alpha=10.0)),
        ]
    )
    basic_model.fit(train, y_train)
    evaluations.append(
        evaluate(
            "Budget + runtime",
            "basic",
            y_test,
            basic_model.predict(test),
        )
    )
    fitted["Budget + runtime"] = basic_model

    for stage, label in [
        ("pre_release", "Engineered pre-release"),
        ("post_release", "Post-release signals"),
    ]:
        numeric, categorical = feature_columns(frame, stage)
        model = Pipeline(
            [
                ("preprocessor", make_preprocessor(numeric, categorical, scale=True)),
                ("model", Ridge(alpha=10.0)),
            ]
        )
        model.fit(train, y_train)
        evaluations.append(evaluate(label, stage, y_test, model.predict(test)))
        fitted[label] = model

    champion_stage = "audience_signal"
    numeric, categorical = feature_columns(frame, champion_stage)
    champion = Pipeline(
        [
            ("preprocessor", make_preprocessor(numeric, categorical, scale=False)),
            (
                "model",
                ExtraTreesRegressor(random_state=42, n_jobs=1),
            ),
        ]
    )
    search = GridSearchCV(
        champion,
        {
            "model__n_estimators": [300, 600],
            "model__max_depth": [None, 12],
            "model__min_samples_leaf": [1, 2, 3],
            "model__max_features": [0.7, 1.0],
        },
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        refit=True,
    )
    search.fit(train, y_train)
    champion = search.best_estimator_
    champion_prediction = champion.predict(test)
    evaluations.append(
        evaluate(
            "Tuned audience-signal model",
            champion_stage,
            y_test,
            champion_prediction,
        )
    )

    leaderboard = pd.DataFrame([evaluation.__dict__ for evaluation in evaluations])
    leaderboard = leaderboard.rename(columns={"name": "model"})
    leaderboard = leaderboard.sort_values("mae").reset_index(drop=True)
    leaderboard.to_csv(ARTIFACTS_DIR / "model_leaderboard.csv", index=False)

    predictions = pd.DataFrame(
        {
            "title": test["title"],
            "actual": y_test,
            "predicted": champion_prediction,
        }
    )
    predictions["absolute_error"] = (
        predictions["actual"] - predictions["predicted"]
    ).abs()
    predictions.to_csv(ARTIFACTS_DIR / "test_predictions.csv", index=False)

    feature_names = champion.named_steps["preprocessor"].get_feature_names_out()
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": champion.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(ARTIFACTS_DIR / "feature_importance.csv", index=False)

    joblib.dump(champion, ARTIFACTS_DIR / "rating_model.joblib")
    joblib.dump(
        {
            "columns": list(frame.columns),
            "numeric_features": numeric,
            "categorical_features": categorical,
            "stage": champion_stage,
        },
        ARTIFACTS_DIR / "model_schema.joblib",
    )

    baseline_metrics = next(item for item in evaluations if item.name == "Mean baseline")
    champion_metrics = next(
        item for item in evaluations if item.name == "Tuned audience-signal model"
    )
    improvement = (baseline_metrics.mae - champion_metrics.mae) / baseline_metrics.mae
    metrics = {
        "dataset_rows": int(len(frame)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "baseline": baseline_metrics.__dict__,
        "champion": champion_metrics.__dict__,
        "mae_improvement_pct": float(improvement * 100),
        "best_params": search.best_params_,
        "random_state": 42,
        "target": "IMDb rating",
        "important_note": (
            "The champion is a post-release model and includes TMDB vote_average. "
            "The pre-release model is reported separately to avoid target-proxy confusion."
        ),
    }
    (ARTIFACTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    save_figures(leaderboard, predictions, importance)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

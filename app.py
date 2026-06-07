from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(
    page_title="CineScore ML",
    page_icon="CS",
    layout="wide",
)


@st.cache_data
def load_artifacts() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    leaderboard = pd.read_csv(ARTIFACTS / "model_leaderboard.csv")
    predictions = pd.read_csv(ARTIFACTS / "test_predictions.csv")
    importance = pd.read_csv(ARTIFACTS / "feature_importance.csv")
    return metrics, leaderboard, predictions, importance


metrics, leaderboard, predictions, importance = load_artifacts()

st.title("CineScore ML")
st.caption(
    "IMDb rating prediction from TMDB metadata and OMDb labels, with explicit "
    "separation between pre-release and post-release feature sets."
)

champion = metrics["champion"]
baseline = metrics["baseline"]
columns = st.columns(4)
columns[0].metric("Movies", f"{metrics['dataset_rows']:,}")
columns[1].metric("Champion MAE", f"{champion['mae']:.3f}")
columns[2].metric("Champion R2", f"{champion['r2']:.3f}")
columns[3].metric(
    "MAE improvement",
    f"{metrics['mae_improvement_pct']:.1f}%",
    delta=f"vs {baseline['mae']:.3f} baseline",
)

st.info(metrics["important_note"])

tab_overview, tab_errors, tab_features, tab_method = st.tabs(
    ["Model comparison", "Error analysis", "Feature importance", "Methodology"]
)

with tab_overview:
    metric = st.radio("Metric", ["mae", "r2"], horizontal=True)
    metric_min = min(0.0, float(leaderboard[metric].min()) * 1.1)
    metric_max = float(leaderboard[metric].max()) * 1.1
    chart = (
        alt.Chart(leaderboard)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X(
                f"{metric}:Q",
                title=metric.upper(),
                stack=None,
                scale=alt.Scale(domain=[metric_min, metric_max]),
            ),
            y=alt.Y("model:N", sort="-x", title=None),
            color=alt.Color("stage:N", title="Feature stage"),
            tooltip=["model", "stage", "mae", "mse", "r2"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(
        leaderboard.style.format({"mae": "{:.3f}", "mse": "{:.3f}", "r2": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )

with tab_errors:
    max_error = float(predictions["absolute_error"].max())
    scatter = (
        alt.Chart(predictions)
        .mark_circle(size=65, opacity=0.65)
        .encode(
            x=alt.X("actual:Q", scale=alt.Scale(domain=[1, 10])),
            y=alt.Y("predicted:Q", scale=alt.Scale(domain=[1, 10])),
            color=alt.Color(
                "absolute_error:Q",
                scale=alt.Scale(domain=[0, max_error], scheme="plasma"),
            ),
            tooltip=["title", "actual", "predicted", "absolute_error"],
        )
        .properties(height=440)
    )
    diagonal = (
        alt.Chart(pd.DataFrame({"x": [1, 10], "y": [1, 10]}))
        .mark_line(strokeDash=[6, 4], color="#ef4444")
        .encode(x="x:Q", y="y:Q")
    )
    st.altair_chart(scatter + diagonal, use_container_width=True)
    st.subheader("Largest holdout errors")
    st.dataframe(
        predictions.nlargest(10, "absolute_error"),
        use_container_width=True,
        hide_index=True,
    )

with tab_features:
    top_features = importance.head(20)
    max_importance = float(top_features["importance"].max())
    chart = (
        alt.Chart(top_features)
        .mark_bar(cornerRadiusEnd=5, color="#f59e0b")
        .encode(
            x=alt.X(
                "importance:Q",
                title="Feature importance",
                stack=None,
                scale=alt.Scale(domain=[0, max_importance * 1.05]),
            ),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature", "importance"],
        )
        .properties(height=520)
    )
    st.altair_chart(chart, use_container_width=True)

with tab_method:
    st.markdown(
        """
        **Target:** IMDb rating collected from OMDb.

        **Source features:** TMDB budget, runtime, release date, genres, language,
        textual metadata counts, popularity, revenue, vote count, and vote average.

        **Evaluation:** A fixed 80/20 holdout split is reserved before model tuning.
        GridSearchCV tunes the Extra Trees model on the training partition only.
        MAE, MSE, and R2 are reported on the untouched holdout set.

        **Interpretation:** The audience-signal model is useful for cross-platform
        rating estimation after release. The pre-release model is the honest choice
        when forecasting a film before audience signals exist.
        """
    )

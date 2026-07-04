"""
preprocess.py
Feature engineering for YouTube analytics.
Transforms raw API data into ML-ready features.
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime


# ── Keywords likely relevant for anime/collectibles channels ──────────────────
HYPE_WORDS = [
    "unboxing", "haul", "review", "figure", "merch", "limited",
    "exclusive", "rare", "collection", "grail", "worth it", "honest",
    "worst", "best", "ranked", "tier list", "vs", "vs.", "comparison",
    "cheap", "expensive", "budget", "spend", "price", "bought",
    "giveaway", "free", "huge", "massive", "insane", "crazy",
    "announcement", "news", "update", "2024", "2025", "new"
]

NEGATIVE_SENTIMENT_WORDS = ["worst", "terrible", "bad", "hate", "never", "stop", "quit"]
QUESTION_WORDS = ["how", "why", "what", "when", "which", "should", "can"]


def load_raw_data(path="data/raw/videos.csv"):
    df = pd.read_csv(path)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df


def engineer_features(df):
    """
    Main feature engineering function.
    Takes raw video DataFrame, returns feature-engineered version.
    """
    df = df.copy()

    # ── TIME FEATURES ─────────────────────────────────────────────────────────
    df["publish_year"] = df["published_at"].dt.year
    df["publish_month"] = df["published_at"].dt.month
    df["publish_day_of_week"] = df["published_at"].dt.dayofweek  # 0=Mon, 6=Sun
    df["publish_hour"] = df["published_at"].dt.hour

    # Days since video was published (older videos have had more time to accumulate views)
    now = pd.Timestamp.now(tz="UTC")
    df["days_since_published"] = (now - df["published_at"]).dt.days.clip(lower=1)

    # Weekend upload (Sat=5, Sun=6)
    df["uploaded_on_weekend"] = df["publish_day_of_week"].isin([5, 6]).astype(int)

    # ── TITLE FEATURES ────────────────────────────────────────────────────────
    df["title_length"] = df["title"].str.len()
    df["title_word_count"] = df["title"].str.split().str.len()
    df["title_upper_ratio"] = df["title"].apply(
        lambda t: sum(1 for c in str(t) if c.isupper()) / max(len(str(t)), 1)
    )
    df["title_has_number"] = df["title"].str.contains(r'\d', na=False).astype(int)
    df["title_has_emoji"] = df["title"].apply(
        lambda t: bool(re.search(r'[^\x00-\x7F]', str(t)))
    ).astype(int)
    df["title_has_brackets"] = df["title"].str.contains(r'[\[\]\(\)]', na=False).astype(int)
    df["title_is_question"] = df["title"].str.strip().str.endswith("?").astype(int)
    df["title_has_pipe"] = df["title"].str.contains(r'\|', na=False).astype(int)

    # Hype word count in title
    df["title_hype_word_count"] = df["title"].apply(
        lambda t: sum(1 for w in HYPE_WORDS if w.lower() in str(t).lower())
    )

    # Negative sentiment in title
    df["title_negative_sentiment"] = df["title"].apply(
        lambda t: sum(1 for w in NEGATIVE_SENTIMENT_WORDS if w.lower() in str(t).lower())
    ).clip(upper=1)

    # Question words in title
    df["title_is_question_word"] = df["title"].apply(
        lambda t: int(any(str(t).lower().startswith(w) for w in QUESTION_WORDS))
    )

    # ── DESCRIPTION FEATURES ─────────────────────────────────────────────────
    df["description_length"] = df["description"].fillna("").str.len()
    df["description_has_links"] = df["description"].fillna("").str.contains(
        r'http[s]?://', na=False
    ).astype(int)
    df["description_has_timestamps"] = df["description"].fillna("").str.contains(
        r'\d+:\d+', na=False
    ).astype(int)

    # ── VIDEO FEATURES ────────────────────────────────────────────────────────
    df["duration_minutes"] = df["duration_seconds"] / 60
    df["is_short"] = (df["duration_seconds"] <= 60).astype(int)  # YouTube Shorts
    df["is_long"] = (df["duration_minutes"] >= 20).astype(int)

    # Duration bins
    df["duration_bin"] = pd.cut(
        df["duration_minutes"],
        bins=[0, 1, 5, 10, 20, 40, float("inf")],
        labels=["short", "quick", "medium", "long", "very_long", "marathon"]
    )

    # ── TAG FEATURES ─────────────────────────────────────────────────────────
    df["tag_count"] = df["tags"].fillna("").apply(
        lambda t: len(t.split("|")) if t else 0
    )

    # ── ENGAGEMENT FEATURES (for EDA, not for predicting views — that's leakage) ──
    df["like_rate"] = df["like_count"] / df["view_count"].clip(lower=1)
    df["comment_rate"] = df["comment_count"] / df["view_count"].clip(lower=1)

    # ── NORMALISED VIEW COUNT (views per day — accounts for video age) ────────
    df["views_per_day"] = df["view_count"] / df["days_since_published"]
    df["log_view_count"] = np.log1p(df["view_count"])  # log transform for regression
    df["log_views_per_day"] = np.log1p(df["views_per_day"])

    # ── CHANNEL FEATURES ─────────────────────────────────────────────────────
    df["log_subscriber_count"] = np.log1p(df["subscriber_count"])

    return df


def get_feature_columns():
    """
    Returns the list of features to use for ML modelling.
    Excludes target variables, IDs, and anything that would cause data leakage.
    """
    return [
        # Time features
        "publish_month",
        "publish_day_of_week",
        "publish_hour",
        "uploaded_on_weekend",

        # Title features
        "title_length",
        "title_word_count",
        "title_upper_ratio",
        "title_has_number",
        "title_has_emoji",
        "title_has_brackets",
        "title_is_question",
        "title_has_pipe",
        "title_hype_word_count",
        "title_negative_sentiment",
        "title_is_question_word",

        # Description features
        "description_length",
        "description_has_links",
        "description_has_timestamps",

        # Video features
        "duration_minutes",
        "is_short",
        "is_long",
        "tag_count",
        "has_custom_thumbnail",

        # Channel features
        "log_subscriber_count",
    ]


def prepare_for_modelling(df, target="log_views_per_day"):
    """
    Returns X (features) and y (target) ready for sklearn.
    Drops rows with missing values in feature columns.
    """
    feature_cols = get_feature_columns()

    # Only keep columns that exist in the dataframe
    available_cols = [c for c in feature_cols if c in df.columns]

    model_df = df[available_cols + [target]].dropna()

    X = model_df[available_cols]
    y = model_df[target]

    return X, y, available_cols


if __name__ == "__main__":
    df = load_raw_data()
    df = engineer_features(df)
    print(f"Features engineered. Shape: {df.shape}")
    print(f"\nFeature columns available: {get_feature_columns()}")
    print(f"\nSample:")
    print(df[["title", "view_count", "views_per_day", "title_hype_word_count"]].head())
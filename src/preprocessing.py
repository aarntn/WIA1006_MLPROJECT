"""
Preprocessing utilities. Keep pure functions here — no global state.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def parse_interest_tags(series: pd.Series) -> list[list[str]]:
    """Turn the comma-separated interest_tags column into a list of lists."""
    return [[t.strip() for t in str(s).split(",") if t.strip()] for s in series]


def multi_hot_interests(series: pd.Series) -> pd.DataFrame:
    """
    Multi-hot encode interest_tags.
    Returns a DataFrame with one binary column per unique interest.
    """
    parsed = parse_interest_tags(series)
    all_tags = sorted({t for row in parsed for t in row})
    data = {tag: [1 if tag in row else 0 for row in parsed] for tag in all_tags}
    return pd.DataFrame(data, index=series.index)


def engineer_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derived behavioral features. Not signal-guaranteed, but worth trying.
    Returns a new DataFrame with only the engineered columns (caller concats).
    """
    eps = 1e-6
    out = pd.DataFrame(index=df.index)
    out["messages_per_match"] = df["message_sent_count"] / (df["mutual_matches"] + eps)
    out["likes_to_swipe_ratio"] = df["likes_received"] / (df["app_usage_time_min"] + eps)
    out["bio_effort"] = df["bio_length"] * df["profile_pics_count"]
    out["night_user"] = ((df["last_active_hour"] >= 22) | (df["last_active_hour"] <= 4)).astype(int)
    out["emoji_heavy"] = (df["emoji_usage_rate"] > 0.5).astype(int)
    return out


def label_encode(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns in place on a copy. Returns (encoded_df, encoders_dict).
    Keep encoders if you need to decode later.
    """
    out = df.copy()
    encoders = {}
    for c in cols:
        le = LabelEncoder()
        out[c] = le.fit_transform(out[c].astype(str))
        encoders[c] = le
    return out, encoders


def outcome_3class(series: pd.Series) -> pd.Series:
    """Collapse the 10-class match_outcome into 3 classes."""
    mapping = {
        "Blocked": "Negative",
        "Catfished": "Negative",
        "Ghosted": "Negative",
        "Chat Ignored": "Negative",
        "No Action": "Neutral",
        "One-sided Like": "Neutral",
        "Instant Match": "Positive",
        "Mutual Match": "Positive",
        "Date Happened": "Positive",
        "Relationship Formed": "Positive",
    }
    return series.map(mapping)

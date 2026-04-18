"""
Data loading utilities for Swipe Atlas.

Usage:
    from src.data import load_raw, CAT_COLS, NUM_COLS, TARGET
    df = load_raw(extended=True)
"""
from pathlib import Path
import pandas as pd

# Project root is the parent of src/
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"

# Column groupings — single source of truth for the whole project.
# Edit here if columns change.
NUM_COLS = [
    "app_usage_time_min",
    "swipe_right_ratio",
    "likes_received",
    "mutual_matches",
    "profile_pics_count",
    "bio_length",
    "message_sent_count",
    "emoji_usage_rate",
    "last_active_hour",
]

CAT_COLS = [
    "gender",
    "sexual_orientation",
    "location_type",
    "income_bracket",
    "education_level",
    "app_usage_time_label",
    "swipe_right_label",
    "swipe_time_of_day",
]

# Only present in the extended CSV
EXTENDED_NUM_COLS = ["age", "height_cm", "weight_kg"]
EXTENDED_CAT_COLS = ["zodiac_sign", "body_type", "relationship_intent"]

MULTI_LABEL_COL = "interest_tags"
TARGET = "match_outcome"


def load_raw(extended: bool = True) -> pd.DataFrame:
    """Load the raw CSV. Pass extended=False for the original 19-column version."""
    fname = (
        "dating_app_behavior_dataset_extended1.csv"
        if extended
        else "dating_app_behavior_dataset.csv"
    )
    path = DATA_DIR / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Couldn't find {path}. Are you running from the project root?"
        )
    return pd.read_csv(path)


def get_feature_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) present in this dataframe."""
    num = [c for c in NUM_COLS + EXTENDED_NUM_COLS if c in df.columns]
    cat = [c for c in CAT_COLS + EXTENDED_CAT_COLS if c in df.columns]
    return num, cat

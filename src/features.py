"""
Reusable feature builders for Swipe Atlas modeling workflows.

The project uses the raw dating-app rows as the model input and lets these
transformers handle safe feature selection, engineered features, and encoding.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from src.data import MULTI_LABEL_COL, TARGET, get_feature_cols
from src.preprocessing import parse_interest_tags


PAIRED_TARGETS = {
    "mutual_matches": "likes_received",
    "likes_received": "mutual_matches",
}


@dataclass(frozen=True)
class FeatureConfig:
    """Controls leakage-sensitive feature construction."""

    target_col: str = "mutual_matches"
    include_paired_target: bool = False
    include_match_outcome: bool = False
    include_interest_tags: bool = True


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float) / denominator.replace(0, np.nan).astype(float)


def make_dense_onehot_encoder() -> OneHotEncoder:
    """Create a dense OneHotEncoder across old and new scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def add_engineered_features(
    df: pd.DataFrame,
    target_col: str = "mutual_matches",
    include_paired_target: bool = False,
) -> pd.DataFrame:
    """
    Add behavioral features that do not directly copy the modeling target.

    Ratio features that require the active target as a denominator are omitted
    for that target, so the official mutual_matches model does not receive
    messages_per_match.
    """
    out = df.copy()
    paired_target = PAIRED_TARGETS.get(target_col)

    def can_use(*cols: str) -> bool:
        blocked = {target_col}
        if paired_target and not include_paired_target:
            blocked.add(paired_target)
        return all(col not in blocked for col in cols)

    if {"bio_length", "profile_pics_count"}.issubset(out.columns):
        out["bio_effort"] = out["bio_length"] * out["profile_pics_count"]

    if "last_active_hour" in out.columns:
        out["night_user"] = (
            (out["last_active_hour"] >= 22) | (out["last_active_hour"] <= 4)
        ).astype(int)

    if "emoji_usage_rate" in out.columns:
        out["emoji_heavy"] = (out["emoji_usage_rate"] > 0.5).astype(int)

    if can_use("likes_received", "app_usage_time_min") and {"likes_received", "app_usage_time_min"}.issubset(out.columns):
        out["likes_per_usage_min"] = _safe_divide(
            out["likes_received"], out["app_usage_time_min"]
        ).fillna(0)

    if can_use("message_sent_count", "mutual_matches") and {"message_sent_count", "mutual_matches"}.issubset(out.columns):
        out["messages_per_match"] = _safe_divide(
            out["message_sent_count"], out["mutual_matches"]
        ).fillna(0)

    if can_use("mutual_matches", "likes_received") and {"mutual_matches", "likes_received"}.issubset(out.columns):
        out["match_yield_from_likes"] = _safe_divide(
            out["mutual_matches"], out["likes_received"]
        ).fillna(0)

    return out


class EngagementFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Convert raw rows into a dense numeric modeling matrix.

    This transformer is intentionally dataframe-first so the saved sklearn
    pipeline can be reused by the Streamlit app with raw user inputs.
    """

    def __init__(
        self,
        target_col: str = "mutual_matches",
        include_paired_target: bool = False,
        include_match_outcome: bool = False,
        include_interest_tags: bool = True,
    ) -> None:
        self.target_col = target_col
        self.include_paired_target = include_paired_target
        self.include_match_outcome = include_match_outcome
        self.include_interest_tags = include_interest_tags

    def fit(self, X: pd.DataFrame, y=None):
        df = self._prepare_dataframe(X)
        num_cols, cat_cols = get_feature_cols(df)

        drop_cols = {self.target_col}
        if not self.include_match_outcome:
            drop_cols.add(TARGET)

        paired = PAIRED_TARGETS.get(self.target_col)
        if paired and not self.include_paired_target:
            drop_cols.add(paired)

        self.numeric_cols_ = [
            c
            for c in df.select_dtypes(include=[np.number]).columns
            if c not in drop_cols
        ]
        self.categorical_cols_ = [
            c
            for c in cat_cols
            if c in df.columns and c not in drop_cols and c != MULTI_LABEL_COL
        ]

        self.onehot_ = make_dense_onehot_encoder()
        if self.categorical_cols_:
            self.onehot_.fit(df[self.categorical_cols_].astype(str))

        if self.include_interest_tags and MULTI_LABEL_COL in df.columns:
            tags = parse_interest_tags(df[MULTI_LABEL_COL])
            self.interest_tags_ = sorted({tag for row in tags for tag in row})
        else:
            self.interest_tags_ = []

        self.feature_names_ = self._feature_names()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = self._prepare_dataframe(X)
        parts: list[pd.DataFrame] = []

        if self.numeric_cols_:
            numeric = df.reindex(columns=self.numeric_cols_, fill_value=0)
            numeric = numeric.apply(pd.to_numeric, errors="coerce").fillna(0)
            parts.append(numeric.astype(float))

        if self.categorical_cols_:
            encoded = self.onehot_.transform(df[self.categorical_cols_].astype(str))
            _get_names = getattr(self.onehot_, "get_feature_names_out", None) or self.onehot_.get_feature_names
            cat_names = _get_names(self.categorical_cols_)
            parts.append(pd.DataFrame(encoded, columns=cat_names, index=df.index))

        if self.interest_tags_:
            tag_rows = parse_interest_tags(df.get(MULTI_LABEL_COL, pd.Series("", index=df.index)))
            tag_sets = [set(row) for row in tag_rows]
            tag_data = {
                f"{MULTI_LABEL_COL}__{tag}": [1.0 if tag in row else 0.0 for row in tag_sets]
                for tag in self.interest_tags_
            }
            parts.append(pd.DataFrame(tag_data, index=df.index))

        if not parts:
            return pd.DataFrame(index=df.index)

        return pd.concat(parts, axis=1).reindex(columns=self.feature_names_, fill_value=0)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(getattr(self, "feature_names_", []), dtype=object)

    def _prepare_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return add_engineered_features(
            X,
            target_col=self.target_col,
            include_paired_target=self.include_paired_target,
        )

    def _feature_names(self) -> list[str]:
        names = list(self.numeric_cols_)
        if self.categorical_cols_:
            _get_names = getattr(self.onehot_, "get_feature_names_out", None) or self.onehot_.get_feature_names
            names.extend(_get_names(self.categorical_cols_).tolist())
        names.extend(f"{MULTI_LABEL_COL}__{tag}" for tag in self.interest_tags_)
        return names

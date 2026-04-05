"""
preprocessing.py
----------------
Data loading, cleaning, feature engineering, and train/test splitting.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "study_hours", "attendance_pct", "prev_gpa", "assignments_pct",
    "sleep_hours", "parent_edu", "internet_access", "extracurricular",
    "travel_time", "part_time_job", "tutoring", "gender",
    "school_type", "peer_group", "absences",
]

TARGET_REG   = "score"   # regression target  (continuous 0–100)
TARGET_CLF   = "grade"   # classification target (Fail/Average/Pass/Distinction)

GRADE_ORDER  = ["Fail", "Average", "Pass", "Distinction"]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_data(path: str = "student_data.csv") -> pd.DataFrame:
    """Load CSV and do lightweight validation."""
    df = pd.read_csv(path)
    required = set(FEATURE_COLS + [TARGET_REG, TARGET_CLF])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    print(f"[load]  {len(df):,} rows · {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, fix dtypes, cap outliers."""
    df = df.copy()

    df.drop_duplicates(inplace=True)

    # Clamp numeric columns to valid ranges
    df["score"]           = df["score"].clip(0, 100)
    df["attendance_pct"]  = df["attendance_pct"].clip(0, 100)
    df["prev_gpa"]        = df["prev_gpa"].clip(0, 4)
    df["study_hours"]     = df["study_hours"].clip(0, 168)
    df["absences"]        = df["absences"].clip(0, None)

    print(f"[clean] {len(df):,} rows after dedup · "
          f"{df.isnull().sum().sum()} missing values total")
    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features that boost model signal."""
    df = df.copy()

    # Study efficiency: assignments done per study hour
    df["study_efficiency"] = (df["assignments_pct"] / (df["study_hours"] + 1)).round(3)

    # Optimal sleep flag (6–9 hrs is healthy)
    df["sleep_optimal"] = df["sleep_hours"].between(6, 9).astype(int)

    # Composite academic effort score
    df["effort_score"] = (
        df["study_hours"] * 0.4
        + df["attendance_pct"] * 0.4
        + df["assignments_pct"] * 0.2
    ).round(2)

    # Parental support proxy
    df["support_index"] = (
        df["parent_edu"] + df["internet_access"] * 2 + df["tutoring"] * 3
    )

    print(f"[feat]  Added 4 engineered features · "
          f"total feature cols = {len(FEATURE_COLS) + 4}")
    return df


def encode_target(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """Ordinal-encode the grade column."""
    le = LabelEncoder()
    le.classes_ = np.array(GRADE_ORDER)
    df = df.copy()
    df["grade_enc"] = le.transform(df[TARGET_CLF])
    return df, le


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict:
    """
    Returns a dict with keys:
        X_train, X_test, y_reg_train, y_reg_test,
        y_clf_train, y_clf_test, feature_names
    """
    # Extended feature list after engineering
    all_features = FEATURE_COLS + [
        "study_efficiency", "sleep_optimal", "effort_score", "support_index"
    ]

    X = df[all_features]
    y_reg = df[TARGET_REG]
    y_clf = df["grade_enc"]

    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_reg, y_clf,
        test_size=test_size,
        random_state=random_state,
        stratify=y_clf,
    )

    print(f"[split] Train {len(X_train):,} · Test {len(X_test):,}")
    return dict(
        X_train=X_train, X_test=X_test,
        y_reg_train=yr_train, y_reg_test=yr_test,
        y_clf_train=yc_train, y_clf_test=yc_test,
        feature_names=all_features,
    )


def make_preprocessor() -> Pipeline:
    """
    Sklearn Pipeline:  impute → scale
    (robust to unseen data with missing values)
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


# ── main entry ────────────────────────────────────────────────────────────────

def run_preprocessing(csv_path: str = "student_data.csv") -> tuple[dict, Pipeline, LabelEncoder]:
    df = load_data(csv_path)
    df = clean_data(df)
    df = feature_engineer(df)
    df, le = encode_target(df)
    splits = split_data(df)
    preprocessor = make_preprocessor()
    return splits, preprocessor, le


if __name__ == "__main__":
    splits, prep, le = run_preprocessing("student_data.csv")
    print("\nFeatures:", splits["feature_names"])
    print("Grade classes:", list(le.classes_))

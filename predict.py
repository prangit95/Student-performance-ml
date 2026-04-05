"""
predict.py
----------
Load a trained model and run inference on new student records.
Supports both single-student and batch (DataFrame / CSV) prediction.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_DIR = Path("models")


def load_artifacts(model_dir: Path = MODEL_DIR):
    clf        = joblib.load(model_dir / "best_classifier.pkl")
    reg        = joblib.load(model_dir / "best_regressor.pkl")
    le         = joblib.load(model_dir / "label_encoder.pkl")
    feat_names = joblib.load(model_dir / "feature_names.pkl")
    return clf, reg, le, feat_names


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the feature engineering done in preprocessing.py."""
    df = df.copy()
    df["study_efficiency"] = (df["assignments_pct"] / (df["study_hours"] + 1)).round(3)
    df["sleep_optimal"]    = df["sleep_hours"].between(6, 9).astype(int)
    df["effort_score"]     = (
        df["study_hours"] * 0.4
        + df["attendance_pct"] * 0.4
        + df["assignments_pct"] * 0.2
    ).round(2)
    df["support_index"]    = (
        df["parent_edu"] + df["internet_access"] * 2 + df["tutoring"] * 3
    )
    return df


def predict_student(student: dict, model_dir: Path = MODEL_DIR) -> dict:
    """
    Predict grade and score for a single student.

    Parameters
    ----------
    student : dict
        Keys must include all base feature columns (see preprocessing.py).

    Returns
    -------
    dict with keys:
        predicted_score   – float (0–100)
        predicted_grade   – str  (Fail / Average / Pass / Distinction)
        grade_probabilities – dict  {grade: probability}
    """
    clf, reg, le, feat_names = load_artifacts(model_dir)

    df = pd.DataFrame([student])
    df = _add_engineered_features(df)

    X = df[feat_names]

    score      = float(reg.predict(X)[0])
    score      = round(np.clip(score, 0, 100), 2)
    grade_enc  = clf.predict(X)[0]
    grade      = le.inverse_transform([grade_enc])[0]

    probas = {}
    if hasattr(clf, "predict_proba"):
        proba_arr = clf.predict_proba(X)[0]
        probas = {
            le.inverse_transform([i])[0]: round(float(p) * 100, 1)
            for i, p in enumerate(proba_arr)
        }

    return {
        "predicted_score":       score,
        "predicted_grade":       grade,
        "grade_probabilities_%": probas,
    }


def predict_batch(
    input_path: str,
    output_path: str = "predictions.csv",
    model_dir: Path = MODEL_DIR,
) -> pd.DataFrame:
    """
    Run predictions on a CSV file of student records.

    Parameters
    ----------
    input_path  : path to CSV (same columns as training data, without score/grade)
    output_path : where to write results

    Returns
    -------
    DataFrame with original columns + predicted_score + predicted_grade
    """
    clf, reg, le, feat_names = load_artifacts(model_dir)

    df = pd.read_csv(input_path)
    df = _add_engineered_features(df)

    X = df[feat_names]

    df["predicted_score"] = np.clip(reg.predict(X), 0, 100).round(2)
    df["predicted_grade"] = le.inverse_transform(clf.predict(X))

    df.to_csv(output_path, index=False)
    print(f"Batch predictions saved → {output_path}  ({len(df):,} rows)")
    return df


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example: high-achieving student
    high_achiever = {
        "study_hours":    25,
        "attendance_pct": 95,
        "prev_gpa":       3.8,
        "assignments_pct": 98,
        "sleep_hours":    8,
        "parent_edu":     4,
        "internet_access": 1,
        "extracurricular": 2,
        "travel_time":    1,
        "part_time_job":  0,
        "tutoring":       1,
        "gender":         0,
        "school_type":    1,
        "peer_group":     3,
        "absences":       1,
    }

    # Example: at-risk student
    at_risk = {
        "study_hours":    5,
        "attendance_pct": 58,
        "prev_gpa":       1.6,
        "assignments_pct": 42,
        "sleep_hours":    4,
        "parent_edu":     1,
        "internet_access": 0,
        "extracurricular": 0,
        "travel_time":    4,
        "part_time_job":  1,
        "tutoring":       0,
        "gender":         1,
        "school_type":    0,
        "peer_group":     1,
        "absences":       15,
    }

    print("\n── High Achiever ─────────────────────────────────")
    result = predict_student(high_achiever)
    print(f"  Score : {result['predicted_score']}")
    print(f"  Grade : {result['predicted_grade']}")
    print(f"  Proba : {result['grade_probabilities_%']}")

    print("\n── At-Risk Student ───────────────────────────────")
    result = predict_student(at_risk)
    print(f"  Score : {result['predicted_score']}")
    print(f"  Grade : {result['predicted_grade']}")
    print(f"  Proba : {result['grade_probabilities_%']}")

"""
main.py
-------
End-to-end pipeline runner.
  1. Generate dataset
  2. Train + tune all models
  3. Run evaluation & save plots
  4. Demo predictions
"""

import sys
from pathlib import Path

# ── Step 1: Generate data ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 1 — Generate Dataset")
print("="*60)
from data.generate_dataset import generate_student_data

df = generate_student_data(n_samples=1287, random_state=42)
df.to_csv("student_data.csv", index=False)
print(f"  Saved student_data.csv  ({len(df):,} rows)\n")
print(df["grade"].value_counts().to_string())


# ── Step 2: Train models ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 2 — Train & Tune Models (GridSearchCV)")
print("="*60)
from train_models import train_all

results = train_all("student_data.csv")


# ── Step 3: Evaluate & plot ───────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 3 — Evaluation & Visualizations")
print("="*60)
from evaluate import run_evaluation

run_evaluation("student_data.csv")


# ── Step 4: Demo inference ────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 4 — Demo Inference")
print("="*60)
from predict import predict_student

students = {
    "High Achiever": {
        "study_hours": 25, "attendance_pct": 95, "prev_gpa": 3.8,
        "assignments_pct": 98, "sleep_hours": 8, "parent_edu": 4,
        "internet_access": 1, "extracurricular": 2, "travel_time": 1,
        "part_time_job": 0, "tutoring": 1, "gender": 0,
        "school_type": 1, "peer_group": 3, "absences": 1,
    },
    "Average Student": {
        "study_hours": 14, "attendance_pct": 78, "prev_gpa": 2.7,
        "assignments_pct": 72, "sleep_hours": 7, "parent_edu": 2,
        "internet_access": 1, "extracurricular": 1, "travel_time": 2,
        "part_time_job": 0, "tutoring": 0, "gender": 1,
        "school_type": 0, "peer_group": 2, "absences": 6,
    },
    "At-Risk Student": {
        "study_hours": 5, "attendance_pct": 58, "prev_gpa": 1.6,
        "assignments_pct": 42, "sleep_hours": 4, "parent_edu": 1,
        "internet_access": 0, "extracurricular": 0, "travel_time": 4,
        "part_time_job": 1, "tutoring": 0, "gender": 1,
        "school_type": 0, "peer_group": 1, "absences": 15,
    },
}

for label, profile in students.items():
    res = predict_student(profile)
    print(f"\n  {label}")
    print(f"    Predicted Score : {res['predicted_score']}")
    print(f"    Predicted Grade : {res['predicted_grade']}")
    if res["grade_probabilities_%"]:
        proba_str = "  |  ".join(
            f"{g}: {p}%" for g, p in res["grade_probabilities_%"].items()
        )
        print(f"    Probabilities   : {proba_str}")

print("\n" + "="*60)
print("  Pipeline complete.")
print("  Models saved  → models/")
print("  Plots saved   → outputs/")
print("="*60 + "\n")

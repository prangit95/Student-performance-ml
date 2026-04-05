# Student Performance Prediction using Machine Learning

An end-to-end ML pipeline that predicts student academic outcomes
(grade classification + numeric score regression) from behavioral,
academic, and demographic features.

---

## Project Structure

```
student_performance/
├── data/
│   └── generate_dataset.py   # Synthetic dataset generator
├── models/                   # Saved model artifacts (auto-created)
├── outputs/                  # Evaluation plots (auto-created)
├── preprocessing.py          # Cleaning, feature engineering, train/test split
├── train_models.py           # GridSearchCV training for 6 classifiers + 3 regressors
├── evaluate.py               # Confusion matrix, ROC, feature importance, etc.
├── predict.py                # Inference for single student or batch CSV
├── main.py                   # Full end-to-end pipeline runner
└── requirements.txt
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (generate → train → evaluate → demo)
python main.py
```

---

## Features Used

| Feature | Type | Description |
|---|---|---|
| `study_hours` | numeric | Weekly study hours |
| `attendance_pct` | numeric | Class attendance % |
| `prev_gpa` | numeric | Previous GPA (1.0–4.0) |
| `assignments_pct` | numeric | Assignments completed % |
| `sleep_hours` | numeric | Avg. sleep per night |
| `parent_edu` | ordinal | Parental education (1–4) |
| `internet_access` | binary | Home internet access |
| `extracurricular` | ordinal | Activities per week (0–2) |
| `travel_time` | ordinal | Commute time (1–4) |
| `part_time_job` | binary | Has part-time job |
| `tutoring` | binary | Receives tutoring |
| `gender` | binary | 0=F, 1=M |
| `school_type` | binary | 0=public, 1=private |
| `peer_group` | ordinal | Academic motivation (1–3) |
| `absences` | numeric | Number of absences |

### Engineered Features (auto-added)

| Feature | Formula |
|---|---|
| `study_efficiency` | `assignments_pct / (study_hours + 1)` |
| `sleep_optimal` | 1 if 6 ≤ sleep ≤ 9 else 0 |
| `effort_score` | `0.4×study + 0.4×attendance + 0.2×assignments` |
| `support_index` | `parent_edu + 2×internet + 3×tutoring` |

---

## Models Trained

### Classification (predicts Fail / Average / Pass / Distinction)

| Model | Typical Accuracy |
|---|---|
| Random Forest ✦ | ~92–93% |
| Gradient Boosting | ~91% |
| SVM | ~88–89% |
| KNN | ~84% |
| Decision Tree | ~81% |
| Logistic Regression | ~78–80% |

### Regression (predicts numeric score 0–100)

| Model | Typical R² |
|---|---|
| Random Forest Regressor ✦ | ~0.93 |
| Gradient Boosting Regressor | ~0.91 |
| Ridge Regression | ~0.85 |

---

## Predicting New Students

```python
from predict import predict_student

student = {
    "study_hours": 20,
    "attendance_pct": 88,
    "prev_gpa": 3.2,
    "assignments_pct": 90,
    "sleep_hours": 7,
    "parent_edu": 3,
    "internet_access": 1,
    "extracurricular": 1,
    "travel_time": 2,
    "part_time_job": 0,
    "tutoring": 0,
    "gender": 0,
    "school_type": 0,
    "peer_group": 2,
    "absences": 3,
}

result = predict_student(student)
print(result)
# {'predicted_score': 79.4, 'predicted_grade': 'Pass', 'grade_probabilities_%': {...}}
```

### Batch prediction

```python
from predict import predict_batch

df = predict_batch("new_students.csv", output_path="predictions.csv")
```

---

## Outputs

After running `main.py`, the `outputs/` folder contains:

- `confusion_matrix.png` — per-class prediction accuracy
- `roc_curves.png` — one-vs-rest ROC with AUC scores
- `feature_importance.png` — which features matter most
- `actual_vs_predicted.png` — regression scatter plot
- `learning_curve.png` — bias-variance diagnosis
- `model_comparison.png` — accuracy bar chart across all models

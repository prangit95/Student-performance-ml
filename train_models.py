"""
train_models.py
---------------
Trains and compares multiple classifiers/regressors using:
  - 5-fold cross-validation
  - GridSearchCV hyper-parameter tuning
  - Full sklearn Pipeline (preprocess → model)
"""

import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error,
)

# Classifiers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Regressors
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

# Local
from preprocessing import run_preprocessing, make_preprocessor

import warnings
warnings.filterwarnings("ignore")


# ── Config ───────────────────────────────────────────────────────────────────

OUTPUT_DIR   = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

CV_FOLDS     = 5
RANDOM_STATE = 42
SCORING_CLF  = "accuracy"
SCORING_REG  = "r2"


# ── Classifier definitions + search grids ─────────────────────────────────────

CLF_REGISTRY = {
    "Random Forest": {
        "model": RandomForestClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "model__n_estimators":  [100, 200, 300],
            "model__max_depth":     [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__max_features":  ["sqrt", "log2"],
        },
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "model__n_estimators":  [100, 200],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__max_depth":     [3, 5],
            "model__subsample":     [0.8, 1.0],
        },
    },
    "SVM": {
        "model": SVC(probability=True, random_state=RANDOM_STATE),
        "param_grid": {
            "model__C":      [0.1, 1, 10, 100],
            "model__kernel": ["rbf", "linear"],
            "model__gamma":  ["scale", "auto"],
        },
    },
    "KNN": {
        "model": KNeighborsClassifier(),
        "param_grid": {
            "model__n_neighbors": [3, 5, 7, 11, 15],
            "model__weights":     ["uniform", "distance"],
            "model__metric":      ["euclidean", "manhattan"],
        },
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "model__max_depth":         [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__criterion":         ["gini", "entropy"],
        },
    },
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "param_grid": {
            "model__C":       [0.01, 0.1, 1, 10],
            "model__penalty": ["l2"],
            "model__solver":  ["lbfgs", "saga"],
        },
    },
}


# ── Regressor definitions ─────────────────────────────────────────────────────

REG_REGISTRY = {
    "Random Forest Regressor": {
        "model": RandomForestRegressor(random_state=RANDOM_STATE),
        "param_grid": {
            "model__n_estimators": [100, 200],
            "model__max_depth":    [None, 10, 20],
            "model__max_features": ["sqrt", 0.5],
        },
    },
    "Gradient Boosting Regressor": {
        "model": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "param_grid": {
            "model__n_estimators":  [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth":     [3, 5],
        },
    },
    "Ridge Regression": {
        "model": Ridge(),
        "param_grid": {
            "model__alpha": [0.1, 1.0, 10.0, 100.0],
        },
    },
}


# ── Training helpers ──────────────────────────────────────────────────────────

def build_pipeline(preprocessor, model) -> Pipeline:
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def train_with_grid_search(
    name: str,
    registry_entry: dict,
    X_train,
    y_train,
    cv: int,
    scoring: str,
) -> tuple[GridSearchCV, float]:
    """Fit GridSearchCV; return (best_estimator, wall_time_seconds)."""
    pipe = build_pipeline(make_preprocessor(), registry_entry["model"])
    grid = GridSearchCV(
        pipe,
        param_grid=registry_entry["param_grid"],
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0,
        refit=True,
        return_train_score=True,
    )
    t0 = time.time()
    grid.fit(X_train, y_train)
    elapsed = round(time.time() - t0, 1)
    return grid, elapsed


def evaluate_classifier(name, estimator, X_test, y_test, label_names):
    y_pred = estimator.predict(X_test)
    row = {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "F1":        round(f1_score(y_test, y_pred, average="weighted"), 4),
        "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, average="weighted"), 4),
    }
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_names)
    return row, cm, report


def evaluate_regressor(name, estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
    row = {
        "Model": name,
        "R²":    round(r2_score(y_test, y_pred), 4),
        "RMSE":  rmse,
        "MAE":   round(mean_absolute_error(y_test, y_pred), 4),
    }
    return row


# ── Main training loop ─────────────────────────────────────────────────────────

def train_all(csv_path: str = "student_data.csv"):
    print("=" * 60)
    print("  Student Performance Prediction — Model Training")
    print("=" * 60)

    splits, _, le = run_preprocessing(csv_path)
    X_train  = splits["X_train"]
    X_test   = splits["X_test"]
    yr_train = splits["y_reg_train"]
    yr_test  = splits["y_reg_test"]
    yc_train = splits["y_clf_train"]
    yc_test  = splits["y_clf_test"]
    label_names = list(le.classes_)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ── Classification ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  CLASSIFICATION  (predicting grade: Fail/Average/Pass/Distinction)")
    print(f"{'─'*60}")

    clf_results = []
    best_clf_name, best_clf_score, best_clf = None, 0, None

    for name, entry in CLF_REGISTRY.items():
        print(f"\n  [{name}]  searching {len(entry['param_grid'])} param groups …", end="", flush=True)
        grid, t = train_with_grid_search(name, entry, X_train, yc_train, cv, SCORING_CLF)
        row, cm, report = evaluate_classifier(name, grid.best_estimator_, X_test, yc_test, label_names)
        row["CV_Accuracy"] = round(grid.best_score_ * 100, 2)
        row["Best_Params"] = grid.best_params_
        row["Time_s"]      = t
        clf_results.append(row)
        print(f"  ✓  acc={row['Accuracy']}%  f1={row['F1']}  ({t}s)")

        if row["Accuracy"] > best_clf_score:
            best_clf_score = row["Accuracy"]
            best_clf_name  = name
            best_clf       = grid.best_estimator_
            best_cm        = cm
            best_report    = report

    clf_df = pd.DataFrame(clf_results).sort_values("Accuracy", ascending=False)

    print(f"\n  ✦  Best Classifier: {best_clf_name}  ({best_clf_score}% accuracy)")
    print(f"\n  Confusion Matrix ({best_clf_name}):\n")
    print(pd.DataFrame(best_cm, index=label_names, columns=label_names).to_string())
    print(f"\n  Classification Report:\n{best_report}")

    # ── Regression ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  REGRESSION  (predicting numeric score 0–100)")
    print(f"{'─'*60}")

    reg_results = []
    best_reg_name, best_reg_r2, best_reg = None, -np.inf, None

    for name, entry in REG_REGISTRY.items():
        print(f"\n  [{name}]  searching …", end="", flush=True)
        grid, t = train_with_grid_search(name, entry, X_train, yr_train, cv, SCORING_REG)
        row = evaluate_regressor(name, grid.best_estimator_, X_test, yr_test)
        row["CV_R²"]      = round(grid.best_score_, 4)
        row["Best_Params"] = grid.best_params_
        row["Time_s"]      = t
        reg_results.append(row)
        print(f"  ✓  R²={row['R²']}  RMSE={row['RMSE']}  ({t}s)")

        if row["R²"] > best_reg_r2:
            best_reg_r2  = row["R²"]
            best_reg_name = name
            best_reg     = grid.best_estimator_

    reg_df = pd.DataFrame(reg_results).sort_values("R²", ascending=False)
    print(f"\n  ✦  Best Regressor: {best_reg_name}  (R²={best_reg_r2})")

    # ── Feature importance (best classifier, if RF/GB) ──────────────────────────
    print(f"\n{'─'*60}")
    print("  FEATURE IMPORTANCE")
    print(f"{'─'*60}")

    inner_model = best_clf.named_steps["model"]
    if hasattr(inner_model, "feature_importances_"):
        importances = inner_model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature":    splits["feature_names"],
            "Importance": importances,
        }).sort_values("Importance", ascending=False)
        print("\n" + feat_df.to_string(index=False))
    else:
        feat_df = pd.DataFrame()
        print("  (feature importances not available for this model type)")

    # ── Save artifacts ─────────────────────────────────────────────────────────
    joblib.dump(best_clf,  OUTPUT_DIR / "best_classifier.pkl")
    joblib.dump(best_reg,  OUTPUT_DIR / "best_regressor.pkl")
    joblib.dump(le,         OUTPUT_DIR / "label_encoder.pkl")
    joblib.dump(splits["feature_names"], OUTPUT_DIR / "feature_names.pkl")
    clf_df.to_csv(OUTPUT_DIR / "classifier_results.csv", index=False)
    reg_df.to_csv(OUTPUT_DIR / "regressor_results.csv", index=False)
    if not feat_df.empty:
        feat_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    print(f"\n  Artifacts saved to → {OUTPUT_DIR}/")
    print(f"    best_classifier.pkl   ({best_clf_name})")
    print(f"    best_regressor.pkl    ({best_reg_name})")
    print(f"    label_encoder.pkl")
    print(f"    classifier_results.csv")
    print(f"    regressor_results.csv")
    print(f"    feature_importance.csv")

    return {
        "best_clf":      best_clf,
        "best_reg":      best_reg,
        "clf_results":   clf_df,
        "reg_results":   reg_df,
        "feature_names": splits["feature_names"],
        "label_encoder": le,
    }


if __name__ == "__main__":
    results = train_all("student_data.csv")
    print("\n  Done.")

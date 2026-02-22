import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
)

import matplotlib.pyplot as plt


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # drop any obvious non-feature columns (adjust if needed)
    drop_cols = [c for c in ["house_id"] if c in df.columns]
    df["is_landslide"] = df.apply(lambda row: 1 if row["num_landslides_100m"] > 0 else 0, axis=1)
    df = df.dropna(subset=["is_landslide"])  # make sure target exists

    print("Loaded dataframe shape:", df.shape)
    print("Label distribution:\n", df["is_landslide"].value_counts())
    return df


def get_X_y(df: pd.DataFrame):
    target_col = "is_landslide"

    feature_cols = [
        c
        for c in df.columns
        if c not in ["house_id", target_col, "num_landslides_100m"]  # drop ID and raw landslide count
    ]

    X = df[feature_cols].copy()
    y = df[target_col].astype(int)

    print("\nUsing features:")
    print(feature_cols)
    return X, y, feature_cols


def train_logreg(X, y):
    print("\n=== Logistic Regression ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42,
    )

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",  # helpful if classes are imbalanced
        ),
    )

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    print("\nClassification report (test):")
    print(classification_report(y_test, y_pred, digits=3))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (test):\n", cm)

    auc = roc_auc_score(y_test, y_prob)
    print("ROC AUC (test):", round(auc, 3))

    # simple cross-validated AUC (with all data)
    cv = StratifiedKFold(n_splits=min(5, len(np.unique(y)) + 1), shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print("Cross-validated ROC AUC:", np.round(cv_scores, 3))
    print("Mean AUC:", round(cv_scores.mean(), 3))

    # plot ROC curve
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("Logistic Regression ROC (test)")
    plt.tight_layout()
    out_path = Path("rerun/figures_2/logreg_roc.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved ROC curve to {out_path}")

    return pipe


def train_random_forest(X, y, feature_cols):
    print("\n=== Random Forest ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42,
    )

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    print("\nClassification report (test):")
    print(classification_report(y_test, y_pred, digits=3))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (test):\n", cm)

    auc = roc_auc_score(y_test, y_prob)
    print("ROC AUC (test):", round(auc, 3))

    # cross-validated AUC
    cv = StratifiedKFold(n_splits=min(5, len(np.unique(y)) + 1), shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
    print("Cross-validated ROC AUC:", np.round(cv_scores, 3))
    print("Mean AUC:", round(cv_scores.mean(), 3))

    # feature importance
    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    print("\nTop feature importances:")
    print(importances.head(10))

    # save to csv
    fi_path = Path("rerun/results_2/rf_feature_importances.csv")
    fi_path.parent.mkdir(parents=True, exist_ok=True)
    importances.to_csv(fi_path, header=["importance"])
    print(f"Saved feature importances to {fi_path}")

    # plot ROC
    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title("Random Forest ROC (test)")
    plt.tight_layout()
    out_path = Path("rerun/figures_2/rf_roc.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved ROC curve to {out_path}")

    return rf
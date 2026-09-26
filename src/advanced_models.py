import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("ADVANCED MODEL TRAINING")
print("=" * 70)

df = pd.read_csv(
    "data/processed/placement_cleaned.csv"
)

print("Dataset shape:", df.shape)


# ============================================================
# 2. PREPARE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=["PlacementStatus"]
)

y = df["PlacementStatus"].map({
    "NotPlaced": 0,
    "Placed": 1
})


# ============================================================
# 3. ENCODE CATEGORICAL FEATURES
# ============================================================

X["ExtracurricularActivities"] = X[
    "ExtracurricularActivities"
].map({
    "No": 0,
    "Yes": 1
})

X["PlacementTraining"] = X[
    "PlacementTraining"
].map({
    "No": 0,
    "Yes": 1
})


# ============================================================
# 4. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 5. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# 6. TRAIN MODELS
# ============================================================

results = []

trained_models = {}


for name, model in models.items():

    print("\n" + "=" * 70)
    print("TRAINING:", name)
    print("=" * 70)

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    })

    trained_models[name] = model


# ============================================================
# 7. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 8. BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

best_accuracy = results_df.iloc[0]["Accuracy"]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Model:", best_model_name)
print(
    f"Accuracy: {best_accuracy * 100:.2f}%"
)


# ============================================================
# 9. BEST MODEL PREDICTIONS
# ============================================================

best_predictions = best_model.predict(
    X_test
)

best_probabilities = best_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    best_predictions
)

print(cm)


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        best_predictions,
        target_names=[
            "NotPlaced",
            "Placed"
        ]
    )
)


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

if hasattr(
    best_model,
    "feature_importances_"
):

    importance = pd.DataFrame({

        "Feature": X.columns,

        "Importance":
            best_model.feature_importances_

    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    print(
        importance.to_string(
            index=False
        )
    )


# ============================================================
# 13. SAVE RESULTS
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

results_df.to_csv(
    "models/advanced_model_comparison.csv",
    index=False
)


# ============================================================
# 14. SAVE BEST MODEL
# ============================================================

import joblib

joblib.dump(
    best_model,
    "models/final_placement_model.pkl"
)

joblib.dump(
    list(X.columns),
    "models/final_feature_columns.pkl"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("ADVANCED MODEL TRAINING COMPLETED")
print("=" * 70)

print(
    "Best model:",
    best_model_name
)

print(
    f"Best accuracy: {best_accuracy * 100:.2f}%"
)

print("\nSaved:")
print("- models/advanced_model_comparison.csv")
print("- models/final_placement_model.pkl")
print("- models/final_feature_columns.pkl")

print("\nNEXT STEP:")
print("EXPLAINABLE ML - SHAP")
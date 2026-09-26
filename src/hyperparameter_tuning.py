import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/processed/placement_cleaned.csv"
)

print("=" * 70)
print("DATASET LOADED")
print("=" * 70)

print("Shape:", df.shape)


# ============================================================
# 2. FEATURES AND TARGET
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


print("\n" + "=" * 70)
print("FEATURES")
print("=" * 70)

print(X.columns.tolist())


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
# 5. MODELS AND PARAMETER SEARCH
# ============================================================

models = {

    "Random Forest": (

        RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        ),

        {
            "n_estimators": [200, 300, 500, 700],
            "max_depth": [None, 5, 8, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None]
        }
    ),


    "Gradient Boosting": (

        GradientBoostingClassifier(
            random_state=42
        ),

        {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "max_depth": [2, 3, 4, 5],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "subsample": [0.8, 1.0]
        }
    ),


    "Extra Trees": (

        ExtraTreesClassifier(
            random_state=42,
            n_jobs=-1
        ),

        {
            "n_estimators": [200, 300, 500, 700],
            "max_depth": [None, 5, 8, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None]
        }
    ),


    "Hist Gradient Boosting": (

        HistGradientBoostingClassifier(
            random_state=42
        ),

        {
            "max_iter": [100, 200, 300, 500],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "max_depth": [None, 3, 5, 8, 10],
            "min_samples_leaf": [10, 20, 30, 50],
            "l2_regularization": [0, 0.1, 1.0]
        }
    )
}


# ============================================================
# 6. STORE RESULTS
# ============================================================

results = []

best_model = None
best_model_name = None
best_accuracy = 0


# ============================================================
# 7. HYPERPARAMETER TUNING
# ============================================================

for name, (model, parameters) in models.items():

    print("\n")
    print("=" * 70)
    print("TUNING:", name)
    print("=" * 70)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=parameters,
        n_iter=20,
        scoring="accuracy",
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    search.fit(
        X_train,
        y_train
    )

    best_estimator = search.best_estimator_

    predictions = best_estimator.predict(
        X_test
    )

    probabilities = best_estimator.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )


    print("\nBest Parameters:")
    print(search.best_params_)

    print("\nCross-validation accuracy:")
    print(round(search.best_score_, 4))

    print("\nTest Results:")

    print(
        "Accuracy :",
        round(accuracy, 4)
    )

    print(
        "Precision:",
        round(precision, 4)
    )

    print(
        "Recall   :",
        round(recall, 4)
    )

    print(
        "F1-Score :",
        round(f1, 4)
    )

    print(
        "ROC-AUC  :",
        round(roc_auc, 4)
    )


    results.append({

        "Model": name,

        "CV_Accuracy":
            search.best_score_,

        "Test_Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1-Score":
            f1,

        "ROC-AUC":
            roc_auc
    })


    # Select best based on TEST accuracy
    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = best_estimator

        best_model_name = name


# ============================================================
# 8. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

print("\n")
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 9. BEST MODEL
# ============================================================

print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    "Model:",
    best_model_name
)

print(
    "Accuracy:",
    round(best_accuracy, 4)
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

best_predictions = best_model.predict(
    X_test
)

cm = confusion_matrix(
    y_test,
    best_predictions
)

print("\n")
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
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

    print("\n")
    print("=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    print(
        importance.to_string(
            index=False
        )
    )

    importance.to_csv(
        "models/feature_importance.csv",
        index=False
    )


# ============================================================
# 13. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "models/tuned_model_comparison.csv",
    index=False
)


# ============================================================
# 14. SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    "models/best_placement_model.pkl"
)

joblib.dump(
    list(X.columns),
    "models/best_feature_columns.pkl"
)


# ============================================================
# 15. COMPLETION
# ============================================================

print("\n")
print("=" * 70)
print("HYPERPARAMETER TUNING COMPLETED")
print("=" * 70)

print(
    "Best model:",
    best_model_name
)

print(
    "Best test accuracy:",
    round(best_accuracy * 100, 2),
    "%"
)

print("\nSaved files:")

print(
    "- models/tuned_model_comparison.csv"
)

print(
    "- models/best_placement_model.pkl"
)

print(
    "- models/best_feature_columns.pkl"
)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import joblib


# ============================================================
# 1. LOAD FEATURE-ENGINEERED DATASET
# ============================================================

input_path = "data/processed/placement_feature_engineered.csv"

df = pd.read_csv(input_path)

print("=" * 70)
print("FEATURE-ENGINEERED DATASET")
print("=" * 70)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["PlacementStatus"])
y = df["PlacementStatus"]


print("\n" + "=" * 70)
print("FEATURES AND TARGET")
print("=" * 70)

print("Number of features:", X.shape[1])
print("Features:")

for column in X.columns:
    print("-", column)

print("\nTarget:")
print("PlacementStatus")


# ============================================================
# 3. ENCODE CATEGORICAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("ENCODING CATEGORICAL FEATURES")
print("=" * 70)

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
# 4. ENCODE TARGET
# ============================================================

y = y.map({
    "NotPlaced": 0,
    "Placed": 1
})

print("NotPlaced -> 0")
print("Placed    -> 1")

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# ============================================================
# 6. SCALE FEATURES FOR LOGISTIC REGRESSION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 7. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}


# ============================================================
# 8. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

best_model = None
best_model_name = None
best_f1 = 0

print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)


for name, model in models.items():

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    # Logistic Regression uses scaled data
    if name == "Logistic Regression":

        model.fit(
            X_train_scaled,
            y_train
        )

        predictions = model.predict(
            X_test_scaled
        )

        probabilities = model.predict_proba(
            X_test_scaled
        )[:, 1]

    else:

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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


    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))


    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    })


    # --------------------------------------------------------
    # SELECT BEST MODEL
    # --------------------------------------------------------

    if f1 > best_f1:

        best_f1 = f1
        best_model = model
        best_model_name = name


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 10. BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Model:", best_model_name)
print("Best F1-Score:", round(best_f1, 4))


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

if best_model_name == "Logistic Regression":

    best_predictions = best_model.predict(
        X_test_scaled
    )

else:

    best_predictions = best_model.predict(
        X_test
    )


cm = confusion_matrix(
    y_test,
    best_predictions
)


print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# 12. CLASSIFICATION REPORT
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
# 13. SAVE RESULTS
# ============================================================

results_path = "models/feature_engineered_model_comparison.csv"

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# 14. SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    "models/feature_engineered_model.pkl"
)

joblib.dump(
    scaler,
    "models/feature_engineered_scaler.pkl"
)

joblib.dump(
    list(X.columns),
    "models/feature_engineered_columns.pkl"
)


# ============================================================
# 15. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE-ENGINEERED TRAINING COMPLETED")
print("=" * 70)

print("Best Model:", best_model_name)

print(
    "Best F1-Score:",
    round(best_f1, 4)
)

print("\nSaved files:")

print("-", results_path)
print("- models/feature_engineered_model.pkl")
print("- models/feature_engineered_scaler.pkl")
print("- models/feature_engineered_columns.pkl")
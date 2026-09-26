import os
import joblib
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


# ============================================================
# 1. LOAD DATASET
# ============================================================

input_path = "data/processed/placement_cleaned.csv"

df = pd.read_csv(input_path)

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["PlacementStatus"])
y = df["PlacementStatus"]

print("\n" + "=" * 60)
print("FEATURES AND TARGET")
print("=" * 60)

print("Features:")
print(X.columns.tolist())

print("\nTarget:")
print("PlacementStatus")


# ============================================================
# 3. ENCODE CATEGORICAL FEATURES
# ============================================================

print("\n" + "=" * 60)
print("ENCODING CATEGORICAL FEATURES")
print("=" * 60)

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

print("ExtracurricularActivities:")
print(X["ExtracurricularActivities"].value_counts())

print("\nPlacementTraining:")
print(X["PlacementTraining"].value_counts())


# ============================================================
# 4. ENCODE TARGET
# ============================================================

print("\n" + "=" * 60)
print("ENCODING TARGET")
print("=" * 60)

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

print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# ============================================================
# 6. FEATURE SCALING
# ============================================================

print("\n" + "=" * 60)
print("FEATURE SCALING")
print("=" * 60)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaling method: StandardScaler")
print("Scaled training shape:", X_train_scaled.shape)
print("Scaled testing shape :", X_test_scaled.shape)


# ============================================================
# 7. DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=10
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}


# ============================================================
# 8. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

trained_models = {}

print("\n" + "=" * 60)
print("MODEL TRAINING STARTED")
print("=" * 60)


for model_name, model in models.items():

    print("\n" + "-" * 60)
    print("Training:", model_name)
    print("-" * 60)

    # Tree-based models don't require scaling.
    # Logistic Regression benefits from scaling.

    if model_name == "Logistic Regression":

        model.fit(
            X_train_scaled,
            y_train
        )

        y_pred = model.predict(
            X_test_scaled
        )

        y_probability = model.predict_proba(
            X_test_scaled
        )[:, 1]

    else:

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_test
        )

        y_probability = model.predict_proba(
            X_test
        )[:, 1]


    # --------------------------------------------------------
    # Evaluation metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    })

    trained_models[model_name] = model


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 10. SELECT BEST MODEL
# ============================================================

best_row = results_df.loc[
    results_df["F1-Score"].idxmax()
]

best_model_name = best_row["Model"]

best_model = trained_models[
    best_model_name
]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Selected model:", best_model_name)

print(
    f"Accuracy : {best_row['Accuracy']:.4f}"
)

print(
    f"Precision: {best_row['Precision']:.4f}"
)

print(
    f"Recall   : {best_row['Recall']:.4f}"
)

print(
    f"F1-Score : {best_row['F1-Score']:.4f}"
)

print(
    f"ROC-AUC  : {best_row['ROC-AUC']:.4f}"
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

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

print(cm)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        best_predictions,
        target_names=[
            "NotPlaced",
            "Placed"
        ],
        zero_division=0
    )
)


# ============================================================
# 13. CREATE MODELS DIRECTORY
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# 14. SAVE BEST MODEL + SCALER + FEATURE INFORMATION
# ============================================================

model_path = "models/placement_model.pkl"
scaler_path = "models/scaler.pkl"
features_path = "models/feature_columns.pkl"


joblib.dump(
    best_model,
    model_path
)

joblib.dump(
    scaler,
    scaler_path
)

joblib.dump(
    X.columns.tolist(),
    features_path
)


# ============================================================
# 15. SAVE MODEL COMPARISON RESULTS
# ============================================================

results_path = "models/model_comparison.csv"

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)

print("Best model:", best_model_name)

print("\nSaved files:")

print("-", model_path)
print("-", scaler_path)
print("-", features_path)
print("-", results_path)

print("\nProject is ready for the evaluation and prediction stages.")
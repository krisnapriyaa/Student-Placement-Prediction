import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split


print("=" * 70)
print("EXPLAINABLE ML - SHAP ANALYSIS")
print("=" * 70)


# ---------------------------------------------------------
# STEP 1: LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(
    "data/processed/placement_cleaned.csv"
)

print("\nDataset shape:", df.shape)


# ---------------------------------------------------------
# STEP 2: SEPARATE FEATURES AND TARGET
# ---------------------------------------------------------

X = df.drop(
    columns=["PlacementStatus"]
)

y = df["PlacementStatus"].map({
    "NotPlaced": 0,
    "Placed": 1
})


# ---------------------------------------------------------
# STEP 3: ENCODE CATEGORICAL FEATURES
# ---------------------------------------------------------

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


print("\nFeatures used by the model:")

for column in X.columns:
    print("-", column)


# ---------------------------------------------------------
# STEP 4: SAME TRAIN-TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# ---------------------------------------------------------
# STEP 5: LOAD FINAL MODEL
# ---------------------------------------------------------

model_path = "models/final_placement_model.pkl"

model = joblib.load(model_path)

print("\nFinal model loaded:")
print(type(model).__name__)


# ---------------------------------------------------------
# STEP 6: CREATE SHAP EXPLAINER
# ---------------------------------------------------------

print("\nCreating SHAP explainer...")

explainer = shap.Explainer(
    model,
    X_train
)


# ---------------------------------------------------------
# STEP 7: CALCULATE SHAP VALUES
# ---------------------------------------------------------

print("Calculating SHAP values...")

shap_values = explainer(
    X_test
)


print("SHAP calculation completed.")


# ---------------------------------------------------------
# STEP 8: CREATE OUTPUT DIRECTORY
# ---------------------------------------------------------

os.makedirs(
    "outputs/shap",
    exist_ok=True
)


# ---------------------------------------------------------
# STEP 9: SHAP SUMMARY PLOT
# ---------------------------------------------------------

print("\nCreating SHAP summary plot...")

plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.tight_layout()

plt.savefig(
    "outputs/shap/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# STEP 10: SHAP BAR PLOT
# ---------------------------------------------------------

print("Creating SHAP feature importance plot...")

plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    "outputs/shap/shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# STEP 11: CALCULATE MEAN ABSOLUTE SHAP VALUES
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SHAP FEATURE IMPORTANCE")
print("=" * 70)


mean_abs_shap = np.abs(
    shap_values.values
).mean(axis=0)


importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Mean_Absolute_SHAP": mean_abs_shap
})


importance_df = importance_df.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
)


print(
    importance_df.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# STEP 12: SAVE SHAP IMPORTANCE
# ---------------------------------------------------------

importance_df.to_csv(
    "outputs/shap/shap_feature_importance.csv",
    index=False
)


# ---------------------------------------------------------
# STEP 13: EXPLAIN ONE STUDENT
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE STUDENT EXPLANATION")
print("=" * 70)


sample_index = 0

sample = X_test.iloc[
    sample_index:sample_index + 1
]

prediction = model.predict(
    sample
)[0]

probability = model.predict_proba(
    sample
)[0][1]


print("\nPrediction:")

if prediction == 1:
    print("PLACED")
else:
    print("NOT PLACED")


print(
    f"Placement probability: {probability * 100:.2f}%"
)


print("\nStudent features:")

print(
    sample.to_string(
        index=False
    )
)


print("\nSHAP contributions:")

sample_shap = shap_values.values[
    sample_index
]


local_explanation = pd.DataFrame({
    "Feature": X.columns,
    "SHAP_Value": sample_shap
})


local_explanation[
    "Absolute_SHAP"
] = np.abs(
    local_explanation["SHAP_Value"]
)


local_explanation = local_explanation.sort_values(
    by="Absolute_SHAP",
    ascending=False
)


print(
    local_explanation[
        [
            "Feature",
            "SHAP_Value"
        ]
    ].to_string(
        index=False
    )
)


# ---------------------------------------------------------
# STEP 14: COMPLETED
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SHAP EXPLAINABILITY COMPLETED")
print("=" * 70)

print("\nGenerated files:")

print(
    "- outputs/shap/shap_summary.png"
)

print(
    "- outputs/shap/shap_feature_importance.png"
)

print(
    "- outputs/shap/shap_feature_importance.csv"
)

print("\nNEXT STEP:")
print("NEW STUDENT PREDICTION")
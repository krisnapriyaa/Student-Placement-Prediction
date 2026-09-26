import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. LOAD CLEANED DATASET
# ============================================================

df = pd.read_csv("data/processed/placement_cleaned.csv")

print("=" * 60)
print("CLEANED DATASET LOADED")
print("=" * 60)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["PlacementStatus"])
y = df["PlacementStatus"]


print("\n" + "=" * 60)
print("FEATURES (X)")
print("=" * 60)

print(X.columns.tolist())


print("\n" + "=" * 60)
print("TARGET (y)")
print("=" * 60)

print("PlacementStatus")


# ============================================================
# 3. ENCODE CATEGORICAL FEATURES
# ============================================================

print("\n" + "=" * 60)
print("ENCODING CATEGORICAL FEATURES")
print("=" * 60)

# Yes / No → 1 / 0

X["ExtracurricularActivities"] = (
    X["ExtracurricularActivities"].map({
        "No": 0,
        "Yes": 1
    })
)

X["PlacementTraining"] = (
    X["PlacementTraining"].map({
        "No": 0,
        "Yes": 1
    })
)

print("ExtracurricularActivities encoded:")
print(X["ExtracurricularActivities"].value_counts())

print("\nPlacementTraining encoded:")
print(X["PlacementTraining"].value_counts())


# ============================================================
# 4. ENCODE TARGET
# ============================================================

y = y.map({
    "NotPlaced": 0,
    "Placed": 1
})

print("\n" + "=" * 60)
print("TARGET ENCODING")
print("=" * 60)

print("NotPlaced → 0")
print("Placed    → 1")

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


print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# ============================================================
# 6. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("\n" + "=" * 60)
print("FEATURE SCALING")
print("=" * 60)

print("Scaling method: StandardScaler")

print("\nScaled training data shape:")
print(X_train_scaled.shape)

print("\nScaled testing data shape:")
print(X_test_scaled.shape)


# ============================================================
# 7. FINAL CHECK
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

print("\nFirst 5 training samples after scaling:")
print(X_train_scaled[:5])
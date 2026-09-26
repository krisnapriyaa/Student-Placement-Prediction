
import pandas as pd


# ============================================================
# LOAD RAW DATASET
# ============================================================

input_path = "data/raw/placementdata.csv"

df = pd.read_csv(input_path)

print("=" * 60)
print("RAW DATASET")
print("=" * 60)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values)


# ============================================================
# CHECK EXACT DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)

if duplicate_count == 0:
    print("No exact duplicate records found.")
else:
    print("Exact duplicate records found.")


# ============================================================
# CHECK CATEGORICAL VALUES
# ============================================================

print("\n" + "=" * 60)
print("CATEGORICAL VALUES")
print("=" * 60)


print("\nExtracurricularActivities:")
print(df["ExtracurricularActivities"].value_counts())


print("\nPlacementTraining:")
print(df["PlacementTraining"].value_counts())


print("\nPlacementStatus:")
print(df["PlacementStatus"].value_counts())


# ============================================================
# CHECK NUMERICAL RANGES
# ============================================================

print("\n" + "=" * 60)
print("NUMERICAL RANGES")
print("=" * 60)

numeric_columns = [
    "CGPA",
    "Internships",
    "Projects",
    "Workshops/Certifications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "SSC_Marks",
    "HSC_Marks"
]

for column in numeric_columns:

    print(
        f"{column}: "
        f"Min = {df[column].min()}, "
        f"Max = {df[column].max()}"
    )


# ============================================================
# REMOVE STUDENT ID
# ============================================================

print("\n" + "=" * 60)
print("REMOVING STUDENT ID")
print("=" * 60)

df = df.drop(columns=["StudentID"])

print("StudentID removed.")

print("New shape:", df.shape)


# ============================================================
# CHECK DUPLICATE FEATURE COMBINATIONS
# ============================================================

print("\n" + "=" * 60)
print("CHECKING DUPLICATE FEATURE ROWS")
print("=" * 60)

duplicate_feature_count = df.duplicated().sum()

print(
    "Identical feature rows:",
    duplicate_feature_count
)

print(
    "These rows are retained because they may represent "
    "different students with identical recorded characteristics."
)


# ============================================================
# FINAL MISSING VALUE CHECK
# ============================================================

print("\n" + "=" * 60)
print("FINAL CLEANING CHECK")
print("=" * 60)

print("Missing values:")

print(df.isnull().sum())


# ============================================================
# FINAL COLUMNS
# ============================================================

print("\nFinal columns:")

print(df.columns.tolist())


# ============================================================
# FINAL DATASET SHAPE
# ============================================================

print("\nFinal shape:", df.shape)


# ============================================================
# SAVE CLEANED DATASET
# ============================================================

output_path = "data/processed/placement_cleaned.csv"

df.to_csv(output_path, index=False)


print("\n" + "=" * 60)
print("CLEANED DATASET SAVED")
print("=" * 60)

print(output_path)

print("Final shape:", df.shape)

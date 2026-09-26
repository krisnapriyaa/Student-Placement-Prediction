import pandas as pd

# -----------------------------------
# 1. Load Dataset
# -----------------------------------

df = pd.read_csv("data/raw/placementdata.csv")


# -----------------------------------
# 2. Basic Dataset Information
# -----------------------------------

print("=" * 50)
print("DATASET SHAPE")
print("=" * 50)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# -----------------------------------
# 3. First 5 Rows
# -----------------------------------

print("\n" + "=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)

print(df.head())


# -----------------------------------
# 4. Column Names
# -----------------------------------

print("\n" + "=" * 50)
print("COLUMN NAMES")
print("=" * 50)

print(df.columns.tolist())


# -----------------------------------
# 5. Data Types
# -----------------------------------

print("\n" + "=" * 50)
print("DATA TYPES")
print("=" * 50)

print(df.dtypes)


# -----------------------------------
# 6. Dataset Information
# -----------------------------------

print("\n" + "=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

df.info()


# -----------------------------------
# 7. Statistical Summary
# -----------------------------------

print("\n" + "=" * 50)
print("STATISTICAL SUMMARY")
print("=" * 50)

print(df.describe())


# -----------------------------------
# 8. Missing Values
# -----------------------------------

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)

print(df.isnull().sum())


# -----------------------------------
# 9. Duplicate Rows
# -----------------------------------

print("\n" + "=" * 50)
print("DUPLICATE ROWS")
print("=" * 50)

print(df.duplicated().sum())


# -----------------------------------
# 10. Target Distribution
# -----------------------------------

print("\n" + "=" * 50)
print("PLACEMENT STATUS DISTRIBUTION")
print("=" * 50)

print(df["PlacementStatus"].value_counts())


# -----------------------------------
# 11. Unique Values
# -----------------------------------

print("\n" + "=" * 50)
print("UNIQUE VALUES")
print("=" * 50)

for column in df.columns:
    print(f"\n{column}:")
    print(df[column].unique())
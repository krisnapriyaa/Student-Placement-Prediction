import pandas as pd


# ============================================================
# LOAD CLEANED DATA
# ============================================================

input_path = "data/processed/placement_cleaned.csv"

df = pd.read_csv(input_path)

print("=" * 60)
print("CLEANED DATASET LOADED")
print("=" * 60)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# FEATURE 1: ACADEMIC AVERAGE
# ============================================================

df["AcademicAverage"] = (
    df["SSC_Marks"] + df["HSC_Marks"]
) / 2


# ============================================================
# FEATURE 2: SOFT SKILLS PERCENTAGE
# ============================================================

df["SoftSkillsPercentage"] = (
    df["SoftSkillsRating"] / 5
) * 100


# ============================================================
# FEATURE 3: SKILL SCORE
# ============================================================

df["SkillScore"] = (
    df["AptitudeTestScore"] +
    df["SoftSkillsPercentage"]
) / 2


# ============================================================
# FEATURE 4: EXPERIENCE SCORE
# ============================================================

df["ExperienceScore"] = (
    df["Internships"] +
    df["Projects"] +
    df["Workshops/Certifications"]
)


# ============================================================
# DISPLAY NEW FEATURES
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)

new_features = [
    "AcademicAverage",
    "SoftSkillsPercentage",
    "SkillScore",
    "ExperienceScore"
]

print("\nNew features created:")

for feature in new_features:
    print("-", feature)


# ============================================================
# DISPLAY SAMPLE
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE OF ENGINEERED FEATURES")
print("=" * 60)

print(
    df[
        [
            "SSC_Marks",
            "HSC_Marks",
            "AcademicAverage",
            "AptitudeTestScore",
            "SoftSkillsRating",
            "SoftSkillsPercentage",
            "SkillScore",
            "Internships",
            "Projects",
            "Workshops/Certifications",
            "ExperienceScore"
        ]
    ].head()
)


# ============================================================
# SAVE FEATURE-ENGINEERED DATASET
# ============================================================

output_path = "data/processed/placement_feature_engineered.csv"

df.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("FEATURE-ENGINEERED DATASET SAVED")
print("=" * 60)

print(output_path)
print("Final shape:", df.shape)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# 1. LOAD CLEANED DATASET
# ============================================================

df = pd.read_csv("data/processed/placement_cleaned.csv")

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("PLACEMENT STATUS DISTRIBUTION")
print("=" * 60)

print(df["PlacementStatus"].value_counts())

print("\nPercentage distribution:")
print(df["PlacementStatus"].value_counts(normalize=True) * 100)


# ============================================================
# 3. CREATE EDA OUTPUT FOLDER
# ============================================================

os.makedirs("outputs/eda", exist_ok=True)


# ============================================================
# 4. PLACEMENT STATUS COUNT PLOT
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="PlacementStatus"
)

plt.title("Placement Status Distribution")
plt.xlabel("Placement Status")
plt.ylabel("Number of Students")

plt.tight_layout()

plt.savefig("outputs/eda/placement_distribution.png")

plt.show()


# ============================================================
# 5. CGPA VS PLACEMENT
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="PlacementStatus",
    y="CGPA"
)

plt.title("CGPA vs Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("CGPA")

plt.tight_layout()

plt.savefig("outputs/eda/cgpa_vs_placement.png")

plt.show()


# ============================================================
# 6. INTERNSHIPS VS PLACEMENT
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Internships",
    hue="PlacementStatus"
)

plt.title("Internships vs Placement Status")
plt.xlabel("Number of Internships")
plt.ylabel("Number of Students")

plt.tight_layout()

plt.savefig("outputs/eda/internships_vs_placement.png")

plt.show()


# ============================================================
# 7. PROJECTS VS PLACEMENT
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Projects",
    hue="PlacementStatus"
)

plt.title("Projects vs Placement Status")
plt.xlabel("Number of Projects")
plt.ylabel("Number of Students")

plt.tight_layout()

plt.savefig("outputs/eda/projects_vs_placement.png")

plt.show()


# ============================================================
# 8. WORKSHOPS / CERTIFICATIONS VS PLACEMENT
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Workshops/Certifications",
    hue="PlacementStatus"
)

plt.title("Workshops / Certifications vs Placement Status")
plt.xlabel("Number of Workshops / Certifications")
plt.ylabel("Number of Students")

plt.tight_layout()

plt.savefig("outputs/eda/workshops_vs_placement.png")

plt.show()


# ============================================================
# 9. APTITUDE TEST SCORE VS PLACEMENT
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="PlacementStatus",
    y="AptitudeTestScore"
)

plt.title("Aptitude Test Score vs Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("Aptitude Test Score")

plt.tight_layout()

plt.savefig("outputs/eda/aptitude_vs_placement.png")

plt.show()


# ============================================================
# 10. SOFT SKILLS VS PLACEMENT
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="PlacementStatus",
    y="SoftSkillsRating"
)

plt.title("Soft Skills Rating vs Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("Soft Skills Rating")

plt.tight_layout()

plt.savefig("outputs/eda/softskills_vs_placement.png")

plt.show()


# ============================================================
# 11. EXTRACURRICULAR ACTIVITIES VS PLACEMENT
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="ExtracurricularActivities",
    hue="PlacementStatus"
)

plt.title("Extracurricular Activities vs Placement Status")
plt.xlabel("Extracurricular Activities")
plt.ylabel("Number of Students")

plt.tight_layout()

plt.savefig("outputs/eda/extracurricular_vs_placement.png")

plt.show()


# ============================================================
# 12. PLACEMENT TRAINING VS PLACEMENT
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="PlacementTraining",
    hue="PlacementStatus"
)

plt.title("Placement Training vs Placement Status")
plt.xlabel("Placement Training")
plt.ylabel("Number of Students")

plt.tight_layout()

plt.savefig("outputs/eda/training_vs_placement.png")

plt.show()


# ============================================================
# 13. SSC MARKS VS PLACEMENT
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="PlacementStatus",
    y="SSC_Marks"
)

plt.title("SSC Marks vs Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("SSC Marks")

plt.tight_layout()

plt.savefig("outputs/eda/ssc_vs_placement.png")

plt.show()


# ============================================================
# 14. HSC MARKS VS PLACEMENT
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="PlacementStatus",
    y="HSC_Marks"
)

plt.title("HSC Marks vs Placement Status")
plt.xlabel("Placement Status")
plt.ylabel("HSC Marks")

plt.tight_layout()

plt.savefig("outputs/eda/hsc_vs_placement.png")

plt.show()


# ============================================================
# 15. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

correlation_df = df.copy()

# Convert categorical columns temporarily
correlation_df["ExtracurricularActivities"] = (
    correlation_df["ExtracurricularActivities"]
    .map({"No": 0, "Yes": 1})
)

correlation_df["PlacementTraining"] = (
    correlation_df["PlacementTraining"]
    .map({"No": 0, "Yes": 1})
)

correlation_df["PlacementStatus"] = (
    correlation_df["PlacementStatus"]
    .map({"NotPlaced": 0, "Placed": 1})
)

correlation_matrix = correlation_df.corr(numeric_only=True)

print(correlation_matrix["PlacementStatus"].sort_values(ascending=False))


# ============================================================
# 16. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()

plt.savefig("outputs/eda/correlation_heatmap.png")

plt.show()


# ============================================================
# 17. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EDA COMPLETED")
print("=" * 60)

print("\nEDA plots saved in:")
print("outputs/eda/")

print("\nGenerated files:")
for file in os.listdir("outputs/eda"):
    print("-", file)

print("\nNext step:")
print("Model Training and Comparison")
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/processed/placement_cleaned.csv")

print("=" * 70)
print("ADVANCED EDA - STUDENT PLACEMENT")
print("=" * 70)

print("Dataset shape:", df.shape)

# Convert target to numeric for analysis
df["Placed"] = df["PlacementStatus"].map({
    "NotPlaced": 0,
    "Placed": 1
})

output_dir = "outputs/advanced_eda"
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 2. OVERALL PLACEMENT RATE
# ============================================================

print("\n" + "=" * 70)
print("OVERALL PLACEMENT RATE")
print("=" * 70)

placement_rate = df["Placed"].mean() * 100

print(f"Placement Rate: {placement_rate:.2f}%")
print(f"Not Placed Rate: {100 - placement_rate:.2f}%")


# ============================================================
# 3. NUMERICAL FEATURE CORRELATION
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION WITH PLACEMENT")
print("=" * 70)

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

correlation = df[numeric_columns + ["Placed"]].corr()["Placed"]
correlation = correlation.drop("Placed").sort_values(ascending=False)

print(correlation)


# ============================================================
# 4. PLACEMENT RATE BY CGPA
# ============================================================

print("\n" + "=" * 70)
print("PLACEMENT RATE BY CGPA")
print("=" * 70)

cgpa_rate = (
    df.groupby("CGPA")["Placed"]
    .mean()
    .mul(100)
)

print(cgpa_rate)

plt.figure(figsize=(10, 5))
cgpa_rate.plot(kind="line", marker="o")
plt.title("Placement Rate by CGPA")
plt.xlabel("CGPA")
plt.ylabel("Placement Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/placement_by_cgpa.png")
plt.close()


# ============================================================
# 5. PLACEMENT RATE BY INTERNSHIPS
# ============================================================

print("\n" + "=" * 70)
print("PLACEMENT RATE BY INTERNSHIPS")
print("=" * 70)

internship_rate = (
    df.groupby("Internships")["Placed"]
    .mean()
    .mul(100)
)

print(internship_rate)

plt.figure(figsize=(8, 5))
internship_rate.plot(kind="bar")
plt.title("Placement Rate by Number of Internships")
plt.xlabel("Internships")
plt.ylabel("Placement Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{output_dir}/placement_by_internships.png")
plt.close()


# ============================================================
# 6. PLACEMENT RATE BY PROJECTS
# ============================================================

print("\n" + "=" * 70)
print("PLACEMENT RATE BY PROJECTS")
print("=" * 70)

project_rate = (
    df.groupby("Projects")["Placed"]
    .mean()
    .mul(100)
)

print(project_rate)

plt.figure(figsize=(8, 5))
project_rate.plot(kind="bar")
plt.title("Placement Rate by Number of Projects")
plt.xlabel("Projects")
plt.ylabel("Placement Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{output_dir}/placement_by_projects.png")
plt.close()


# ============================================================
# 7. PLACEMENT RATE BY APTITUDE SCORE
# ============================================================

print("\n" + "=" * 70)
print("PLACEMENT RATE BY APTITUDE SCORE")
print("=" * 70)

aptitude_rate = (
    df.groupby("AptitudeTestScore")["Placed"]
    .mean()
    .mul(100)
)

print(aptitude_rate)

plt.figure(figsize=(10, 5))
aptitude_rate.plot(kind="line", marker="o")
plt.title("Placement Rate by Aptitude Test Score")
plt.xlabel("Aptitude Test Score")
plt.ylabel("Placement Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/placement_by_aptitude.png")
plt.close()


# ============================================================
# 8. PLACEMENT RATE BY SSC MARKS
# ============================================================

print("\n" + "=" * 70)
print("PLACEMENT RATE BY SSC MARKS")
print("=" * 70)

ssc_rate = (
    df.groupby("SSC_Marks")["Placed"]
    .mean()
    .mul(100)
)

print(ssc_rate.head(20))

plt.figure(figsize=(10, 5))
ssc_rate.plot(kind="line")
plt.title("Placement Rate by SSC Marks")
plt.xlabel("SSC Marks")
plt.ylabel("Placement Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/placement_by_ssc.png")
plt.close()


# ============================================================
# 9. PLACEMENT RATE BY HSC MARKS
# ============================================================

print("\n" + "=" * 70)
print("PLACEMENT RATE BY HSC MARKS")
print("=" * 70)

hsc_rate = (
    df.groupby("HSC_Marks")["Placed"]
    .mean()
    .mul(100)
)

print(hsc_rate.head(20))

plt.figure(figsize=(10, 5))
hsc_rate.plot(kind="line")
plt.title("Placement Rate by HSC Marks")
plt.xlabel("HSC Marks")
plt.ylabel("Placement Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/placement_by_hsc.png")
plt.close()


# ============================================================
# 10. CATEGORICAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CATEGORICAL FEATURE ANALYSIS")
print("=" * 70)

for column in [
    "ExtracurricularActivities",
    "PlacementTraining"
]:

    print(f"\n{column}")

    result = (
        df.groupby(column)["Placed"]
        .mean()
        .mul(100)
    )

    print(result)

    plt.figure(figsize=(7, 5))
    result.plot(kind="bar")
    plt.title(f"Placement Rate by {column}")
    plt.xlabel(column)
    plt.ylabel("Placement Rate (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/placement_by_{column}.png"
    )
    plt.close()


# ============================================================
# 11. FEATURE IMPORTANCE STYLE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("FEATURE CORRELATION RANKING")
print("=" * 70)

for feature, value in correlation.items():
    print(f"{feature:30s}: {value:.4f}")


# ============================================================
# 12. SAVE CORRELATION
# ============================================================

correlation.to_csv(
    f"{output_dir}/feature_correlation.csv"
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("ADVANCED EDA COMPLETED")
print("=" * 70)

print("Graphs saved in:")
print(output_dir)

print("\nNext step:")
print("ADVANCED MODEL TRAINING")
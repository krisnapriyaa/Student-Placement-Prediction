import pandas as pd
import joblib


print("=" * 70)
print("STUDENT PLACEMENT PREDICTION")
print("=" * 70)


# --------------------------------------------------
# 1. LOAD TRAINED MODEL
# --------------------------------------------------

model = joblib.load("models/final_placement_model.pkl")
feature_columns = joblib.load("models/final_feature_columns.pkl")

print("\nModel loaded successfully.")
print("Model:", type(model).__name__)


# --------------------------------------------------
# 2. GET NEW STUDENT DETAILS
# --------------------------------------------------

print("\n" + "=" * 70)
print("ENTER NEW STUDENT DETAILS")
print("=" * 70)

CGPA = float(input("CGPA (6.5 - 9.1): "))

Internships = int(input("Number of Internships (0 - 2): "))

Projects = int(input("Number of Projects (0 - 3): "))

Workshops = int(
    input("Workshops/Certifications (0 - 3): ")
)

AptitudeTestScore = int(
    input("Aptitude Test Score (60 - 90): ")
)

SoftSkillsRating = float(
    input("Soft Skills Rating (3.0 - 4.8): ")
)

ExtracurricularActivities = input(
    "Extracurricular Activities (Yes/No): "
).strip().title()

PlacementTraining = input(
    "Placement Training (Yes/No): "
).strip().title()

SSC_Marks = int(
    input("SSC Marks (55 - 90): ")
)

HSC_Marks = int(
    input("HSC Marks (57 - 88): ")
)


# --------------------------------------------------
# 3. ENCODE CATEGORICAL FEATURES
# --------------------------------------------------

ExtracurricularActivities = (
    1 if ExtracurricularActivities == "Yes" else 0
)

PlacementTraining = (
    1 if PlacementTraining == "Yes" else 0
)


# --------------------------------------------------
# 4. CREATE DATAFRAME
# --------------------------------------------------

student = pd.DataFrame(
    [[
        CGPA,
        Internships,
        Projects,
        Workshops,
        AptitudeTestScore,
        SoftSkillsRating,
        ExtracurricularActivities,
        PlacementTraining,
        SSC_Marks,
        HSC_Marks
    ]],
    columns=feature_columns
)


# --------------------------------------------------
# 5. MAKE PREDICTION
# --------------------------------------------------

prediction = model.predict(student)[0]

probabilities = model.predict_proba(student)[0]

not_placed_probability = probabilities[0]

placed_probability = probabilities[1]


# --------------------------------------------------
# 6. DISPLAY RESULT
# --------------------------------------------------

print("\n" + "=" * 70)
print("PLACEMENT PREDICTION RESULT")
print("=" * 70)

if prediction == 1:
    print("\nPrediction: PLACED")
else:
    print("\nPrediction: NOT PLACED")


print(
    f"\nPlaced Probability     : "
    f"{placed_probability * 100:.2f}%"
)

print(
    f"Not Placed Probability: "
    f"{not_placed_probability * 100:.2f}%"
)


# --------------------------------------------------
# 7. DISPLAY STUDENT DATA
# --------------------------------------------------

print("\n" + "=" * 70)
print("STUDENT INPUT")
print("=" * 70)

print(student.to_string(index=False))


print("\n" + "=" * 70)
print("PREDICTION COMPLETED")
print("=" * 70)
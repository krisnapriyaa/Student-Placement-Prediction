
import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "models/final_placement_model.pkl"
FEATURE_PATH = "models/final_feature_columns.pkl"

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)


# ============================================================
# TRAINING DATA RANGES
# ============================================================
# These are the ranges observed in our training dataset.
# They are used ONLY for OOD warnings.
# They do NOT restrict the user's input.

training_ranges = {
    "CGPA": (6.5, 9.1),
    "Internships": (0, 2),
    "Projects": (0, 3),
    "Workshops/Certifications": (0, 3),
    "AptitudeTestScore": (60, 90),
    "SoftSkillsRating": (3.0, 4.8),
    "SSC_Marks": (55, 90),
    "HSC_Marks": (57, 88)
}


# ============================================================
# TITLE
# ============================================================

st.title("🎓 Student Placement Prediction System")

st.write(
    "Enter a student's academic, skill, and activity details "
    "to predict placement status using Machine Learning."
)

st.info(
    "💡 Higher values are allowed for internships, projects, "
    "and certifications. If a value is outside the range seen "
    "during model training, the application will show a warning."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📌 Model Information")

st.sidebar.write(
    "**Model:** Logistic Regression"
)

st.sidebar.write(
    "**Problem:** Binary Classification"
)

st.sidebar.write(
    "**Target:** PlacementStatus"
)

st.sidebar.write(
    "**Classes:** Placed / Not Placed"
)

st.sidebar.divider()

st.sidebar.subheader("📊 Training Data Ranges")

st.sidebar.write("CGPA: 6.5 – 9.1")
st.sidebar.write("Internships: 0 – 2")
st.sidebar.write("Projects: 0 – 3")
st.sidebar.write("Certifications: 0 – 3")
st.sidebar.write("Aptitude: 60 – 90")
st.sidebar.write("Soft Skills: 3.0 – 4.8")
st.sidebar.write("SSC: 55 – 90")
st.sidebar.write("HSC: 57 – 88")


# ============================================================
# INPUT SECTION
# ============================================================

st.header("📝 Student Information")

col1, col2 = st.columns(2)


# ============================================================
# COLUMN 1
# ============================================================

with col1:

    cgpa = st.number_input(
        "CGPA",
        min_value=0.0,
        max_value=10.0,
        value=7.5,
        step=0.1
    )

    internships = st.number_input(
        "Number of Internships",
        min_value=0,
        max_value=10,
        value=1,
        step=1
    )

    projects = st.number_input(
        "Number of Projects",
        min_value=0,
        max_value=15,
        value=2,
        step=1
    )

    workshops = st.number_input(
        "Workshops / Certifications",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )

    aptitude = st.number_input(
        "Aptitude Test Score",
        min_value=0,
        max_value=100,
        value=75,
        step=1
    )


# ============================================================
# COLUMN 2
# ============================================================

with col2:

    softskills = st.number_input(
        "Soft Skills Rating",
        min_value=0.0,
        max_value=5.0,
        value=4.0,
        step=0.1
    )

    extracurricular = st.selectbox(
        "Extracurricular Activities",
        ["Yes", "No"]
    )

    placement_training = st.selectbox(
        "Placement Training",
        ["Yes", "No"]
    )

    ssc = st.number_input(
        "SSC Marks",
        min_value=0,
        max_value=100,
        value=75,
        step=1
    )

    hsc = st.number_input(
        "HSC Marks",
        min_value=0,
        max_value=100,
        value=75,
        step=1
    )


st.divider()


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔮 Predict Placement",
    use_container_width=True
):

    # ========================================================
    # VALIDATION
    # ========================================================

    valid_input = True

    if not 0 <= cgpa <= 10:
        st.error("CGPA must be between 0 and 10.")
        valid_input = False

    if not 0 <= internships:
        st.error("Internships cannot be negative.")
        valid_input = False

    if not 0 <= projects:
        st.error("Projects cannot be negative.")
        valid_input = False

    if not 0 <= workshops:
        st.error("Workshops / Certifications cannot be negative.")
        valid_input = False

    if not 0 <= aptitude <= 100:
        st.error("Aptitude score must be between 0 and 100.")
        valid_input = False

    if not 0 <= softskills <= 5:
        st.error("Soft Skills Rating must be between 0 and 5.")
        valid_input = False

    if not 0 <= ssc <= 100:
        st.error("SSC marks must be between 0 and 100.")
        valid_input = False

    if not 0 <= hsc <= 100:
        st.error("HSC marks must be between 0 and 100.")
        valid_input = False


    # ========================================================
    # OUT-OF-DISTRIBUTION CHECK
    # ========================================================

    warnings = []

    input_values = {
        "CGPA": cgpa,
        "Internships": internships,
        "Projects": projects,
        "Workshops/Certifications": workshops,
        "AptitudeTestScore": aptitude,
        "SoftSkillsRating": softskills,
        "SSC_Marks": ssc,
        "HSC_Marks": hsc
    }

    for feature, value in input_values.items():

        minimum, maximum = training_ranges[feature]

        if value < minimum or value > maximum:

            warnings.append(
                f"{feature}: entered value {value} is outside "
                f"the training range ({minimum}–{maximum})."
            )


    if warnings:

        st.warning(
            "⚠️ Some inputs are outside the range observed "
            "in the training dataset."
        )

        for warning in warnings:
            st.write("•", warning)

        st.info(
            "The model will still generate a prediction, "
            "but this prediction may be less reliable because "
            "the model has limited training evidence for these values."
        )


    # ========================================================
    # CONTINUE PREDICTION
    # ========================================================

    if valid_input:

        # ----------------------------------------------------
        # CATEGORICAL ENCODING
        # ----------------------------------------------------

        extracurricular_value = (
            1 if extracurricular == "Yes" else 0
        )

        training_value = (
            1 if placement_training == "Yes" else 0
        )


        # ----------------------------------------------------
        # CREATE INPUT DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [[
                cgpa,
                internships,
                projects,
                workshops,
                aptitude,
                softskills,
                extracurricular_value,
                training_value,
                ssc,
                hsc
            ]],
            columns=feature_columns
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        not_placed_probability = probabilities[0] * 100

        placed_probability = probabilities[1] * 100


        # ====================================================
        # RESULT
        # ====================================================

        st.header("📊 Prediction Result")

        if prediction == 1:

            st.success(
                "🎉 Prediction: PLACED"
            )

        else:

            st.error(
                "⚠️ Prediction: NOT PLACED"
            )


        # ====================================================
        # PROBABILITY
        # ====================================================

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "🎯 Placed Probability",
                f"{placed_probability:.2f}%"
            )

        with result_col2:

            st.metric(
                "📉 Not Placed Probability",
                f"{not_placed_probability:.2f}%"
            )


        # ====================================================
        # PROBABILITY BAR
        # ====================================================

        st.subheader("📈 Placement Probability")

        st.progress(
            min(max(int(placed_probability), 0), 100)
        )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        st.subheader("📋 Student Input")

        display_data = pd.DataFrame({

            "Feature": [
                "CGPA",
                "Internships",
                "Projects",
                "Workshops / Certifications",
                "Aptitude Test Score",
                "Soft Skills Rating",
                "Extracurricular Activities",
                "Placement Training",
                "SSC Marks",
                "HSC Marks"
            ],

            "Value": [
                cgpa,
                internships,
                projects,
                workshops,
                aptitude,
                softskills,
                extracurricular,
                placement_training,
                ssc,
                hsc
            ]
        })

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Student Placement Prediction with Explainable Machine Learning"
)


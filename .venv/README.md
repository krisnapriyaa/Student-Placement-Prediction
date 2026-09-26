Student Placement Prediction using Machine Learning

📌 Project Overview

Student Placement Prediction is a machine learning classification project that predicts whether a student is likely to be Placed or Not Placed based on student-related academic, skill, soft-skill, and experience information.

The project follows a complete data science workflow:

Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis (EDA)
   ↓
Feature Engineering
   ↓
Train / Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Prediction for New Students

🎯 Objective

The main objective is to build a machine learning system that can learn patterns from historical student data and predict the student's placement status.

Target Variable

PlacementStatus

Placed

NotPlaced

The model is designed to support early identification of students who may need additional academic, technical, soft-skill, or experience-building support.

📊 Dataset Summary

The project uses a dataset containing 10,000 student records.

Target Distribution

Placement Status

Records

NotPlaced

5,803

Placed

4,197

Total

10,000

This corresponds to approximately:

58.03% NotPlaced

41.97% Placed

The dataset was checked for missing values and duplicate records during preprocessing.

🧹 Data Preprocessing

The preprocessing pipeline includes:

Loading the student dataset.

Checking for missing values.

Checking for duplicate records.

Removing StudentID because it is an identifier and does not provide useful predictive information.

Preparing the feature matrix and target variable.

Splitting the data into training and testing sets.

The training workflow uses 8,000 samples for model training, with the remaining data used for evaluation.

🛠️ Feature Engineering

Additional features were created to represent the student's overall performance more effectively.

Engineered Features

AcademicAverage

Represents the student's overall academic performance using the available academic-related attributes.

SoftSkillsPercentage

Represents the student's soft-skill performance as a percentage-based feature.

SkillScore

Combines relevant skill-related information into a single model feature.

ExperienceScore

Represents the student's experience-related information in a model-friendly form.

Feature engineering helps the model learn higher-level patterns instead of depending only on individual raw columns.

🤖 Machine Learning Approach

This is a supervised binary classification problem.

The model learns from historical examples where the placement outcome is already known.

Input Student Data
       ↓
Feature Engineering
       ↓
Trained Classification Model
       ↓
PlacementStatus
       ↓
Placed / NotPlaced

The trained model is saved using joblib so it can be reused for prediction without retraining every time.

💾 Saved Model Files

The final trained artifacts include:

models/
├── final_placement_model.pkl
└── final_feature_columns.pkl

final_placement_model.pkl

Contains the trained machine learning model.

final_feature_columns.pkl

Stores the feature-column order used during training. This helps ensure that new prediction data is passed to the model using the same feature structure as the training data.

🔮 Prediction Workflow

For a new student, the system follows this process:

New Student Details
        ↓
Preprocessing
        ↓
Feature Engineering
        ↓
Arrange Features in Training Order
        ↓
Load final_placement_model.pkl
        ↓
Predict Placement Status
        ↓
Placed / NotPlaced

The prediction stage uses the saved model instead of retraining the algorithm.

🧠 Technologies Used

Python

Pandas — data loading and manipulation

NumPy — numerical operations

Scikit-learn — machine learning and evaluation

Joblib — model serialization

Matplotlib / Seaborn — visualization and EDA, where applicable

Jupyter Notebook / VS Code — development environment

📁 Project Structure

A typical project organization is:

Student-Placement-Prediction/
│
├── data/
│   └── placement_dataset.csv
│
├── models/
│   ├── final_placement_model.pkl
│   └── final_feature_columns.pkl
│
├── notebooks/
│   └── EDA and model development notebooks
│
├── src/
│   ├── preprocessing / feature engineering files
│   ├── model training files
│   └── prediction files
│
├── requirements.txt
└── README.md

Adjust the filenames above to match the exact files in the repository.

⚙️ Installation

Create and activate a virtual environment:

Windows

python -m venv .venv
.venv\Scripts\activate

Install the required packages:

pip install pandas numpy scikit-learn joblib matplotlib seaborn

If the project contains a requirements.txt file, use:

pip install -r requirements.txt

▶️ Running the Project

Activate the virtual environment.

Run the preprocessing / training script.

The trained model is saved to the models/ directory.

Run the prediction script.

Enter the new student's details when prompted.

The model returns the predicted placement status.

Example prediction flow:

============================================================
ENTER NEW STUDENT DETAILS
============================================================

Enter student information...

Prediction: Placed

📈 Model Evaluation

The model should be evaluated using classification metrics such as:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

These metrics help determine how effectively the classifier distinguishes between Placed and NotPlaced students.

Note: Do not report an accuracy value in this README unless it has been measured from the final trained model on the project's evaluation set.

🔍 Why This Is a Data Science / Machine Learning Project

This project covers the core stages of a practical data science workflow:

Data Collection
     ↓
Data Understanding
     ↓
Data Cleaning
     ↓
EDA
     ↓
Feature Engineering
     ↓
Machine Learning
     ↓
Model Evaluation
     ↓
Prediction

It is therefore more than a simple prediction script: the project demonstrates how raw student data is transformed into features, used to train a supervised learning model, evaluated, saved, and finally reused for new predictions.

🚀 Future Improvements

Possible future enhancements include:

Compare multiple classification algorithms.

Add hyperparameter tuning.

Perform cross-validation.

Add feature-importance analysis.

Build an interactive Streamlit dashboard.

Add probability/confidence output.

Track model performance on new student batches.

Deploy the model as a web API.

⚠️ Disclaimer

This project is intended for educational and predictive-analysis purposes. Placement predictions are model-based estimates and should not be treated as a guarantee of employment.

👩‍💻 Author

Krisnapriyaa Kanagasundaram

B.Tech Information Technology | Data Science & Analytics

⭐ Project Summary

Student Placement Prediction demonstrates an end-to-end supervised machine learning workflow for predicting student placement status from academic, skill, soft-skill, and experience-related information. The project includes preprocessing, feature engineering, model training, evaluation, model serialization, and prediction for new students.
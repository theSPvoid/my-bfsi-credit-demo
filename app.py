import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pyrebase
import streamlit as st
import json
from streamlit_lottie import st_lottie
from firebase_config import firebase_config

# 1. Define custom CSS (gradient background, center text, fade animation)
st.markdown("""
<style>
    /* Gradient background for the entire page */
    body {
        background: linear-gradient(to right, #009fff, #ec2f4b);
        margin: 0;
        padding: 0;
    }
    /* Make the sidebar background black and all its text white */
    section[data-testid="stSidebar"] {
        background-color: black !important;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    /* Change the accent color (slider color) for all range inputs in the sidebar to red */
    section[data-testid="stSidebar"] input[type="range"] {
        accent-color: red !important;
    }
    /* Custom heading styles with fade-down animation */
    h1.title {
        font-family: "Arial Black", Gadget, sans-serif;
        color: #ffffff;
        text-align: center;
        margin-top: 20px;
        animation: fadeDown 1.5s ease-in;
    }
    h2.subtitle {
        color: #ffe6e6;
        text-align: center;
        font-weight: normal;
        animation: fadeDown 2s ease-in;
    }
    h3.group {
        color: #ffffcc;
        text-align: center;
        animation: fadeDown 2.5s ease-in;
    }
    h4.project {
        color: #ffffff;
        text-align: center;
        animation: fadeDown 3s ease-in;
    }
    @keyframes fadeDown {
      0% {
        opacity: 0;
        transform: translateY(-20px);
      }
      100% {
        opacity: 1;
        transform: translateY(0);
      }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/theSPvoid/my-bfsi-credit-demo/main/xim_logo.png" alt="XIM University Logo" width="180">
</div>
""", unsafe_allow_html=True)

# 3. Display the headings with animations
st.markdown("<h1 class='title'>XIM University</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='subtitle'>BIS Group Project</h2>", unsafe_allow_html=True)
st.markdown("<h3 class='group'>Group 8</h3>", unsafe_allow_html=True)
st.markdown("<h4 class='project'>AI-Driven BFSI Credit Risk Demo</h4>", unsafe_allow_html=True)

# Load Lottie Animation from GitHub Repository
@st.cache_data
def load_lottie_local(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load the saved JSON Lottie file from the GitHub repo folder
try:
    lottie_animation = load_lottie_local("finance_animation.json")  # Adjust path if needed
    st_lottie(lottie_animation, height=300, key="finance_animation")
except:
    st.write("Lottie animation file not found. Make sure 'finance_animation.json' is in the same GitHub folder.")

# 4. Add a small spacing line before continuing
st.write("---")

# Now you can continue with the BFSI input forms, model loading, etc.
# Example (just placeholders):
st.header("Enter Applicant Data (Below)")

# 0. Initialize Firebase
firebase = pyrebase.initialize_app(firebase_config)
db = firebase.database()

# 1. Load Models & Feature Names
logreg_model = joblib.load("logreg_model.pkl")
dtree_model = joblib.load("dtree_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.sidebar.header("Enter Applicant Data")

# 1. Collect user inputs
Gender = st.sidebar.selectbox("Gender", ["Male","Female"])
Married = st.sidebar.selectbox("Married?", ["No","Yes"])
Dependents = st.sidebar.slider("Dependents", 0, 3, 0)
Education = st.sidebar.selectbox("Education", ["Graduate","Not Graduate"])
Self_Employed = st.sidebar.selectbox("Self Employed?", ["No","Yes"])
ApplicantIncome = st.sidebar.number_input("Applicant Income", min_value=0, value=5000)
CoapplicantIncome = st.sidebar.number_input("Co-applicant Income", min_value=0, value=0)
LoanAmount = st.sidebar.number_input("Loan Amount (in thousands)", min_value=0, value=100)
Loan_Amount_Term = st.sidebar.selectbox("Loan Term (months)", [360, 240, 180, 120])
Credit_History = st.sidebar.selectbox("Credit History (1=Yes, 0=No)", [1.0, 0.0])
Property_Area = st.sidebar.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

# Alternative data
Utility_Payment_Score = st.sidebar.slider("Utility Payment Score", 0.0, 1.0, 0.5, 0.01)
Mobile_Transactions = st.sidebar.slider("Mobile Transactions/month", 0, 200, 50)
Social_Media_Score = st.sidebar.slider("Social Media Score", 0, 10, 5)

# For demonstration, we simulate manual model calculations rather than using pre-trained models.
model_choice = st.sidebar.selectbox("Model", ["Manual Logistic Regression", "Manual Decision Tree"])

if st.sidebar.button("Predict & Save to Firebase"):
    # 2. Define explicit weights for each input (these are illustrative and can be tuned)
    weights = {
        'ApplicantIncome': 0.00005,          # Small positive weight: higher income increases credit score.
        'CoapplicantIncome': 0.00003,          # Similar, but a bit smaller.
        'LoanAmount': -0.002,                  # Negative weight: larger loan amount lowers credit score.
        'Loan_Amount_Term': 0.001,             # Slight positive weight: longer term can help.
        'Credit_History': 2.5,                 # Strong positive weight: good credit history greatly increases probability.
        'Dependents': -0.05,                   # More dependents may slightly lower credit score.
        'Utility_Payment_Score': 2.0,          # Very high weight: excellent utility payments boost score.
        'Mobile_Transactions': 0.001,          # Small positive effect: more transactions imply financial activity.
        'Social_Media_Score': 0.05,            # Modest positive weight.
        'Gender_Male': 0.1,                    # Slight positive impact if male (example).
        'Married_Yes': 0.2,                    # Being married can be seen as stability.
        'Education_Not Graduate': -0.2,        # Not graduating lowers score.
        'Self_Employed_Yes': -0.1,             # Slightly negative impact if self-employed.
        'Property_Area_Semiurban': 0.1,        # Positive impact if semiurban.
        'Property_Area_Urban': 0.2             # Higher positive impact if urban.
    }
    bias = -1  # Intercept term

    # 3. Construct the input vector manually
    x = {}
    x['ApplicantIncome'] = ApplicantIncome
    x['CoapplicantIncome'] = CoapplicantIncome
    x['LoanAmount'] = LoanAmount
    x['Loan_Amount_Term'] = Loan_Amount_Term
    x['Credit_History'] = Credit_History
    x['Dependents'] = Dependents
    x['Utility_Payment_Score'] = Utility_Payment_Score
    x['Mobile_Transactions'] = Mobile_Transactions
    x['Social_Media_Score'] = Social_Media_Score

    # Categorical dummies:
    x['Gender_Male'] = 1 if Gender == "Male" else 0
    x['Married_Yes'] = 1 if Married == "Yes" else 0
    x['Education_Not Graduate'] = 1 if Education == "Not Graduate" else 0
    x['Self_Employed_Yes'] = 1 if Self_Employed == "Yes" else 0
    x['Property_Area_Semiurban'] = 1 if Property_Area == "Semiurban" else 0
    x['Property_Area_Urban'] = 1 if Property_Area == "Urban" else 0

    # 4. Calculate the weighted sum z
    z = bias
    for feature, value in x.items():
        z += weights.get(feature, 0) * value

    # 5. Calculate probability using the sigmoid function:
    prob = 1 / (1 + np.exp(-z))

    # For the manual decision tree simulation, we adjust the logic:
    if model_choice == "Manual Decision Tree":
        # For decision trees, the flow is sequential. We simulate it:
        # Start with base probability based on credit history:
        base_prob = 0.3 if Credit_History == 0.0 else 0.7
        # Then, modify based on utility payment score:
        # For every 0.1 above 0.5, increase probability by 0.03, and vice versa.
        utility_effect = (Utility_Payment_Score - 0.5) * 0.3
        # Also consider income: if ApplicantIncome > 20000, add 0.05
        income_effect = 0.05 if ApplicantIncome > 20000 else 0
        prob = base_prob + utility_effect + income_effect
        prob = max(0, min(prob, 1))  # Clamp between 0 and 1
    
    prediction = 1 if prob >= 0.5 else 0

    # 6. Convert probability to Indian-style Credit Score (300 to 900)
    credit_score = 300 + (prob * 600)

    # 7. Display the result
    if prediction == 1:
        st.success("Loan Approved!")
    else:
        st.error("Loan Denied!")
    st.write(f"**Credit Score:** {int(round(credit_score, 0))}")
    st.write(f"**Approval Probability:** {round(prob, 3)}")

    # 8. Build a short heuristic-based explanation
    explanation_list = []
    if Credit_History == 0.0:
        explanation_list.append("No credit history reduces approval odds.")
    else:
        explanation_list.append("Good credit history boosts approval odds.")
    if Utility_Payment_Score < 0.3:
        explanation_list.append("Low utility payment score suggests risk of late payments.")
    elif Utility_Payment_Score > 0.7:
        explanation_list.append("High utility payment score indicates strong on-time payments.")
    # More explanations can be added similarly.
    st.write("### Explanation:")
    for expl in explanation_list:
        st.write("-", expl)

    # 9. Save applicant data and results to Firebase
    applicant_data = {
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": float(Credit_History),
        "Property_Area": Property_Area,
        "Utility_Payment_Score": Utility_Payment_Score,
        "Mobile_Transactions": Mobile_Transactions,
        "Social_Media_Score": Social_Media_Score,
        "ModelUsed": model_choice,
        "CalculatedProbability": round(prob, 3),
        "Prediction": "Approved" if prediction == 1 else "Denied",
        "CreditScore": int(round(credit_score, 0))
    }
    db.child("loan_applications").push(applicant_data)
    st.info("Applicant data and result saved to Firebase Realtime Database.")

st.markdown("---")
st.subheader("Retrieve Records from Firebase")
if st.button("Fetch All Records"):
    records = db.child("loan_applications").get()
    if records.each():
        for r in records.each():
            rec_data = r.val()
            st.write(rec_data)
    else:
        st.write("No records found in 'loan_applications' node.")

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

model_choice = st.sidebar.selectbox("Model", ["Manual Logistic Regression", "Manual Decision Tree"])

if st.sidebar.button("Predict & Save to Firebase"):

    # 2. Define explicit weights for each input (illustrative)
    # We'll treat LoanAmount by ratio to (ApplicantIncome + CoapplicantIncome + 1) to avoid huge negative sums
    weights = {
        'ApplicantIncome': 0.00005,         # Higher incomes => higher probability
        'CoapplicantIncome': 0.00003,       # Additional income => small positive
        'Loan_Amount_Ratio': -0.05,         # Compare loan to total income
        'Loan_Amount_Term': 0.001,          # Slight positive if longer term
        'Credit_History': 2.5,             # Good credit => strong boost
        'Dependents': -0.05,               # More dependents => slightly lower
        'Utility_Payment_Score': 2.0,       # High utility => strong positive
        'Mobile_Transactions': 0.001,       # More transactions => small positive
        'Social_Media_Score': 0.05,         # Modest positive
        'Gender_Male': 0.1,                 # Slight positive if male
        'Married_Yes': 0.2,                 # Being married => stability
        'Education_Not Graduate': -0.2,     # Not graduate => negative
        'Self_Employed_Yes': -0.1,          # Slight negative
        'Property_Area_Semiurban': 0.1,     # Semiurban => mild positive
        'Property_Area_Urban': 0.2          # Urban => stronger positive
    }
    bias = -1  # Intercept term

    # 3. Construct the input dictionary
    x = {}

    # We'll define a ratio: LoanAmount / (ApplicantIncome + CoapplicantIncome + 1)
    # This keeps large LoanAmounts from dominating
    total_income = ApplicantIncome + CoapplicantIncome + 1
    loan_income_ratio = 0
    if total_income > 0:
        loan_income_ratio = LoanAmount / total_income

    x['ApplicantIncome'] = ApplicantIncome
    x['CoapplicantIncome'] = CoapplicantIncome
    x['Loan_Amount_Ratio'] = loan_income_ratio
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

    # 4. Calculate Weighted Sum (z) for the logistic approach
    z = bias
    for feature, value in x.items():
        w = weights.get(feature, 0)
        z += w * value

    # We'll remove clipping of z so you can see real changes:
    # but to avoid extreme overflow in math.exp, let's do a smaller clamp
    z = np.clip(z, -20, 20)  # This is a mild clamp

    # 5. Probability using the Sigmoid (Manual Logistic Regression)
    prob_log = 1 / (1 + np.exp(-z))

    # 6. Manual Decision Tree simulation
    if model_choice == "Manual Decision Tree":
        # Start with base prob depending on credit history
        base_prob = 0.3 if Credit_History == 0.0 else 0.7
        # Adjust for Utility Payment Score:
        # e.g., if Utility_Payment_Score = 0.9 => +0.4 above 0.5 => 0.4 * 0.3 = +0.12
        utility_effect = (Utility_Payment_Score - 0.5) * 0.3
        # Adjust for income: if (ApplicantIncome + CoapplicantIncome) > 20000 => +0.05, else -0.05
        if (ApplicantIncome + CoapplicantIncome) > 20000:
            income_effect = 0.05
        else:
            income_effect = -0.05
        # Loan Ratio effect: if loan_income_ratio is too high => reduce prob
        # e.g., if ratio > 5 => reduce 0.1
        ratio_effect = 0
        if loan_income_ratio > 5:
            ratio_effect = -0.1
        prob_tree = base_prob + utility_effect + income_effect + ratio_effect
        prob_tree = max(0, min(prob_tree, 1))
        prob = prob_tree
    else:
        prob = prob_log

    prediction = 1 if prob >= 0.5 else 0

    # Convert Probability to Credit Score (300-900)
    credit_score = 300 + (prob * 600)

    # Debug: Show Weighted Sum, Probability, Loan Ratio
    st.write("Weighted sum (z):", round(z, 4))
    st.write("Calculated Probability:", round(prob, 4))
    st.write("Loan-Income Ratio:", round(loan_income_ratio, 4))

    # Show result
    if prediction == 1:
        st.success("Loan Approved!")
    else:
        st.error("Loan Denied!")
    st.write(f"**Credit Score:** {int(round(credit_score, 0))}")

    # Simple explanation
    explanation_list = []
    if Credit_History == 0.0:
        explanation_list.append("No credit history lowers approval odds.")
    else:
        explanation_list.append("Good credit history strongly boosts approval odds.")
    if Utility_Payment_Score < 0.3:
        explanation_list.append("Low utility payment score => risk of late bill payments.")
    elif Utility_Payment_Score > 0.7:
        explanation_list.append("High utility payment score => strong on-time payments.")
    st.write("### Explanation:")
    for expl in explanation_list:
        st.write("-", expl)

    # Save data + result to Firebase
    applicant_data = {
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount_in_thousands": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": float(Credit_History),
        "Property_Area": Property_Area,
        "Utility_Payment_Score": Utility_Payment_Score,
        "Mobile_Transactions": Mobile_Transactions,
        "Social_Media_Score": Social_Media_Score,
        "ModelUsed": model_choice,
        "CalculatedProbability": round(prob, 4),
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

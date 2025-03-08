import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pyrebase
from firebase_config import firebase_config

#  DECORATIVE LANDING PAGE SECTION
# --------------------------------------------
st.markdown("<h1 style='text-align: center;'>XIM University</h1>", unsafe_allow_html=True)

# If you want slightly smaller text or a subheader style:
st.markdown("<h2 style='text-align: center;'>BIS Group Project</h2>", unsafe_allow_html=True)

# Another line for Group 8
st.markdown("<h3 style='text-align: center;'>Group 8</h3>", unsafe_allow_html=True)

# Project / Demo name
st.markdown("<h4 style='text-align: center;'>AI-Driven BFSI Credit Risk Demo</h4>", unsafe_allow_html=True)

# 0. Initialize Firebase
firebase = pyrebase.initialize_app(firebase_config)
db = firebase.database()

# 1. Load Models & Feature Names
logreg_model = joblib.load("logreg_model.pkl")
dtree_model = joblib.load("dtree_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.sidebar.header("Enter Applicant Data")

# 2. Collect user inputs
Gender = st.sidebar.selectbox("Gender", ["Male","Female"])
Married = st.sidebar.selectbox("Married?", ["No","Yes"])
Dependents = st.sidebar.slider("Dependents", 0,3,0)
Education = st.sidebar.selectbox("Education", ["Graduate","Not Graduate"])
Self_Employed = st.sidebar.selectbox("Self Employed?", ["No","Yes"])
ApplicantIncome = st.sidebar.number_input("Applicant Income", min_value=0, value=5000)
CoapplicantIncome = st.sidebar.number_input("Co-applicant Income", min_value=0, value=0)
LoanAmount = st.sidebar.number_input("Loan Amount (in thousands)", min_value=0, value=100)
Loan_Amount_Term = st.sidebar.selectbox("Loan Term (months)", [360,240,180,120])
Credit_History = st.sidebar.selectbox("Credit History (1=Yes,0=No)", [1.0,0.0])
Property_Area = st.sidebar.selectbox("Property Area", ["Rural","Semiurban","Urban"])

# Alternative data
Utility_Payment_Score = st.sidebar.slider("Utility Payment Score", 0.0,1.0,0.5,0.01)
Mobile_Transactions = st.sidebar.slider("Mobile Transactions/month", 0,200,50)
Social_Media_Score = st.sidebar.slider("Social Media Score", 0,10,5)

model_choice = st.sidebar.selectbox("Model", ["Logistic Regression","Decision Tree"])

if st.sidebar.button("Predict & Save to Firebase"):
    # 3. Build a single row DataFrame for our models
    single_row = pd.DataFrame(np.zeros((1,len(feature_names))), columns=feature_names)

    # Numeric columns
    numeric_map = {
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'Dependents': Dependents,
        'Utility_Payment_Score': Utility_Payment_Score,
        'Mobile_Transactions': Mobile_Transactions,
        'Social_Media_Score': Social_Media_Score
    }
    for col,val in numeric_map.items():
        if col in single_row.columns:
            single_row[col] = val

    # Dummy columns for the categories
    if 'Gender_Male' in single_row.columns and Gender=='Male':
        single_row['Gender_Male'] = 1
    if 'Married_Yes' in single_row.columns and Married=='Yes':
        single_row['Married_Yes'] = 1
    if 'Education_Not Graduate' in single_row.columns and Education=='Not Graduate':
        single_row['Education_Not Graduate'] = 1
    if 'Self_Employed_Yes' in single_row.columns and Self_Employed=='Yes':
        single_row['Self_Employed_Yes'] = 1
    if 'Property_Area_Semiurban' in single_row.columns and Property_Area=='Semiurban':
        single_row['Property_Area_Semiurban'] = 1
    if 'Property_Area_Urban' in single_row.columns and Property_Area=='Urban':
        single_row['Property_Area_Urban'] = 1

    # 4. Predict Probability & Outcome
    if model_choice=="Logistic Regression":
        prob_approve = logreg_model.predict_proba(single_row)[0][1]
    else:
        prob_approve = dtree_model.predict_proba(single_row)[0][1]

    prediction = 1 if prob_approve>=0.5 else 0

    # 5. Convert prob -> Credit Score (Indian style: 300-900)
    credit_score = 300 + (prob_approve * 600)

    # Show result
    if prediction==1:
        st.success("Loan Approved!")
    else:
        st.error("Loan Denied")

    st.write(f"**Credit Score**: {int(round(credit_score,0))}")

    # 6. Build a short explanation (heuristic-based)
    explanation_list = []
    if Credit_History==0.0:
        explanation_list.append("No credit history -> lowers approval odds.")
    else:
        explanation_list.append("Positive credit history -> boosts approval odds.")

    if Utility_Payment_Score < 0.3:
        explanation_list.append("Low utility payment score -> risk of late bill payments.")
    elif Utility_Payment_Score > 0.7:
        explanation_list.append("High utility payment score -> strong on-time payment history.")

    # etc. (Add more if you like)
    st.write("### Explanation:")
    for expl in explanation_list:
        st.write("-", expl)

    # 7. Save the data + result to Firebase
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
        "ProbabilityOfApproval": round(prob_approve,3),
        "Prediction": "Approved" if prediction==1 else "Denied",
        "CreditScore": int(round(credit_score,0))
    }

    db.child("loan_applications").push(applicant_data)
    st.info("Applicant data + result saved to Firebase Realtime Database.")


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

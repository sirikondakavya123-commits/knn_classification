import streamlit as st
import pickle
import numpy as np

# PAGE CONFIG

st.set_page_config(
    page_title="Loan Prediction",
    page_icon="💰",
    layout="centered"
)

# CUSTOM BACKGROUND

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #dbeafe,
        #fef3c7,
        #dcfce7,
        #fae8ff
    );
    color: black;
}

h1 {
    color: #1e3a8a;
    text-align: center;
}

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

div[data-baseweb="input"] {
    background-color: white;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# LOAD MODEL

model = pickle.load(
    open(r"/models/model.pkl", "rb")
)

# LOAD SCALER

scaler = pickle.load(
    open(r"/models/scaler.pkl", "rb")
)

# TITLE

st.title("💰 Loan Prediction System")

st.write("Enter Applicant Details")

# USER INPUTS

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

married = st.selectbox(
    "Marital Status",
    ["No", "Yes"]
)

dependents = st.selectbox(
    "Dependents",
    ["0", "1", "2", "3+"]
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["No", "Yes"]
)

applicant_income = st.number_input(
    "Applicant Income",
    min_value=0,
    value=5000
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0.0,
    value=1500.0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0.0,
    value=120.0
)

loan_amount_term = st.number_input(
    "Loan Amount Term",
    min_value=0.0,
    value=360.0
)

credit_history = st.selectbox(
    "Credit History",
    [0.0, 1.0]
)

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

# MANUAL ENCODING

gender = 1 if gender == "Male" else 0

married = 1 if married == "Yes" else 0

dependents_map = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3+": 3
}

dependents = dependents_map[dependents]

education = 0 if education == "Graduate" else 1

self_employed = 1 if self_employed == "Yes" else 0

property_map = {
    "Rural": 0,
    "Semiurban": 1,
    "Urban": 2
}

property_area = property_map[property_area]

# PREDICT BUTTON

if st.button("Predict Loan Status"):

    features = np.array([
        [
            gender,
            married,
            dependents,
            education,
            self_employed,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_amount_term,
            credit_history,
            property_area
        ]
    ])

    # SCALE FEATURES

    scaled_features = scaler.transform(features)

    # PREDICT

    prediction = model.predict(
        scaled_features
    )

    # DISPLAY RESULT

    if prediction[0] == 1:

        st.success(
            "Loan Approved ✅"
        )

    else:

        st.error(
            "Loan Rejected ❌"
        )

import streamlit as st
import pickle
import pandas as pd
import numpy as np

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
<style>

/* Background */
.main {
    background: linear-gradient(135deg, #1f1c2c, #928dab);
}

/* Title */
h1 {
    color: #00E5FF;
    text-align: center;
}

/* Fix label background */
label {
    background-color: transparent !important;
    color: white !important;
    font-weight: 500;
}

/* Input boxes */
.stNumberInput input {
    background-color: #2b2b3c;
    color: white;
}

div[data-baseweb="select"] {
    background-color: #2b2b3c !important;
    border-radius: 8px;
    color: white !important;
}

/* Predict Button */
.stButton>button {
    background: linear-gradient(90deg, #2563EB, #3B82F6);
    color: white;
    font-size: 18px;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-weight: 600;
    transition: 0.3s ease;
}

/* Hover effect */
.stButton>button:hover {
    background: linear-gradient(90deg, #1D4ED8, #2563EB);
    transform: scale(1.02);
    box-shadow: 0px 0px 10px rgba(37, 99, 235, 0.6);
}

/* Result Box */
.result-box {
    background: linear-gradient(90deg, #00FFA3, #00E5FF);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    color: black;
    margin-top: 20px;
}

/* Confidence text */
.confidence-text {
    color: white;
    margin-top: 10px;
    font-size: 16px;
}

/* Footer */
.footer {
    text-align: center;
    color: #cccccc;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.title("🚗 Car Price Prediction App")

# -----------------------------
# Load model & columns
# -----------------------------
model = pickle.load(open("model/model.pkl", "rb"))
columns = pickle.load(open("model/columns.pkl", "rb"))

# -----------------------------
# User Inputs
# -----------------------------
st.header("Enter Car Details")

year = st.number_input("Year of Purchase", min_value=2000, max_value=2025, value=2015)
present_price = st.number_input("Present Price (in Lakhs)", min_value=0.0, value=5.0)
kms = st.number_input("KMs Driven", min_value=0, value=30000)

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel"])
seller = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner", [0, 1, 2, 3])

# -----------------------------
# Feature Engineering
# -----------------------------
car_age = max(0, 2025 - year)

fuel_diesel = 1 if fuel == "Diesel" else 0
seller_individual = 1 if seller == "Individual" else 0
transmission_manual = 1 if transmission == "Manual" else 0

# -----------------------------
# Prediction
# -----------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Predict Price"):

    input_dict = {
        'Present_Price': present_price,
        'Driven_kms': kms,
        'Owner': owner,
        'Car_Age': car_age,
        'Fuel_Type_Diesel': fuel_diesel,
        'Selling_type_Individual': seller_individual,
        'Transmission_Manual': transmission_manual
    }

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Main Prediction
    prediction = model.predict(input_df)

    # -----------------------------
    # Confidence Calculation
    # -----------------------------
    predictions = np.array([tree.predict(input_df)[0] for tree in model.estimators_])
    confidence = 100 - (np.std(predictions) * 10)
    confidence = max(0, min(100, confidence))

    # -----------------------------
    # Output
    # -----------------------------
    st.markdown(f"""
        <div class="result-box">
            💰 Estimated Price: ₹ {prediction[0]:.2f} Lakhs
        </div>
    """, unsafe_allow_html=True)

    # Confidence Display
    st.markdown(f"""
        <div class="confidence-text">
            📊 Confidence Score: {confidence:.2f}%
        </div>
    """, unsafe_allow_html=True)

    st.progress(int(confidence))

# -----------------------------
# Footer
# -----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="footer">Built with ❤️ by Tanishka Yadav</div>', unsafe_allow_html=True)
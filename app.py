import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Mobile Price Predictor", page_icon="📱", layout="centered")

# ---------------------------------------------------------------------------
# Load the saved model package
# ---------------------------------------------------------------------------
MODEL_CANDIDATES = [
    "models/mobile_price_model.pkl",
    "../models/mobile_price_model.pkl",
    "mobile_price_model.pkl",
]


@st.cache_resource
def load_package():
    base = os.path.dirname(os.path.abspath(__file__))
    for rel in MODEL_CANDIDATES:
        path = os.path.join(base, rel)
        if os.path.exists(path):
            return joblib.load(path)
    return None


package = load_package()
if package is None:
    st.error(
        "Model file not found. Save it from your notebook as "
        "`mobile_price_model.pkl` and put it in a `models/` folder next to app.py."
    )
    st.stop()

model = package["model"]
feature_columns = package["feature_columns"]


def options_for(prefix, saved_key):
    """Dropdown options: use the full list saved in the pickle if present,
    otherwise recover them from the one-hot column names."""
    if saved_key in package:
        return sorted(package[saved_key])
    return sorted(c[len(prefix):] for c in feature_columns if c.startswith(prefix))


brands = options_for("Brand_", "brands")
processors = options_for("Processor_", "processors")

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("📱 Mobile Price Predictor")
st.caption("Enter the phone specifications and the Random Forest model will estimate its price (₹).")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", brands)
    ram = st.selectbox("RAM (GB)", [1, 2, 3, 4, 6, 8, 12, 16], index=4)
    battery = st.number_input("Battery (mAh)", 1000, 10000, 5000, step=100)
    display = st.number_input("Display size (inch)", 3.0, 8.5, 6.5, step=0.1)
    camera = st.number_input("Main camera (MP)", 2, 250, 50, step=1)

with col2:
    processor = st.selectbox("Processor", processors)
    charging = st.number_input("Fast charging (W, 0 if none)", 0, 240, 33, step=1)
    ext_mem = st.number_input("External memory (GB, 0 if none)", 0, 2048, 256, step=64)
    android = st.number_input("Android version", 4.0, 16.0, 13.0, step=0.5)
    rating = st.slider("Rating", 1.0, 5.0, 4.3, step=0.1)

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if st.button("Predict price", type="primary"):
    # Start from an all-zero row with exactly the training columns, in order
    row = pd.DataFrame(0.0, index=[0], columns=feature_columns)

    numeric_values = {
        "Rating": rating,
        "Android_version": android,
        "Ram_GB": ram,
        "Battery_mAh": battery,
        "Display_inch": display,
        "Charge_W": charging,
        "Ext_Mem_GB": ext_mem,
        "Main_Cam_MP": camera,
    }
    for col, value in numeric_values.items():
        if col in row.columns:
            row.at[0, col] = value

    # One-hot columns. If the chosen value is not a column, it was the
    # category dropped by drop_first=True, so all zeros is correct.
    for col in (f"Brand_{brand}", f"Processor_{processor}"):
        if col in row.columns:
            row.at[0, col] = 1.0

    price = float(model.predict(row)[0])
    st.success(f"Estimated price: ₹ {price:,.0f}")
    st.caption("This is an estimate. The model's typical error on test data is a few thousand rupees.")

    with st.expander("Show the input sent to the model"):
        st.dataframe(row.loc[:, (row != 0).any()])
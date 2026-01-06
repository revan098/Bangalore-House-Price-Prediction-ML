import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

# ===================== CONFIG =====================
st.set_page_config(
    page_title="Bangalore House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ===================== LOAD ARTIFACTS =====================
model = pickle.load(open("bhsp.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# ===================== DATA =====================
# --- IMPORTANT: cloud will read from same folder
df = pd.read_csv("house_prices_bangalore.csv")

# basic cleaning
df["location"] = df["location"].astype(str).str.strip()

# log target ONLY for internal use
if "price_log" not in df.columns:
    df["price_log"] = np.log1p(df["price"])

# encoded dataframe for visual pages
df_encoded = pd.get_dummies(
    df,
    columns=["location", "furnishing", "property_type"],
    drop_first=True
)

# ===================== HEADER =====================
st.markdown(
    "<h1 style='text-align:center;'>🏠 Bangalore House Sales Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;color:gray;'>Machine Learning – Indian Real Estate Portfolio</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ===================== SIDEBAR =====================
st.sidebar.title("Pages")
page = st.sidebar.radio(
    "Navigate",
    ["Predict Price", "EDA Dashboard", "About"]
)

# =====================================================
# PAGE 1 — PREDICTION
# =====================================================
if page == "Predict Price":

    st.subheader("🧾 Enter What Kind of House You Want")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Size Features")
        area = st.number_input("Area (sqft)", 300, 5000, 1200)
        bhk = st.slider("Bedrooms (BHK)", 1, 5, 3)
        bath = st.slider("Bathrooms", 1, 5, 2)
        age = st.slider("Age (years)", 0, 25, 5)

    with col2:
        st.markdown("### Lifestyle Features")
        balcony = st.slider("Balcony", 0, 3, 1)
        parking = st.slider("Parking Slots", 0, 3, 1)

        furnishing = st.selectbox(
            "Furnishing",
            ["Semi Furnished", "Fully Furnished", "Unfurnished"]
        )

        property_type = st.selectbox(
            "Property Type",
            ["Apartment", "Villa", "Independent"]
        )

        location = st.selectbox("Bangalore Area", df["location"].unique())

    st.markdown("---")

    if st.button("📈 Predict Bangalore Selling Price"):

        # -------- EXACT FEATURE REBUILD --------
        input_df = pd.DataFrame(0, index=[0], columns=features)

        # numeric
        for c in ["area", "bhk", "bath", "balcony", "parking", "age"]:
            if c in input_df.columns:
                input_df[c] = locals()[c]

        # one–hot align
        for col in features:

            if col == f"location_{location}":
                input_df[col] = 1

            if col == f"furnishing_{furnishing}":
                input_df[col] = 1

            if col == f"property_type_{property_type}":
                input_df[col] = 1

        # predict
        pred = model.predict(scaler.transform(input_df))

        st.metric("Predicted Price (₹ Lakhs)", f"{float(pred[0]):.2f}")

        st.success(
            "This estimation helps Bengaluru buyers understand FAIR market value."
        )

# =====================================================
# PAGE 2 — EDA VISUALS
# =====================================================
if page == "EDA Dashboard":

    st.subheader("📊 Understanding Bangalore Market")

    # 1. CORRELATION HEATMAP
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(
        df[['area','bhk','bath','balcony','parking','age','price']].corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )
    st.pyplot(fig)

    st.info("📌 Insight: Area is strongest driver; age weakest.")

    # 2. AREA VS PRICE
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(df["area"], df["price"])
    ax.set_title("Area vs Price – Bangalore 🇮🇳")
    st.pyplot(fig)

    st.info("📌 Insight: Curve visible after 1800 sqft → non-linear behavior.")

    # 3. BHK VS PRICE
    fig, ax = plt.subplots(figsize=(6,3))
    sns.boxplot(x="bhk", y="price", data=df, ax=ax)
    st.pyplot(fig)

    st.success("📌 Insight: 3→4 BHK gives strong Bengaluru premium.")

# =====================================================
# PAGE 3 — ABOUT
# =====================================================
if page == "About":

    st.subheader("About This Portfolio")

    st.markdown("""
    ✔ Bangalore domain awareness 🇮🇳  
    ✔ IQR outlier handling  
    ✔ categorical one-hot alignment  
    ✔ Streamlit dashboard  
    ✔ Business communication  
    """)

    st.markdown("---")

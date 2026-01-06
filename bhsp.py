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
df = pd.read_csv("house_prices_bangalore.csv")

# basic cleaning
df["location"] = df["location"].astype(str).str.strip()

# log target ONLY for internal prediction
if "price_log" not in df.columns:
    df["price_log"] = np.log1p(df["price"])

# encoded dataframe for visual pages
df_encoded = pd.get_dummies(
    df,
    columns=["location", "furnishing", "property_type"],
    drop_first=True
)

# ===================== HEADER =====================
st.markdown("<h1 style='text-align:center;'>🏠 Bangalore House Sales Prediction</h1>",
            unsafe_allow_html=True)

st.markdown("<p style='text-align:center;color:gray;'>Machine Learning – Indian Real Estate Portfolio</p>",
            unsafe_allow_html=True)

st.markdown("---")

# ===================== SIDEBAR =====================
st.sidebar.title("Pages")
page = st.sidebar.radio(
    "Navigate",
    ["Predict Price", "Bangalore Insights", "EDA Dashboard", "About"]
)

# =====================================================
# PAGE 1 — PREDICTION (HR ATTRACTIVE ALIGNMENT)
# =====================================================
if page == "Predict Price":

    st.subheader("🧾 Enter What Kind of House You Want to Live In")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Size Related")
        area = st.number_input("Area (sqft)", 300, 5000, 1200)
        bhk = st.slider("Bedrooms (BHK)", 1, 5, 3)
        bath = st.slider("Bathrooms", 1, 5, 2)
        age = st.slider("Age (years)", 0, 25, 5)

    with col2:
        st.markdown("### Lifestyle")
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

    # prediction logic
    if st.button("📈 Predict Bangalore Selling Price"):

        # EXACT FEATURE REBUILD
        input_df = pd.DataFrame(0, index=[0], columns=features)

        # numeric fields
        for c in ["area", "bhk", "bath", "balcony", "parking", "age"]:
            if c in input_df.columns:
                input_df[c] = locals()[c]

        # one-hot alignment
        for col in features:

            if col.startswith("location_"):
                input_df[col] = 1 if col == f"location_{location}" else 0

            if col.startswith("furnishing_"):
                input_df[col] = 1 if col == f"furnishing_{furnishing}" else 0

            if col.startswith("property_type_"):
                input_df[col] = 1 if col == f"property_type_{property_type}" else 0

        pred_log = model.predict(scaler.transform(input_df))

        st.markdown("## 📊 Result – Bangalore")
        st.metric("Predicted Price (₹ Lakhs)", f"{float(pred_log[0]):.2f}")

        st.success(
            "This dashboard helps Indian Bangalore buyers estimate FAIR market price and reduce emotional negotiation time."
        )

        st.info("HR Insight: Shows ML + Indian domain + deployment in one portfolio.")

# =====================================================
# PAGE 2 — VISUAL PREMIUM ANALYSIS
# =====================================================
if page == "Bangalore Insights":

    st.subheader("📊 Bangalore Market Premiums")

    # --- chart 1
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(df["area"], df["price")
    ax.set_title("Area vs Price – Bengaluru 🇮🇳")
    st.pyplot(fig)

    st.info("📌 Insight: Larger sqft in Bangalore directly increases price; curve visible after ~1800 sqft → non-linear behavior.")

    # --- chart 2
    fig, ax = plt.subplots(figsize=(6,3))
    sns.boxplot(x="bhk", y="price", data=df, palette="Set1", ax=ax)
    st.pyplot(fig)

    st.info("📌 Insight: 3 → 4 BHK gives strong Bengaluru premium; 1–2 BHK mostly budget zones.")

    # --- chart 3
    fig, ax = plt.subplots(figsize=(5,3))
    sns.boxplot(x="furnishing", y="price", data=df, palette="Set2", ax=ax)
    st.pyplot(fig)

    st.success("📌 Insight: Fully furnished villas in Bangalore carry lifestyle premium 🇮🇳.")

# =====================================================
# PAGE 3 — EXTENDED EDA VISUALS
# =====================================================
if page == "EDA Dashboard":

    st.subheader("📐 Understanding Bangalore Data Visually")

    # --- correlation
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(df[['area','bhk','bath','balcony','parking','age','price']].corr(),
                annot=True,
                cmap="coolwarm",
                ax=ax)
    st.pyplot(fig)

    st.info("📌 Insight: Area top driver; bhk & bath inter-correlated; age weak.")

    # --- price distribution
    fig, ax = plt.subplots(figsize=(6,3))
    sns.histplot(df["price"], kde=True, ax=ax, palette="Set1")
    st.pyplot(fig)

    st.info("📌 Insight: Bangalore price is right-skewed; log transform stabilizes Indian ₹ predictions.")

    # --- demand count
    fig, ax = plt.subplots(figsize=(6,4))
    df["location"].value_counts().head(12).plot(kind="bar")
    ax.set_title("Top Bangalore Transaction Areas")
    st.pyplot(fig)

    st.success("📌 Insight: Specific Bengaluru neighborhoods dominate market behavior 🇮🇳.")

# =====================================================
# PAGE 4 — ABOUT
# =====================================================
if page == "About":

    st.subheader("About This Portfolio")

    st.markdown("""
    ✔ Bangalore domain awareness  
    ✔ IQR outlier handling  
    ✔ categorical one-hot alignment  
    ✔ Streamlit multi-page dashboard  
    ✔ Indian real-estate communication 🇮🇳  
    """)

    st.markdown("---")
    st.markdown("<p style='text-align:center;color:gray;'>Built with HardWork | Black Leg Sanji | ML Portfolio</p>",
                unsafe_allow_html=True)


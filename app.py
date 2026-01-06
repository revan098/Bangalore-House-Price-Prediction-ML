import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns

from sklearn.preprocessing import StandardScaler

# ===================== CONFIG =====================
st.set_page_config(
    page_title="Bangalore House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ===================== LOAD ARTIFACTS =====================
# these must exist in same GitHub folder
model = pickle.load(open("bhsp.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# ===================== DATA =====================
df = pd.read_csv("house_prices_bangalore.csv")

# basic cleaning
df["location"] = df["location"].astype(str).str.strip()

# log target ONLY for internal use
if "price_log" not in df.columns:
    df["price_log"] = np.log1p(df["price"])

# encoded dataframe for visualization copies
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
    "<p style='text-align:center;color:gray;'>Machine Learning Portfolio – Indian Real Estate Context</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ===================== SIDEBAR =====================
st.sidebar.title("Navigate")
page = st.sidebar.radio(
    "Pages",
    ["Predict Price", "Bangalore Insights", "EDA Dashboard", "About"]
)

# =====================================================
# PAGE 1 — PREDICT PRICE
# =====================================================
if page == "Predict Price":

    st.subheader("🧾 Enter House Details")

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

    if st.button("📈 Predict Bangalore Selling Price"):

        # EXACT FEATURE RECONSTRUCTION
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

        # Predict
        pred = model.predict(scaler.transform(input_df))

        st.markdown("## 📊 Result – Bangalore")
        st.metric("Predicted Price (₹ Lakhs)", f"{float(pred[0]):.2f}")

        st.success(
            "This estimation helps Bengaluru buyers negotiate fair prices and shows HR real business impact."
        )

        st.info("HR Insight: Demonstrates ML + Indian domain + deployment skills.")

# =====================================================
# PAGE 2 — BANGALORE INSIGHTS
# =====================================================
if page == "Bangalore Insights":

    st.subheader("📊 Bangalore Market Premiums")

    # ---------- CHART 1: AREA VS PRICE ----------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["area"], df["price"], color="darkcyan")
    ax.set_title("Area vs Price – Bangalore 🇮🇳")
    ax.set_xlabel("Area (sqft)")
    ax.set_ylabel("Price (₹ Lakhs)")
    st.pyplot(fig)

    st.info(
        "📌 Insight: Larger sqft in Bangalore increases price sharply; curve visible after ~1800 sqft → non-linear behavior."
    )

    # ---------- CHART 2: BHK PREMIUM ----------
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.boxplot(x="bhk", y="price", data=df, palette="Set1", ax=ax)
    ax.set_title("BHK vs Price Premium – Bengaluru")
    st.pyplot(fig)

    st.info(
        "📌 Insight: 3 → 4 BHK houses in Bengaluru carry major configuration premium."
    )

    # ---------- CHART 3: FURNISHING ----------
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.boxplot(x="furnishing", y="price", data=df, palette="Set2", ax=ax)
    ax.set_title("Furnishing Impact – Bangalore")
    st.pyplot(fig)

    st.success(
        "📌 Insight: Fully furnished villas/apartments show lifestyle premium in Indian context 🇮🇳."
    )

    # ---------- CHART 4: LOCATION DEMAND ----------
    fig, ax = plt.subplots(figsize=(6, 4))
    df["location"].value_counts().head(10).plot(kind="bar", color="orchid")
    ax.set_title("Top Demand Areas – Bengaluru")
    ax.set_xlabel("Location")
    st.pyplot(fig)

    st.metric("Total Unique Locations", df["location"].nunique())

    st.info(
        "📌 Insight: Specific neighborhoods dominate Bangalore transactions and price behavior."
    )

# =====================================================
# PAGE 3 — EDA DASHBOARD
# =====================================================
if page == "EDA Dashboard":

    st.subheader("📐 Understanding Bangalore Data Visually")

    # ---------- CORRELATION ----------
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        df[['area','bhk','bath','balcony','parking','age','price']].corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )
    ax.set_title("Correlation Heat – Bengaluru")
    st.pyplot(fig)

    st.info(
        "📌 Insight: Area strongest driver; BHK & bathrooms inter-correlated; age weak."
    )

    # ---------- PRICE DISTRIBUTION ----------
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(df["price"], kde=True, color="navy", ax=ax)
    ax.set_title("Bangalore Price Distribution")
    st.pyplot(fig)

    st.info(
        "📌 Insight: Bengaluru prices are right-skewed; log transform stabilizes predictions."
    )

    # ---------- OUTLIERS ----------
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.boxplot(x="property_type", y="price", data=df, palette="coolwarm", ax=ax)
    ax.set_title("Outliers by Property Type – Bangalore")
    st.pyplot(fig)

    st.success(
        "📌 Insight: Villas and independent houses create extreme Bangalore outliers."
    )

    # ---------- LINEARITY PAIR ----------
    sns.pairplot(df[['area','bhk','bath','price']])
    st.pyplot(plt.gcf())

    st.info(
        "📌 Insight: Non-linear interactions visible → need Random Forest / XGB."
    )

# =====================================================
# PAGE 4 — ABOUT
# =====================================================
if page == "About":

    st.subheader("About This Portfolio")

    st.markdown("""
    ✔ Bangalore domain awareness 🇮🇳  
    ✔ IQR outlier handling  
    ✔ categorical one-hot alignment  
    ✔ Streamlit multi-page dashboard  
    ✔ Indian real-estate communication  
    ✔ Regression + Clustering portfolio  
    """)

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:gray;'>Built with HardWork | ML Portfolio</p>",
        unsafe_allow_html=True
    )


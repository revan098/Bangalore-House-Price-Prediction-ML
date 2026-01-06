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
    page_title="Bangalore House Sales Prediction",
    page_icon="🏠",
    layout="wide"
)

# ===================== LOAD =====================
model = pickle.load(open("bhsp.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# ===================== DATA =====================
df = pd.read_csv("house_prices_bangalore.csv")
df["location"] = df["location"].astype(str).str.strip()

# create target log only for y (not used in X)
if "price_log" not in df.columns:
    df["price_log"] = np.log1p(df["price"])

# One-hot for visual pages
df_encoded = pd.get_dummies(
    df,
    columns=["location", "furnishing", "property_type"],
    drop_first=True
)

# ===================== SIDEBAR =====================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Pages",
    ["Predict Price", "Bangalore Insights", "EDA Dashboard", "About"]
)

st.markdown(
    "<h1 style='text-align:center;color:#e74c3c;'>🏠 Bangalore House Sales Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;color:gray;'>Machine Learning Regression | Indian Real Estate</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# =====================================================
# PAGE 1 — PREDICTION
# =====================================================
if page == "Predict Price":

    st.subheader("🧾 Enter What Kind of You House Want to live")

    card1, card2 = st.columns(2)

    with card1:
        st.markdown("### Size Features")
        area = st.number_input("Area (sqft)", 300, 5000, 1200)
        bhk = st.slider("Bedrooms (BHK)", 1, 5, 3)
        bath = st.slider("Bathrooms", 1, 5, 2)

    with card2:
        st.markdown("### Lifestyle Features")
        balcony = st.slider("Balcony", 0, 3, 1)
        parking = st.slider("Parking Slots", 0, 3, 1)
        age = st.slider("Age (years)", 0, 25, 5)

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

        input_df = pd.DataFrame(0, index=[0], columns=features)

        # numeric safe
        for c in ["area", "bhk", "bath", "balcony", "parking", "age"]:
            if c in input_df.columns:
                input_df[c] = locals()[c]

        # exact one-hot match
        for col in features:

            if col.startswith("location_"):
                input_df[col] = 1 if col == f"location_{location}" else 0

            if col.startswith("furnishing_"):
                input_df[col] = 1 if col == f"furnishing_{furnishing}" else 0

            if col.startswith("property_type_"):
                input_df[col] = 1 if col == f"property_type_{property_type}" else 0

        pred = model.predict(scaler.transform(input_df))
        # inside button
        #pred = model.predict(scaler.transform(input_df))

        st.markdown("## 📊 Result – Bangalore")
        st.metric("Predicted Price (₹ Lakhs)", f"{float(pred[0]):.2f}")




        st.success(
            "This dashboard assists Indian buyers to estimate FAIR Bangalore market price and reduce negotiation time."
        )

        st.info("HR Insight: Demonstrates ML + Indian domain + deployment skills.")

# =====================================================
# PAGE 2 — KEY BUSINESS VISUALS
# =====================================================
if page == "Bangalore Insights":

    st.subheader("📊 Key Bangalore Visualizations")

    # ---------- AREA VS PRICE ----------
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(x="area", y="price", data=df, ax=ax, cmap="coolwarm")
    st.pyplot(fig)

    st.info("📌 Insight: Area shows steep positive growth; curved after 1800 sqft.")

    # ---------- AGE TREND ----------
    fig, ax = plt.subplots(figsize=(6,3))
    sns.boxplot(x="age", y="price", data=df, palette="coolwarm", ax=ax)
    st.pyplot(fig)

    st.info("📌 Insight: Older Bangalore houses lose premium → depreciation visible.")

    # ---------- PARKING ----------
    fig, ax = plt.subplots(figsize=(5,3))
    sns.boxplot(x="parking", y="price", data=df, palette="Set1", ax=ax)
    st.pyplot(fig)

    st.info("📌 Insight: Parking gives additional Bangalore premium.")

    # ---------- LOCATION DEMAND ----------
    fig, ax = plt.subplots(figsize=(6,4))
    df["location"].value_counts().head(10).plot(kind="bar")
    ax.set_title("Top Bangalore Demand Areas")
    st.pyplot(fig)

    st.success("📌 Insight: Specific neighborhoods dominate Bengaluru prices 🇮🇳.")

# =====================================================
# PAGE 3 — EXTENDED EDA DASHBOARD
# =====================================================
if page == "EDA Dashboard":

    st.subheader("📐 More Visual Understanding – Bangalore")

    # ---------- CORRELATION ----------
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(
        df[['area','bhk','bath','balcony','parking','age','price']].corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )
    st.pyplot(fig)

    st.info("📌 Insight: Confirms area + BHK + bath strongly related.")

    # ---------- DISTRIBUTION ----------
    fig, ax = plt.subplots(figsize=(5,3))
    sns.histplot(df["price"], kde=True, palette="Set1", ax=ax)
    st.pyplot(fig)

    st.info("📌 Insight: Bangalore price is right-skewed; need log transform.")

    # ---------- PAIR LINEARITY ----------
    sns.pairplot(df[['area','bhk','bath','price']])
    st.pyplot(plt.gcf())

    st.success("📌 Insight: Non-linear relations justify tree models.")

# =====================================================
# PAGE 4 — ABOUT
# =====================================================
if page == "About":

    st.subheader("About This Portfolio")

    st.markdown("""
    ✔ Bangalore domain  
    ✔ IQR Outliers  
    ✔ VIF validation  
    ✔ Streamlit dashboard  
    ✔ Indian Rupees RMSE  
    ✔ Regression + Clustering  
    """)

    st.markdown("---")
    st.markdown("<p style='text-align:center;color:gray;'>Build with HardWork | Black Leg Sanji  | ML Portfolio</p>", unsafe_allow_html=True)


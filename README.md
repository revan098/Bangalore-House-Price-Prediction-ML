# Bangalore-House-Price-Prediction-ML
🏠 End-to-End Bangalore House Price Prediction using Machine Learning Regression with IQR Outlier Handling, VIF Validation, and Streamlit Multi-Page Dashboard (Indian Real-Estate Context).

# 🏠 Customer House Price Prediction — Bangalore

## 📌 Project Overview
The Bangalore housing market is highly dependent on location, property size (BHK), furnishing, and age of the building.  
This portfolio project builds an advanced Machine Learning regression pipeline to estimate **selling price in ₹ Lakhs** based on realtime Indian real-estate behavior.

## 🎯 Business Objective (HR View)
- Reduce price negotiation time for buyers and sellers  
- Provide data-driven fair valuation for Bengaluru neighborhoods  
- Understand premiums for Villas vs Apartments  
- Quantify depreciation for older properties

## 🧾 Dataset Features
Used columns from dataset:

- area (sqft)  
- location  
- bhk  
- bath  
- balcony  
- parking  
- furnishing  
- property_type  
- age  
- price (target)

## ⚙ Preprocessing & Validation
✔ Handling Missing Values using median  
✔ Converting categorical text with One-Hot Encoding  
✔ Outlier Detection using IQR  
✔ Multicollinearity check using VIF  
✔ Linearity and Normal distribution validation

## 🤖 Models Used
- Linear Regression – baseline interpretable  
- Random Forest – robust for Bangalore non-linear interactions  
- XGBoost – production-level performance

## 📊 Deployment
Built a Streamlit dashboard with:

✔ Multiple pages  
✔ Indian rupees metrics  
✔ Area vs Price charts  
✔ BHK premium visuals  
✔ Furnishing impact  
✔ Location demand analysis

## 🧠 Final Solution
The app predicts fair Bangalore selling price considering:

> Location dominance + configuration premium + lifestyle premium + age depreciation.

## 📂 Artifacts
- bhsp.pkl  
- scaler.pkl  
- features.pkl  
- house_prices_bangalore.csv  
- app.py  
- requirements.txt

## ❤️ Outcome
Demonstrates to HR:

✔ Machine Learning  
✔ Indian domain understanding 🇮🇳  
✔ Deployment skills  
✔ Business communication

---

### Author
**Prathap — Aspiring Data Analyst | Machine Learning | Indian Real Estate**

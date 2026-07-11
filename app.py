import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Load model
model = joblib.load("model/xgboost_model.pkl")
features = joblib.load("model/features.pkl")


# Data preprocessing
def preprocess_data(df):

    # Replace Select with missing values
    df = df.replace("Select", np.nan)

    # Remove columns with more than 50% missing values
    missing = (df.isnull().sum() / len(df)) * 100
    drop_cols = missing[missing > 50].index
    df = df.drop(drop_cols, axis=1)

    # Remove ID columns
    remove_cols = ["Prospect ID", "Lead Number"]

    for col in remove_cols:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # Fill missing values
    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())

        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    # Remove target column
    if "Converted" in df.columns:
        df = df.drop("Converted", axis=1)

    # Convert categorical columns
    cat_cols = df.select_dtypes(include="object").columns

    df = pd.get_dummies(
        df,
        columns=cat_cols,
        drop_first=True
    )

    # Add missing columns
    for col in features:
        if col not in df.columns:
            df[col] = 0

    # Keep only required columns
    df = df[features]

    return df


# Lead category
def categorize(score):

    if score >= 70:
        return "Hot"

    elif score >= 40:
        return "Warm"

    else:
        return "Cold"


# Streamlit App
st.title("Lead Scoring App")

st.write("Upload a CSV file to generate lead scores.")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type="csv"
)
if uploaded_file is not None:

    # Read CSV file
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # Process data
    df_processed = preprocess_data(df.copy())

    # Predict conversion probability
    y_prob = model.predict_proba(df_processed)[:, 1]

    # Lead score (0-100)
    lead_scores = np.round(y_prob * 100, 2)

    # Lead category
    category = []

    for score in lead_scores:
        category.append(categorize(score))

    # Final result
    results = pd.DataFrame({
        "Lead Score": lead_scores,
        "Category": category,
        "Conversion Probability": np.round(y_prob, 3)
    })

    # Highest score first
    results = results.sort_values(
        by="Lead Score",
        ascending=False
    ).reset_index(drop=True)

    st.subheader("Lead Scoring Results")
    st.dataframe(results)

    # Summary
    st.subheader("Summary")

    hot = len(results[results["Category"] == "Hot"])
    warm = len(results[results["Category"] == "Warm"])
    cold = len(results[results["Category"] == "Cold"])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Hot Leads", hot)

    with col2:
        st.metric("Warm Leads", warm)

    with col3:
        st.metric("Cold Leads", cold)

    # Graph
    st.subheader("Lead Category")

    chart = results["Category"].value_counts()
    st.bar_chart(chart)

    # Download result
    csv = results.to_csv(index=False)

    st.download_button(
        label="Download Results",
        data=csv,
        file_name="lead_scores.csv",
        mime="text/csv"
    )

else:

    st.info("Upload a CSV file to start prediction.")
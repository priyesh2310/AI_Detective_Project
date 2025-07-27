import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import google.generativeai as genai
from dotenv import load_dotenv
import os
from PIL import Image

# Load Gemini API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Function to get Gemini "detective-style" insights
def detective_analysis(summary):
    prompt = f"""
    You're a data detective analyzing a suspicious dataset with unknown origins.
    Summary: {summary}
    Your mission: uncover 3 strange behaviors or anomalies—patterns that defy logic, sudden spikes,
    hidden relationships, or missing information.
    Deliver a thrilling 150-word case report, rich with suspense and investigative flavor.
    Think noir-style drama, vivid metaphors, and a twist at the end.
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

# Create dataset summary
def generate_summary(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "numeric_columns": list(df.select_dtypes(include=['number']).columns)
    }

# Simple chart for first numeric column
def create_visual(df):
    buffer = io.BytesIO()
    num_cols = df.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        plt.figure(figsize=(6,4))
        sns.histplot(df[num_cols[0]], bins=20, kde=True)
        plt.title(f"Distribution of {num_cols[0]}")
        plt.tight_layout()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
    return buffer

# Streamlit App
st.title("Detective Mode: AI Data Analyst 🕵️")
st.write("Upload a CSV or Excel file, and let Gemini narrate the mysteries within...")

uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

if uploaded_file:
    # Load data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Dataset summary
    summary = generate_summary(df)
    st.write("### Dataset Summary")
    st.json(summary)

    # AI noir report
    st.subheader("🕵️ Detective Report")
    noir_report = detective_analysis(summary)
    st.write(noir_report)

    # Show visualization
    img_buf = create_visual(df)
    if img_buf:
        image = Image.open(img_buf)
        st.image(image, caption="Suspicious Distribution")


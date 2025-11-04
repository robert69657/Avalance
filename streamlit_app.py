# ===============================
# 🧠 GenAI Sentiment Dashboard (Fixed Average Sentiment Chart)
# ===============================

import streamlit as st
import pandas as pd
import re
import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
import altair as alt

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.warning("⚠️ No GEMINI_API_KEY found in .env — please add it to use sentiment analysis.")

# -----------------------------
# Helper Functions
# -----------------------------
def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def analyze_sentiment_gemini(text):
    """Analyze sentiment using Gemini with fallback to Neutral."""
    if not GEMINI_API_KEY:
        return "Neutral", 0.0

    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    Analyze the sentiment of this customer review text.
    Classify it as: Positive, Negative, or Neutral.
    Provide a sentiment score between -1 and +1.
    Review: {text}
    Respond ONLY in valid JSON, like:
    {{
      "sentiment": "Positive",
      "score": 0.85
    }}
    """

    try:
        response = model.generate_content(prompt)
        result_text = response.text.strip()

        # Extract JSON safely
        if not result_text.startswith("{"):
            start = result_text.find("{")
            end = result_text.rfind("}") + 1
            if start != -1 and end != -1:
                result_text = result_text[start:end]
            else:
                return "Neutral", 0.0

        result = json.loads(result_text)
        sentiment = result.get("sentiment", "Neutral").title()
        score = float(result.get("score", 0))
        return sentiment, score

    except Exception:
        return "Neutral", 0.0

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="GenAI Sentiment Dashboard", layout="wide")
st.title("🧠 Ceniza-Butawan:GenAI Sentiment Dashboard")
st.write("Analyze customer reviews using **Google Gemini**, clean data, and visualize sentiment insights.")

col1, col2, col3 = st.columns(3)

# 📥 Load Dataset
with col1:
    if st.button("📥 Load Dataset"):
        try:
            st.session_state["df"] = pd.read_csv("data/customer_reviews.csv")
            st.success("✅ Dataset loaded successfully!")
        except FileNotFoundError:
            st.error("❌ Could not find 'customer_reviews.csv' — make sure it exists in your project folder.")

# 🧹 Clean Reviews
with col2:
    if st.button("🧹 Clean Reviews"):
        if "df" in st.session_state:
            st.session_state["df"]["CLEANED_SUMMARY"] = st.session_state["df"]["SUMMARY"].apply(clean_text)
            st.success("✨ Reviews cleaned successfully!")
        else:
            st.warning("⚠️ Please load the dataset first.")

# 💬 Analyze Sentiment
with col3:
    if st.button("💬 Analyze Sentiment (Gemini)"):
        if "df" in st.session_state and "CLEANED_SUMMARY" in st.session_state["df"].columns:
            st.info("🔍 Analyzing sentiments... please wait ⏳")
            sentiments, scores = [], []
            progress = st.progress(0)
            df_len = len(st.session_state["df"])

            for i, review in enumerate(st.session_state["df"]["CLEANED_SUMMARY"]):
                sentiment, score = analyze_sentiment_gemini(review)
                sentiments.append(sentiment)
                scores.append(score)
                progress.progress((i + 1) / df_len)

            st.session_state["df"]["SENTIMENT"] = sentiments
            st.session_state["df"]["SENTIMENT_SCORE"] = scores
            st.session_state["analysis_done"] = True
            progress.empty()
            st.success("✅ Sentiment analysis completed successfully!")
        else:
            st.warning("⚠️ Please clean the reviews first.")

# -----------------------------
# Display Dataset and Charts
# -----------------------------
if "df" in st.session_state:
    df = st.session_state["df"]

    st.subheader("📋 Dataset Preview")
    st.dataframe(df, use_container_width=True)

    # Product Filter
    st.subheader("🔍 Filter by Product")
    products = ["All Products"] + sorted(df["PRODUCT"].unique())
    product = st.selectbox("Choose a product", products)

    filtered_df = df if product == "All Products" else df[df["PRODUCT"] == product]
    st.dataframe(filtered_df, use_container_width=True)

    # -----------------------------
    # Charts AFTER Analysis
    # -----------------------------
    if st.session_state.get("analysis_done", False):

        # 📊 Average Sentiment Score by Product
        if "SENTIMENT_SCORE" in df.columns and "PRODUCT" in df.columns:
            st.subheader("📊 Average Sentiment Score by Product")

            avg_scores = (
                df.groupby("PRODUCT")["SENTIMENT_SCORE"]
                .mean()
                .reset_index()
                .rename(columns={"PRODUCT": "Product", "SENTIMENT_SCORE": "Average_Score"})
            )

            if not avg_scores.empty:
                chart_avg = (
                    alt.Chart(avg_scores)
                    .mark_bar(size=60)
                    .encode(
                        x=alt.X("Product:N", title="Product"),
                        y=alt.Y("Average_Score:Q", title="Average Sentiment Score", scale=alt.Scale(domain=[-1, 1])),
                        color=alt.Color("Average_Score:Q", scale=alt.Scale(domain=[-1, 0, 1], range=["#F44336", "#FFC107", "#4CAF50"])),
                        tooltip=["Product", "Average_Score"]
                    )
                    .properties(title="Average Sentiment Score by Product", height=400)
                )
                st.altair_chart(chart_avg, use_container_width=True)
            else:
                st.info("ℹ️ No sentiment scores available to plot.")

        # 📈 Sentiment Distribution
        if "SENTIMENT" in df.columns:
            st.subheader("📈 Sentiment Distribution")

            sentiment_counts = (
                df["SENTIMENT"]
                .value_counts()
                .reset_index()
            )
            sentiment_counts.columns = ["Sentiment", "Count"]

            if not sentiment_counts.empty:
                sentiment_counts["Sentiment"] = sentiment_counts["Sentiment"].astype(str)
                sentiment_counts["Count"] = sentiment_counts["Count"].astype(int)

                chart_dist = (
                    alt.Chart(sentiment_counts)
                    .mark_bar(size=60)
                    .encode(
                        x=alt.X("Sentiment:N", sort=["Positive", "Neutral", "Negative"]),
                        y=alt.Y("Count:Q", title="Number of Reviews"),
                        color=alt.Color(
                            "Sentiment:N",
                            scale=alt.Scale(
                                domain=["Positive", "Neutral", "Negative"],
                                range=["#4CAF50", "#FFC107", "#F44336"]
                            )
                        ),
                        tooltip=["Sentiment", "Count"]
                    )
                    .properties(title="Sentiment Distribution", height=400)
                )

                st.altair_chart(chart_dist, use_container_width=True)
            else:
                st.info("ℹ️ No sentiment data available to plot.")

st.caption("💡 Built with Streamlit, Pandas, Altair & Google Gemini API (Average Chart Fixed)")
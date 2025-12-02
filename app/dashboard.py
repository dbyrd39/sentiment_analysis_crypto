import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Crypto Social Media Sentiment Dashboard")

df = pd.read_csv("data/processed/daily_sentiment.csv")
coins = df["coin"].unique()

selected_coin = st.selectbox("Select coin:", coins)
filtered = df[df["coin"] == selected_coin]

fig = px.line(filtered, x="date", y="mean_sentiment",
              title=f"{selected_coin} Daily Sentiment Score")

st.plotly_chart(fig)

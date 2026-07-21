
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="GDP Growth Forecast Dashboard", layout="wide")

# Load data
historical = pd.read_csv("PowerBI_GDP_Historical_Forecast.csv")
evaluation = pd.read_csv("PowerBI_Model_Evaluation.csv")
ranking = pd.read_csv("PowerBI_Country_Growth_Ranking.csv")
test = pd.read_csv("PowerBI_Test_Predictions.csv")

st.title("📈 GDP Growth Forecast Dashboard")

country = st.sidebar.selectbox(
    "Select Country",
    sorted(historical["Country"].unique())
)

data = historical[historical["Country"] == country]

fig = px.line(
    data,
    x="Year",
    y="GDP_Value",
    color="Data_Type",
    title=f"{country} GDP Forecast"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Model Evaluation")
st.dataframe(evaluation[evaluation["Country"] == country])

st.subheader("Growth Ranking")
st.dataframe(ranking)

st.subheader("Test Predictions")
st.dataframe(test[test["Country"] == country])

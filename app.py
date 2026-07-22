%%writefile app.py

import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Economic Growth Forecasting",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            color: #666666;
            margin-bottom: 30px;
        }

        .accuracy-box {
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            font-size: 22px;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data
def load_data():

    files_required = [
        "PowerBI_GDP_Historical_Forecast.csv",
        "PowerBI_Model_Evaluation.csv",
        "PowerBI_Country_Growth_Ranking.csv",
        "PowerBI_Test_Predictions.csv"
    ]

    missing_files = [
        file_name
        for file_name in files_required
        if not os.path.exists(file_name)
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing files: " + ", ".join(missing_files)
        )

    gdp = pd.read_csv(
        "PowerBI_GDP_Historical_Forecast.csv"
    )

    evaluation = pd.read_csv(
        "PowerBI_Model_Evaluation.csv"
    )

    ranking = pd.read_csv(
        "PowerBI_Country_Growth_Ranking.csv"
    )

    test_predictions = pd.read_csv(
        "PowerBI_Test_Predictions.csv"
    )

    datasets = [
        gdp,
        evaluation,
        ranking,
        test_predictions
    ]

    for dataset in datasets:

        dataset.columns = (
            dataset.columns
            .str.strip()
            .str.replace(" ", "_")
        )

        if "Country" in dataset.columns:
            dataset["Country"] = (
                dataset["Country"]
                .astype(str)
                .str.strip()
            )

    return gdp, evaluation, ranking, test_predictions


try:
    (
        gdp_data,
        evaluation_data,
        ranking_data,
        test_data
    ) = load_data()

except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()


# =========================================================
# COLUMN DETECTION
# =========================================================

def find_column(dataframe, possible_names):

    for name in possible_names:
        if name in dataframe.columns:
            return name

    return None


gdp_column = find_column(
    gdp_data,
    [
        "GDP_Value",
        "GDP",
        "GDP_Amount"
    ]
)

data_type_column = find_column(
    gdp_data,
    [
        "Data_Type",
        "Type",
        "Category"
    ]
)

growth_column = find_column(
    gdp_data,
    [
        "Annual_Growth_Percent",
        "Annual_Growth",
        "Growth_Percent"
    ]
)

lower_column = find_column(
    gdp_data,
    [
        "Lower_Estimate",
        "Lower_Bound",
        "yhat_lower"
    ]
)

upper_column = find_column(
    gdp_data,
    [
        "Upper_Estimate",
        "Upper_Bound",
        "yhat_upper"
    ]
)

mape_column = find_column(
    evaluation_data,
    [
        "MAPE_Percent",
        "MAPE",
        "MAPE_(%)"
    ]
)

cagr_column = find_column(
    ranking_data,
    [
        "CAGR_Percent",
        "CAGR",
        "CAGR_(%)"
    ]
)

gdp_2025_column = find_column(
    ranking_data,
    [
        "GDP_2025",
        "GDP2025"
    ]
)

gdp_2030_column = find_column(
    ranking_data,
    [
        "Predicted_GDP_2030",
        "GDP_2030",
        "Predicted_2030"
    ]
)

actual_column = find_column(
    test_data,
    [
        "Actual_GDP",
        "Actual",
        "GDP_Actual"
    ]
)

predicted_column = find_column(
    test_data,
    [
        "Predicted_GDP",
        "Predicted",
        "GDP_Predicted"
    ]
)

test_lower_column = find_column(
    test_data,
    [
        "Lower_Estimate",
        "Lower_Bound",
        "yhat_lower"
    ]
)

test_upper_column = find_column(
    test_data,
    [
        "Upper_Estimate",
        "Upper_Bound",
        "yhat_upper"
    ]
)


if gdp_column is None:
    st.error(
        "GDP column not found in "
        "PowerBI_GDP_Historical_Forecast.csv"
    )
    st.stop()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def format_gdp(value):

    if value is None or pd.isna(value):
        return "N/A"

    value = float(value)

    if abs(value) >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"

    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"

    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"

    return f"${value:,.2f}"


def get_accuracy_category(mape):

    if mape is None or pd.isna(mape):
        return "Unavailable", "#808080"

    if mape < 5:
        return "Excellent", "#2E8B57"

    if mape < 10:
        return "Good", "#1E90FF"

    if mape < 20:
        return "Moderate", "#D4A017"

    return "Poor", "#DC143C"


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Dashboard Navigation")

countries = sorted(
    gdp_data["Country"]
    .dropna()
    .unique()
)

selected_country = st.sidebar.selectbox(
    "Select Country",
    countries
)

selected_page = st.sidebar.radio(
    "Select Dashboard Page",
    [
        "Economic Growth Overview",
        "Country Forecast Analysis",
        "Model Evaluation"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    Project: Economic Growth Forecasting

    Model: Prophet

    Historical period: 1975–2025

    Forecast period: 2026–2030
    """
)


# =========================================================
# FILTER DATA
# =========================================================

country_gdp = gdp_data[
    gdp_data["Country"] == selected_country
].copy()

country_evaluation = evaluation_data[
    evaluation_data["Country"] == selected_country
].copy()

country_ranking = ranking_data[
    ranking_data["Country"] == selected_country
].copy()

country_test = test_data[
    test_data["Country"] == selected_country
].copy()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    'Economic Growth Forecasting Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'GDP analysis and Prophet forecasting for '
    'China, Germany, India, Japan and the United States'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# PAGE 1: ECONOMIC GROWTH OVERVIEW
# =========================================================

if selected_page == "Economic Growth Overview":

    st.header("Economic Growth Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Countries Analysed",
        gdp_data["Country"].nunique()
    )

    col2.metric(
        "Historical Period",
        "1975–2025"
    )

    col3.metric(
        "Forecast Period",
        "2026–2030"
    )

    col4.metric(
        "Forecasting Model",
        "Prophet"
    )

    st.markdown("---")

    st.subheader("Historical and Forecast GDP")

    overview_data = gdp_data[
        gdp_data["Year"] >= 2000
    ].copy()

    if data_type_column:

        fig_overview = px.line(
            overview_data,
            x="Year",
            y=gdp_column,
            color="Country",
            line_dash=data_type_column,
            markers=True,
            labels={
                gdp_column: "GDP",
                data_type_column: "Data Type"
            }
        )

    else:

        fig_overview = px.line(
            overview_data,
            x="Year",
            y=gdp_column,
            color="Country",
            markers=True,
            labels={
                gdp_column: "GDP"
            }
        )

    fig_overview.add_vline(
        x=2025.5,
        line_dash="dash",
        annotation_text="Forecast begins"
    )

    fig_overview.update_layout(
        hovermode="x unified",
        legend_title_text=""
    )

    st.plotly_chart(
        fig_overview,
        use_container_width=True
    )

    col_left, col_right = st.columns(2)

    with col_left:

        st.subheader("Forecast CAGR Ranking")

        if cagr_column:

            ranking_chart = ranking_data.sort_values(
                cagr_column,
                ascending=True
            )

            fig_ranking = px.bar(
                ranking_chart,
                x=cagr_column,
                y="Country",
                orientation="h",
                text=cagr_column,
                labels={
                    cagr_column: "CAGR (%)"
                }
            )

            fig_ranking.update_traces(
                texttemplate="%{text:.2f}%"
            )

            fig_ranking.update_layout(
                showlegend=False
            )

            st.plotly_chart(
                fig_ranking,
                use_container_width=True
            )

        else:
            st.warning("CAGR column was not found.")

    with col_right:

        st.subheader("GDP Comparison: 2025 vs 2030")

        if gdp_2025_column and gdp_2030_column:

            comparison = ranking_data[
                [
                    "Country",
                    gdp_2025_column,
                    gdp_2030_column
                ]
            ].copy()

            comparison = comparison.rename(
                columns={
                    gdp_2025_column: "GDP 2025",
                    gdp_2030_column: "Predicted GDP 2030"
                }
            )

            comparison_long = comparison.melt(
                id_vars="Country",
                var_name="Period",
                value_name="GDP"
            )

            fig_comparison = px.bar(
                comparison_long,
                x="Country",
                y="GDP",
                color="Period",
                barmode="group"
            )

            fig_comparison.update_layout(
                legend_title_text=""
            )

            st.plotly_chart(
                fig_comparison,
                use_container_width=True
            )

        else:
            st.warning(
                "GDP 2025 or predicted GDP 2030 column was not found."
            )

    st.subheader("Country Growth Ranking Table")

    st.dataframe(
        ranking_data,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# PAGE 2: COUNTRY FORECAST ANALYSIS
# =========================================================

elif selected_page == "Country Forecast Analysis":

    st.header(
        f"{selected_country} Forecast Analysis"
    )

    historical_2025 = country_gdp[
        country_gdp["Year"] == 2025
    ]

    forecast_2030 = country_gdp[
        country_gdp["Year"] == 2030
    ]

    gdp_2025 = (
        historical_2025[gdp_column].iloc[0]
        if not historical_2025.empty
        else None
    )

    gdp_2030 = (
        forecast_2030[gdp_column].iloc[0]
        if not forecast_2030.empty
        else None
    )

    cagr_value = (
        country_ranking[cagr_column].iloc[0]
        if cagr_column and not country_ranking.empty
        else None
    )

    mape_value = (
        country_evaluation[mape_column].iloc[0]
        if mape_column and not country_evaluation.empty
        else None
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "GDP in 2025",
        format_gdp(gdp_2025)
    )

    col2.metric(
        "Predicted GDP in 2030",
        format_gdp(gdp_2030)
    )

    col3.metric(
        "Forecast CAGR",
        (
            f"{cagr_value:.2f}%"
            if cagr_value is not None
            else "N/A"
        )
    )

    col4.metric(
        "Model MAPE",
        (
            f"{mape_value:.2f}%"
            if mape_value is not None
            else "N/A"
        )
    )

    st.markdown("---")

    st.subheader("Historical and Forecast GDP Trend")

    recent_data = country_gdp[
        country_gdp["Year"] >= 2000
    ].copy()

    if data_type_column:

        fig_country = px.line(
            recent_data,
            x="Year",
            y=gdp_column,
            color=data_type_column,
            markers=True,
            labels={
                gdp_column: "GDP",
                data_type_column: "Data Type"
            }
        )

    else:

        fig_country = px.line(
            recent_data,
            x="Year",
            y=gdp_column,
            markers=True
        )

    fig_country.add_vline(
        x=2025.5,
        line_dash="dash",
        annotation_text="Forecast begins"
    )

    fig_country.update_layout(
        hovermode="x unified",
        legend_title_text=""
    )

    st.plotly_chart(
        fig_country,
        use_container_width=True
    )

    left_chart, right_chart = st.columns(2)

    with left_chart:

        st.subheader("Annual GDP Growth Rate")

        if growth_column:

            growth_data = recent_data.dropna(
                subset=[growth_column]
            )

            fig_growth = px.line(
                growth_data,
                x="Year",
                y=growth_column,
                markers=True,
                labels={
                    growth_column: "Annual Growth (%)"
                }
            )

            fig_growth.add_hline(
                y=0,
                line_dash="dash"
            )

            st.plotly_chart(
                fig_growth,
                use_container_width=True
            )

        else:
            st.warning(
                "Annual growth column was not found."
            )

    with right_chart:

        st.subheader("Forecast Confidence Interval")

        if data_type_column:

            forecast_data = country_gdp[
                country_gdp[data_type_column]
                .astype(str)
                .str.lower()
                .str.contains("forecast")
            ].copy()

        else:

            forecast_data = country_gdp[
                country_gdp["Year"] >= 2026
            ].copy()

        if (
            lower_column
            and upper_column
            and not forecast_data.empty
        ):

            fig_interval = go.Figure()

            fig_interval.add_trace(
                go.Scatter(
                    x=forecast_data["Year"],
                    y=forecast_data[upper_column],
                    mode="lines",
                    name="Upper Estimate",
                    line=dict(width=0)
                )
            )

            fig_interval.add_trace(
                go.Scatter(
                    x=forecast_data["Year"],
                    y=forecast_data[lower_column],
                    mode="lines",
                    name="Lower Estimate",
                    fill="tonexty",
                    line=dict(width=0)
                )
            )

            fig_interval.add_trace(
                go.Scatter(
                    x=forecast_data["Year"],
                    y=forecast_data[gdp_column],
                    mode="lines+markers",
                    name="Predicted GDP"
                )
            )

            fig_interval.update_layout(
                xaxis_title="Year",
                yaxis_title="GDP",
                legend_title_text=""
            )

            st.plotly_chart(
                fig_interval,
                use_container_width=True
            )

        else:
            st.warning(
                "Confidence interval columns were not found."
            )

    st.subheader("Forecast Table: 2026–2030")

    forecast_table = country_gdp[
        country_gdp["Year"] >= 2026
    ].copy()

    st.dataframe(
        forecast_table,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# PAGE 3: MODEL EVALUATION
# =========================================================

else:

    st.header(
        f"Model Evaluation: {selected_country}"
    )

    st.caption(
        "Model performance was evaluated using predictions "
        "for the test period 2021–2025."
    )

    if country_evaluation.empty:

        st.warning(
            f"No evaluation data found for {selected_country}."
        )

    else:

        mae_value = (
            country_evaluation["MAE"].iloc[0]
            if "MAE" in country_evaluation.columns
            else None
        )

        rmse_value = (
            country_evaluation["RMSE"].iloc[0]
            if "RMSE" in country_evaluation.columns
            else None
        )

        mape_value = (
            country_evaluation[mape_column].iloc[0]
            if mape_column
            else None
        )

        category, category_color = get_accuracy_category(
            mape_value
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "MAPE",
            (
                f"{mape_value:.2f}%"
                if mape_value is not None
                else "N/A"
            )
        )

        col2.metric(
            "MAE",
            format_gdp(mae_value)
        )

        col3.metric(
            "RMSE",
            format_gdp(rmse_value)
        )

        with col4:

            st.markdown(
                f"""
                <div
                    class="accuracy-box"
                    style="background-color: {category_color};"
                >
                    Forecast Accuracy<br>
                    {category}
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    st.subheader("Actual vs Predicted GDP")

    if (
        not country_test.empty
        and actual_column
        and predicted_column
    ):

        fig_test = go.Figure()

        fig_test.add_trace(
            go.Scatter(
                x=country_test["Year"],
                y=country_test[actual_column],
                mode="lines+markers",
                name="Actual GDP"
            )
        )

        fig_test.add_trace(
            go.Scatter(
                x=country_test["Year"],
                y=country_test[predicted_column],
                mode="lines+markers",
                name="Predicted GDP"
            )
        )

        if test_upper_column and test_lower_column:

            fig_test.add_trace(
                go.Scatter(
                    x=country_test["Year"],
                    y=country_test[test_upper_column],
                    mode="lines",
                    name="Upper Estimate",
                    line=dict(width=0)
                )
            )

            fig_test.add_trace(
                go.Scatter(
                    x=country_test["Year"],
                    y=country_test[test_lower_column],
                    mode="lines",
                    name="Lower Estimate",
                    fill="tonexty",
                    line=dict(width=0)
                )
            )

        fig_test.update_layout(
            xaxis_title="Year",
            yaxis_title="GDP",
            hovermode="x unified",
            legend_title_text=""
        )

        st.plotly_chart(
            fig_test,
            use_container_width=True
        )

        country_test["Prediction_Error_Percent"] = (
            abs(
                country_test[predicted_column]
                - country_test[actual_column]
            )
            / country_test[actual_column]
        ) * 100

        st.subheader("Prediction Error by Year")

        fig_error = px.bar(
            country_test,
            x="Year",
            y="Prediction_Error_Percent",
            text="Prediction_Error_Percent",
            labels={
                "Prediction_Error_Percent":
                "Prediction Error (%)"
            }
        )

        fig_error.update_traces(
            texttemplate="%{text:.2f}%"
        )

        st.plotly_chart(
            fig_error,
            use_container_width=True
        )

    else:

        st.warning(
            "Actual or predicted GDP columns were not found."
        )

    st.subheader("Model Performance Comparison")

    comparison_col1, comparison_col2, comparison_col3 = (
        st.columns(3)
    )

    with comparison_col1:

        if mape_column:

            mape_chart = evaluation_data.sort_values(
                mape_column,
                ascending=True
            )

            fig_mape = px.bar(
                mape_chart,
                x=mape_column,
                y="Country",
                orientation="h",
                title="MAPE by Country",
                labels={
                    mape_column: "MAPE (%)"
                }
            )

            st.plotly_chart(
                fig_mape,
                use_container_width=True
            )

    with comparison_col2:

        if "MAE" in evaluation_data.columns:

            mae_chart = evaluation_data.sort_values(
                "MAE",
                ascending=True
            )

            fig_mae = px.bar(
                mae_chart,
                x="MAE",
                y="Country",
                orientation="h",
                title="MAE by Country"
            )

            st.plotly_chart(
                fig_mae,
                use_container_width=True
            )

    with comparison_col3:

        if "RMSE" in evaluation_data.columns:

            rmse_chart = evaluation_data.sort_values(
                "RMSE",
                ascending=True
            )

            fig_rmse = px.bar(
                rmse_chart,
                x="RMSE",
                y="Country",
                orientation="h",
                title="RMSE by Country"
            )

            st.plotly_chart(
                fig_rmse,
                use_container_width=True
            )

    st.subheader("Complete Evaluation Table")

    st.dataframe(
        evaluation_data,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Economic Growth Forecasting Project | "
    "Prophet Model | Historical data 1975–2025 | "
    "Forecast 2026–2030"
)

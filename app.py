################################################################################
# SECTION 1 - IMPORT LIBRARIES
################################################################################

from pathlib import Path

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, dash_table, dcc, html
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs.layout import xaxis


################################################################################
# SECTION 2 - GLOBAL CONSTANTS
################################################################################

APP_TITLE = "Walmart Sales Dashboard"
APP_SUBTITLE = "Dashboard for Sales Performance Analysis"

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "walmart_data.xlsx"

PLOT_TEMPLATE = "plotly_white"
FONT_FAMILY = "Arial, Helvetica, sans-serif"

PRIMARY_BLUE = "#1f77b4"
SECONDARY_BLUE = "#4f9fe6"
DARK_BLUE = "#123b63"
LIGHT_BLUE = "#dceeff"
SUCCESS_GREEN = "#2ca58d"
WARNING_ORANGE = "#f4a261"
ALERT_RED = "#d1495b"
LIGHT_GREY = "#f5f7fb"
MID_GREY = "#6b7280"
BORDER_GREY = "#d9e2ef"
TEXT_DARK = "#1f2937"

CARD_RADIUS = "16px"
CARD_SHADOW = "0 8px 24px rgba(18, 59, 99, 0.08)"
CHART_FONT_SIZE = 12

MONTH_ORDER = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
QUARTER_ORDER = ["Q1", "Q2", "Q3", "Q4"]
NUMERIC_COLUMNS = [
    "Weekly_Sales",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
]

CHART_CONFIG = {"displayModeBar": False, "responsive": True}

SIDEBAR_STYLE = {
    "width": "200px",
    "backgroundColor": DARK_BLUE,
    "color": "white",
    "padding": "24px 20px",
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "overflowY": "auto",
}

CONTENT_STYLE = {
    "marginLeft": "240px",
    "padding": "24px",
    "backgroundColor": LIGHT_GREY,
    "fontFamily": FONT_FAMILY,
    "color": TEXT_DARK,
}

SECTION_CARD_STYLE = {
    "backgroundColor": "white",
    "borderRadius": CARD_RADIUS,
    "padding": "20px",
    "marginBottom": "24px",
    "boxShadow": CARD_SHADOW,
    "border": f"1px solid {BORDER_GREY}",
}

KPI_CARD_STYLE = {
    "backgroundColor": "white",
    "borderRadius": CARD_RADIUS,
    "padding": "18px",
    "boxShadow": CARD_SHADOW,
    "border": f"1px solid {BORDER_GREY}",
    "minHeight": "120px",
}

INSIGHT_CARD_STYLE = {
    "backgroundColor": "#fbfdff",
    "borderLeft": f"5px solid {PRIMARY_BLUE}",
    "borderRadius": "12px",
    "padding": "16px 18px",
    "marginTop": "12px",
    "marginBottom": "8px",
    "boxShadow": "0 4px 12px rgba(31, 119, 180, 0.06)",
}

FILTER_LABEL_STYLE = {
    "fontWeight": "bold",
    "fontSize": "13px",
    "marginBottom": "6px",
    "display": "block",
    "color": DARK_BLUE,
}

TABLE_STYLE_CELL = {
    "textAlign": "left",
    "padding": "10px",
    "fontFamily": FONT_FAMILY,
    "fontSize": "12px",
    "border": f"1px solid {BORDER_GREY}",
    "whiteSpace": "normal",
    "height": "auto",
}

TABLE_STYLE_HEADER = {
    "backgroundColor": DARK_BLUE,
    "color": "white",
    "fontWeight": "bold",
    "border": f"1px solid {DARK_BLUE}",
}


################################################################################
# SECTION 3 - LOAD INPUT FILES
################################################################################

def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Purpose:
    Load the Walmart sales dataset from the Excel file.

    Input:
    Path object pointing to the Excel file.

    Output:
    Raw pandas DataFrame.
    """
    return pd.read_excel(file_path, engine="openpyxl")


raw_df = load_dataset(DATA_FILE)


################################################################################
# SECTION 4 - DATA PREPARATION
################################################################################

def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Purpose:
    Clean the dataset and capture preparation diagnostics.

    Input:
    Raw pandas DataFrame.

    Output:
    Tuple containing cleaned DataFrame and preparation metadata.
    """
    working_df = df.copy()

    initial_shape = working_df.shape
    missing_values = working_df.isna().sum().reset_index()
    missing_values.columns = ["Column", "Missing Values"]

    duplicate_count = int(working_df.duplicated().sum())
    if duplicate_count > 0:
        working_df = working_df.drop_duplicates().copy()

    working_df["Date"] = pd.to_datetime(working_df["Date"], errors="coerce")
    working_df = working_df.sort_values(["Date", "Store"]).reset_index(drop=True)

    metadata = {
        "initial_shape": initial_shape,
        "cleaned_shape": working_df.shape,
        "missing_values": missing_values,
        "duplicate_count": duplicate_count,
    }
    return working_df, metadata


clean_df, prep_metadata = prepare_dataset(raw_df)


################################################################################
# SECTION 5 - FEATURE ENGINEERING
################################################################################

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
    Create additional calendar and descriptive fields required for analysis.

    Input:
    Cleaned pandas DataFrame.

    Output:
    Feature-engineered pandas DataFrame.
    """
    feature_df = df.copy()
    feature_df["Year"] = feature_df["Date"].dt.year
    feature_df["Month"] = feature_df["Date"].dt.month
    feature_df["Month_Name"] = feature_df["Date"].dt.strftime("%b")
    feature_df["Quarter"] = "Q" + feature_df["Date"].dt.quarter.astype(str)
    feature_df["Week Number"] = feature_df["Date"].dt.isocalendar().week.astype(int)
    feature_df["Holiday Label"] = np.where(
        feature_df["Holiday_Flag"] == 1,
        "Holiday Week",
        "Non-Holiday Week",
    )
    return feature_df


processed_df = engineer_features(clean_df)


################################################################################
# SECTION 6 - SUMMARY STATISTICS
################################################################################

def build_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Purpose:
    Generate summary statistics for numeric columns.

    Input:
    Processed pandas DataFrame.

    Output:
    Summary statistics DataFrame.
    """
    summary_df = df[NUMERIC_COLUMNS].describe().round(2).reset_index()
    return summary_df.rename(columns={"index": "Statistic"})


summary_statistics_df = build_summary_statistics(processed_df)


################################################################################
# SECTION 7 - KPI CALCULATIONS
################################################################################

def calculate_kpis(df: pd.DataFrame) -> dict:
    """
    Purpose:
    Calculate top-level KPI values for the dashboard.

    Input:
    Filtered pandas DataFrame.

    Output:
    Dictionary of KPI labels and values.
    """
    if df.empty:
        return {
            "Total Sales": 0,
            "Average Weekly Sales": 0,
            "Maximum Weekly Sales": 0,
            "Minimum Weekly Sales": 0,
            "Number of Stores": 0,
            "Number of Holiday Weeks": 0,
            "Average Fuel Price": 0,
            "Average CPI": 0,
            "Average Temperature": 0,
            "Average Unemployment": 0,
        }

    return {
        "Total Sales": df["Weekly_Sales"].sum(),
        "Average Weekly Sales": df["Weekly_Sales"].mean(),
        "Maximum Weekly Sales": df["Weekly_Sales"].max(),
        "Minimum Weekly Sales": df["Weekly_Sales"].min(),
        "Number of Stores": df["Store"].nunique(),
        "Number of Holiday Weeks": int(df["Holiday_Flag"].sum()),
        "Average Fuel Price": df["Fuel_Price"].mean(),
        "Average CPI": df["CPI"].mean(),
        "Average Temperature": df["Temperature"].mean(),
        "Average Unemployment": df["Unemployment"].mean(),
    }


def format_kpi_value(label: str, value: float) -> str:
    """
    Purpose:
    Format KPI values consistently for executive display.

    Input:
    KPI label and numeric value.

    Output:
    Formatted string.
    """
    if label in {
        "Total Sales",
        "Average Weekly Sales",
        "Maximum Weekly Sales",
        "Minimum Weekly Sales",
    }:
        return f"${value:,.0f}"
    if label in {"Number of Stores", "Number of Holiday Weeks"}:
        return f"{int(value)}"
    return f"{value:,.2f}"


################################################################################
# SECTION 8 - FILTER COMPONENTS
################################################################################

def build_filter_options(values: list) -> list[dict]:
    """
    Purpose:
    Convert a list of values into Dash dropdown options.

    Input:
    List of values.

    Output:
    List of dictionaries for Dash dropdown options.
    """
    return [{"label": "All", "value": "All"}] + [
        {"label": str(value), "value": value} for value in values
    ]


store_options = build_filter_options(sorted(processed_df["Store"].unique().tolist()))
year_options = build_filter_options(sorted(processed_df["Year"].unique().tolist()))
quarter_options = build_filter_options(sorted(processed_df["Quarter"].unique().tolist()))
holiday_options = [
    {"label": "All", "value": "All"},
    {"label": "Holiday Week", "value": 1},
    {"label": "Non-Holiday Week", "value": 0},
]


def filter_dataset(
    df: pd.DataFrame,
    selected_store,
    selected_year,
    selected_quarter,
    selected_holiday,
) -> pd.DataFrame:
    """
    Purpose:
    Apply dashboard filters to the processed dataset.

    Input:
    Processed DataFrame and selected filter values.

    Output:
    Filtered DataFrame.
    """
    filtered_df = df.copy()

    if selected_store != "All":
        filtered_df = filtered_df[filtered_df["Store"] == selected_store]

    if selected_year != "All":
        filtered_df = filtered_df[filtered_df["Year"] == selected_year]

    if selected_quarter != "All":
        filtered_df = filtered_df[filtered_df["Quarter"] == selected_quarter]

    if selected_holiday != "All":
        filtered_df = filtered_df[filtered_df["Holiday_Flag"] == selected_holiday]

    return filtered_df


################################################################################
# SECTION 9 - CHART FUNCTIONS
################################################################################

def apply_standard_layout(fig: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    """
    Purpose:
    Apply a consistent executive layout to Plotly figures.

    Input:
    Plotly figure, chart title, x-axis title, and y-axis title.

    Output:
    Styled Plotly figure.
    """
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title={
            "text": title,
            "x": 0.01,
            "xanchor": "left",
            "font": {"size": 18, "color": DARK_BLUE},
        },
        font={"family": FONT_FAMILY, "size": CHART_FONT_SIZE, "color": TEXT_DARK},
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 50, "r": 30, "t": 60, "b": 50},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.01,
        },
        hoverlabel={"font_size": 12, "font_family": FONT_FAMILY},
        transition={"duration": 500, "easing": "cubic-in-out"},
    )
    fig.update_xaxes(
        title=x_title,
        showgrid=False,
        zeroline=False,
        linecolor=BORDER_GREY,
        tickfont={"size": 11},
    )
    fig.update_yaxes(
        title=y_title,
        showgrid=True,
        gridcolor="#edf2f7",
        zeroline=False,
        linecolor=BORDER_GREY,
        tickfont={"size": 11},
    )
    return fig


def create_empty_figure(message: str) -> go.Figure:
    """
    Purpose:
    Create a placeholder figure when filtered data is unavailable.

    Input:
    Message string.

    Output:
    Plotly Figure.
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16, "color": MID_GREY},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return apply_standard_layout(fig, message, "", "")


################################################################################
# >>>>Exploratory Data Analysis
# Q1 - Distribution of Weekly Sales (Histogram)
################################################################################

def create_weekly_sales_histogram(df: pd.DataFrame) -> go.Figure:
    """
    Purpose:
    Create histogram for Weekly Sales.

    Input:
    Processed DataFrame.

    Output:
    Plotly Figure.
    """
    fig = px.histogram(
        df,
        x="Weekly_Sales",
        nbins=min(12, max(5, len(df))),
        color_discrete_sequence=[PRIMARY_BLUE],
        opacity=0.9,
    )
    fig.update_traces(
        hovertemplate="Weekly Sales: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
        texttemplate="%{y}",
        textposition="outside",
    )
    return apply_standard_layout(fig, "Q1. Distribution of Weekly Sales", "Weekly Sales", "Number of Weeks (Records)")

################################################################################
# Q2 - Weekly Sales by Store (Bar Chart)
################################################################################

def create_weekly_sales_by_store(df: pd.DataFrame) -> go.Figure:
    """
    Purpose:
    Create bar chart for total weekly sales by store.

    Input:
    Processed DataFrame.

    Output:
    Plotly Figure.
    """
    store_sales = (
        df.groupby("Store", as_index=False)["Weekly_Sales"]
        .sum()
        .sort_values("Weekly_Sales", ascending=False)
    )
    fig = px.bar(
        store_sales,
        x="Store",
        y="Weekly_Sales",
        text="Weekly_Sales",
        color="Weekly_Sales",
        color_continuous_scale=[[0, LIGHT_BLUE], [1, PRIMARY_BLUE]],
    )
    fig.update_traces(
        texttemplate="$%{text:,.0f}",
        textposition="outside",
        hovertemplate="Store %{x}<br>Total Sales: $%{y:,.0f}<extra></extra>",
    )
    fig.update_layout(coloraxis_showscale=False, showlegend=True, legend={"xanchor": "right", "x": 1})
    fig.update_yaxes(rangemode="tozero")
    return apply_standard_layout(fig, "Q2. Weekly Sales by Store", "Store", "Total Weekly Sales")
    # fig.update_yaxes(rangemode="tozero")
    # return apply_standard_layout(fig, "Q2. Weekly Sales by Store", "Store", "Total Weekly Sales")


################################################################################
# Q3 - Monthly Sales Trend (Line Chart)
################################################################################

def create_monthly_sales_trend(df: pd.DataFrame) -> go.Figure:
    """
    Purpose:
    Create line chart for monthly sales trend.

    Input:
    Processed DataFrame.

    Output:
    Plotly Figure.
    """
    monthly_sales = (
        df.groupby(["Year", "Month", "Month_Name"], as_index=False)["Weekly_Sales"]
        .sum()
        .sort_values(["Year", "Month"])
    )
    monthly_sales["Period"] = monthly_sales["Month_Name"] + " " + monthly_sales["Year"].astype(str)
    fig = px.line(monthly_sales, x="Period", y="Weekly_Sales", markers=True)
    fig.update_traces(
        line={"color": PRIMARY_BLUE, "width": 3},
        marker={"size": 9, "color": SECONDARY_BLUE},
        hovertemplate="Period: %{x}<br>Total Sales: $%{y:,.0f}<extra></extra>",
    )
    # fig.update_layout(legend={"xanchor": "right", "x": 1})
    return apply_standard_layout(fig, "Q3. Monthly Sales Trend", "Month", "Total Sales")


################################################################################
# Q4 - Holiday vs Non-Holiday Sales (Box Plot)
################################################################################

def create_holiday_sales_boxplot(df: pd.DataFrame) -> go.Figure:
    """
    Purpose:
    Create box plot for holiday versus non-holiday sales.

    Input:
    Processed DataFrame.

    Output:
    Plotly Figure.
    """
    fig = px.box(
        df,
        x="Holiday Label",
        y="Weekly_Sales",
        color="Holiday Label",
        color_discrete_map={"Holiday Week": WARNING_ORANGE, "Non-Holiday Week": PRIMARY_BLUE},
        points="all",
    )
    fig.update_traces(
        jitter=0.25,
        pointpos=0,
        hovertemplate="%{x}<br>Weekly Sales: $%{y:,.0f}<extra></extra>",
    )
    fig = apply_standard_layout(fig, "Q4. Holiday vs Non-Holiday Sales", "Week Type", "Weekly Sales")
    fig.update_layout(legend={"xanchor": "right", "x": 1})
    return fig


################################################################################
# Q5 - Correlation Heatmap
################################################################################

def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Purpose:
    Create correlation heatmap for numeric variables.

    Input:
    Processed DataFrame.

    Output:
    Plotly Figure.
    """
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["Holiday_Flag", "Year"], errors="ignore")
    correlation_matrix = numeric_df.corr().round(2)
    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale=[[0.0, "#d73027"], [0.5, "#f7f7f7"], [1.0, "#1f78b4"]],
            zmin=-1,
            zmax=1,
            text=correlation_matrix.values,
            texttemplate="%{text}",
            hovertemplate="X: %{x}<br>Y: %{y}<br>Correlation: %{z:.2f}<extra></extra>",
            colorbar={"title": "Correlation"},
        )
    )
    return apply_standard_layout(fig, "Q5. Correlation Heatmap", "", "")

################################################################################
# Q6 - Fuel Price vs Weekly Sales (Scatter Plot)
################################################################################

def create_fuel_price_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Purpose:
    Create scatter plot for fuel price versus weekly sales.

    Input:
    Processed DataFrame.

    Output:
    Plotly Figure.
    """
    scatter_df = df.copy()
    scatter_df["Store_Label"] = scatter_df["Store"].astype(str)
    fig = px.scatter(
        scatter_df,
        x="Fuel_Price",
        y="Weekly_Sales",
        color="Store_Label",
        size="Weekly_Sales",
        size_max=28,
        color_discrete_sequence=[PRIMARY_BLUE, SECONDARY_BLUE, SUCCESS_GREEN, WARNING_ORANGE],
        hover_data={
            "Store_Label": False,
            "Store": True,
            "Fuel_Price": ":.2f",
            "Weekly_Sales": ":,.0f",
            "Date": True,
        },
        labels={"Store_Label": "Store"},
    )
    fig.update_traces(
        marker={"line": {"width": 1, "color": "white"}, "opacity": 0.85},
        hovertemplate=(
            "Store %{customdata[0]}<br>"
            "Fuel Price: %{x:.2f}<br>"
            "Weekly Sales: $%{y:,.0f}<br>"
            "Date: %{customdata[3]|%Y-%m-%d}<extra></extra>"
        ),
    )
    fig = apply_standard_layout(fig, "Q6. Fuel Price vs Weekly Sales", "Fuel Price", "Weekly Sales")
    fig.update_layout(legend={"xanchor": "right", "x": 1})
    return fig

################################################################################
# Q7 - Temperature Distribution (Histogram)
################################################################################

def create_temperature_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Purpose:
    Create histogram for temperature distribution.

    Input:
    Processed DataFrame.

    Output:
    Plotly Figure.
    """
    fig = px.histogram(
        df,
        x="Temperature",
        nbins=min(10, max(5, len(df))),
        color_discrete_sequence=[SECONDARY_BLUE],
        opacity=0.9,
    )
    fig.update_traces(
        hovertemplate="Temperature: %{x:.1f}<br>Count: %{y}<extra></extra>",
        texttemplate="%{x:.1f} ℃<br>%{y} Nos",
        textposition="outside",
    )
    return apply_standard_layout(fig, "Q7. Temperature Distribution", "Temperature", "Number of Records")

################################################################################
# Q8 - CPI Trend (Line Chart)
################################################################################

def create_cpi_trend(df: pd.DataFrame) -> go.Figure:
    """
    Purpose:
    Create line chart for CPI trend.

    Input:
    Processed DataFrame.

    Output:
    Plotly Figure.
    """
    cpi_trend = df.groupby("Date", as_index=False)["CPI"].mean().sort_values("Date")
    fig = px.line(cpi_trend, x="Date", y="CPI", markers=True)
    fig.update_traces(
        line={"color": PRIMARY_BLUE, "width": 3},
        marker={"size": 8, "color": SECONDARY_BLUE},
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Average CPI: %{y:.2f}<extra></extra>",
    )
    return apply_standard_layout(fig, "Q8. CPI Trend", "Date", "Average CPI")


################################################################################
# Q9 - Unemployment Trend (Line Chart)
################################################################################

def create_unemployment_trend(df: pd.DataFrame) -> go.Figure:
    """
    Purpose:
    Create line chart for unemployment trend.

    Input:
    Processed DataFrame.

    Output:
    Plotly Figure.
    """
    unemployment_trend = df.groupby("Date", as_index=False)["Unemployment"].mean().sort_values("Date")
    fig = px.line(unemployment_trend, x="Date", y="Unemployment", markers=True)
    fig.update_traces(
        line={"color": WARNING_ORANGE, "width": 3},
        marker={"size": 8, "color": ALERT_RED},
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Average Unemployment: %{y:.2f}<extra></extra>",
    )
    return apply_standard_layout(fig, "Q9. Unemployment Trend", "Date", "Average Unemployment")


################################################################################
# Q10 - Sales Distribution by Quarter (Box Plot)
################################################################################

def create_quarter_sales_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Purpose:
    Create box plot for sales distribution by quarter.

    Input:
    Processed DataFrame.

    Output:
    Plotly Figure.
    """
    fig = px.box(
        df,
        x="Quarter",
        y="Weekly_Sales",
        color="Quarter",
        category_orders={"Quarter": QUARTER_ORDER},
        color_discrete_sequence=[PRIMARY_BLUE, SECONDARY_BLUE, SUCCESS_GREEN, WARNING_ORANGE],
        points="all",
    )
    fig.update_traces(
        jitter=0.25,
        pointpos=0,
        hovertemplate="Quarter %{x}<br>Weekly Sales: $%{y:,.0f}<extra></extra>",
    )
    fig = apply_standard_layout(fig, "Q10. Sales Distribution by Quarter", "Quarter", "Weekly Sales")
    fig.update_layout(legend={"xanchor": "right", "x": 1})
    return fig

################################################################################
# >>>>Advanced Visualization
# Q1 - Monthly Sales Heatmap
################################################################################

def create_monthly_sales_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Purpose:
    Create heatmap for monthly sales by store.

    Input:
    Processed DataFrame.

    Output:
    Plotly Figure.
    """
    heatmap_df = (
        df.groupby(["Store", "Month_Name"], as_index=False)["Weekly_Sales"]
        .sum()
        .pivot(index="Store", columns="Month_Name", values="Weekly_Sales")
        .reindex(columns=[month for month in MONTH_ORDER if month in df["Month_Name"].unique()])
        .fillna(0)
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_df.values,
            x=heatmap_df.columns,
            y=[f"Store {store}" for store in heatmap_df.index],
            colorscale=[[0, "#eef5ff"], [0.5, "#7fb3ff"], [1, "#1f4e8c"]],
            text=np.vectorize(lambda x: f"${x:,.0f}")(heatmap_df.values),
            texttemplate="%{text}",
            hovertemplate="Store: %{y}<br>Month: %{x}<br>Total Sales: $%{z:,.0f}<extra></extra>",
            colorbar={"title": "Sales"},
        )
    )
    return apply_standard_layout(fig, "Q1. Monthly Sales Heatmap", "Month", "Store")


################################################################################
# Q2 - Calendar Heatmap
################################################################################

def create_calendar_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Purpose:
    Create calendar-style heatmap using month and week number.

    Input:
    Processed DataFrame.

    Output:
    Plotly Figure.
    """
    calendar_df = (
        df.groupby(["Month_Name", "Week Number"], as_index=False)["Weekly_Sales"]
        .sum()
        .copy()
    )
    ordered_months = [month for month in MONTH_ORDER if month in calendar_df["Month_Name"].unique()]
    calendar_pivot = (
        calendar_df.pivot(index="Month_Name", columns="Week Number", values="Weekly_Sales")
        .reindex(ordered_months)
        .fillna(0)
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=calendar_pivot.values,
            x=calendar_pivot.columns,
            y=calendar_pivot.index,
            colorscale=[[0, "#eef5ff"], [0.5, "#7fb3ff"], [1, "#0b4f9c"]],
            text=np.vectorize(lambda x: f"${x:,.0f}" if x > 0 else "")(calendar_pivot.values),
            texttemplate="%{text}",
            hovertemplate="Month: %{y}<br>Week Number: %{x}<br>Total Sales: $%{z:,.0f}<extra></extra>",
            colorbar={"title": "Sales Intensity"},
        )
    )
    return apply_standard_layout(fig, "Q2. Calendar-Style Sales Heatmap", "Week Number", "Month")


################################################################################
# SECTION 10 - EXECUTIVE INSIGHT FUNCTIONS
################################################################################

def safe_percentage_change(current_value: float, reference_value: float) -> float:
    """
    Purpose:
    Calculate percentage change safely.

    Input:
    Current value and reference value.

    Output:
    Percentage change as float.
    """
    if reference_value == 0 or pd.isna(reference_value):
        return 0.0
    return ((current_value - reference_value) / reference_value) * 100


def build_insight_card(title: str, why_text: str, insight_text: str, recommendation_text: str):
    """
    Purpose:
    Create a reusable executive insight card.

    Input:
    Title and three explanatory text blocks.

    Output:
    Dash HTML component.
    """
    return html.Div(
        [
            html.Div(title, style={"fontWeight": "bold", "fontSize": "15px", "color": DARK_BLUE, "marginBottom": "10px"}),
            html.P([html.Strong("Why this visualization? "), why_text], style={"margin": "4px 0"}),
            html.P([html.Strong("Business Insight: "), insight_text], style={"margin": "4px 0"}),
            html.P([html.Strong("Recommendation: "), recommendation_text], style={"margin": "4px 0"}),
        ],
        style=INSIGHT_CARD_STYLE,
    )


def generate_chart_insights(df: pd.DataFrame) -> dict:
    """
    Purpose:
    Generate concise executive insights for each question.

    Input:
    Filtered DataFrame.

    Output:
    Dictionary keyed by question number.
    """
    if df.empty:
        empty_message = "No data is available for the selected filters."
        return {
            key: build_insight_card(key, empty_message, empty_message, "Adjust the filters to restore a meaningful executive view.")
            for key in [f"Q{i}" for i in range(1, 13)]
        }

    store_sales = df.groupby("Store")["Weekly_Sales"].sum().sort_values(ascending=False)
    top_store = int(store_sales.index[0])
    top_store_sales = float(store_sales.iloc[0])

    monthly_sales = (
        df.groupby(["Year", "Month", "Month_Name"], as_index=False)["Weekly_Sales"]
        .sum()
        .sort_values(["Year", "Month"])
    )
    best_month_row = monthly_sales.loc[monthly_sales["Weekly_Sales"].idxmax()]

    holiday_sales = df.groupby("Holiday Label")["Weekly_Sales"].mean()
    holiday_avg = float(holiday_sales.get("Holiday Week", 0))
    non_holiday_avg = float(holiday_sales.get("Non-Holiday Week", 0))
    holiday_change = safe_percentage_change(holiday_avg, non_holiday_avg)

    corr_df = df.select_dtypes(include=[np.number]).corr()
    fuel_corr = float(corr_df.loc["Fuel_Price", "Weekly_Sales"])
    temp_corr = float(corr_df.loc["Temperature", "Weekly_Sales"])
    cpi_corr = float(corr_df.loc["CPI", "Weekly_Sales"])
    unemployment_corr = float(corr_df.loc["Unemployment", "Weekly_Sales"])

    quarter_sales = df.groupby("Quarter")["Weekly_Sales"].mean().sort_values(ascending=False)
    best_quarter = quarter_sales.index[0]

    return {
        "Q1": build_insight_card("Q1 Executive Insight", "A histogram is the clearest way to show how weekly sales are distributed and whether performance is concentrated in a narrow or broad range.", f"Most weekly sales observations cluster around the mid-range, with the filtered average at ${df['Weekly_Sales'].mean():,.0f}. This helps management judge sales consistency and volatility.", "Use this distribution as the baseline for setting realistic weekly sales targets and identifying unusually weak or strong trading weeks."),
        "Q2": build_insight_card("Q2 Executive Insight", "A sorted bar chart makes store-to-store comparison immediate and reduces cognitive effort for executives.", f"Store {top_store} leads the selected view with total sales of ${top_store_sales:,.0f}, indicating stronger local demand or better execution than peer stores.", "Review the operating practices of the leading store and replicate successful merchandising, staffing, and inventory decisions across lower-performing stores."),
        "Q3": build_insight_card("Q3 Executive Insight", "A line chart is best for continuous time-based analysis and highlights momentum, seasonality, and turning points.", f"The strongest month in the filtered data is {best_month_row['Month_Name']} {int(best_month_row['Year'])}, delivering ${best_month_row['Weekly_Sales']:,.0f} in sales.", "Align promotional calendars, replenishment planning, and labour scheduling with high-demand months to protect service levels and maximise revenue."),
        "Q4": build_insight_card("Q4 Executive Insight", "A box plot compares both central tendency and spread, making it ideal for holiday versus non-holiday sales behaviour.", f"Holiday weeks show an average sales change of {holiday_change:+.1f}% versus non-holiday weeks, indicating that holiday periods materially influence revenue performance.", "Treat holiday weeks as strategic trading windows with tighter inventory planning, stronger promotions, and contingency staffing."),
        "Q5": build_insight_card("Q5 Executive Insight", "A heatmap is effective for scanning multiple relationships quickly while keeping the focus on the strongest drivers.", f"Weekly sales show correlations of {fuel_corr:.2f} with fuel price, {temp_corr:.2f} with temperature, {cpi_corr:.2f} with CPI, and {unemployment_corr:.2f} with unemployment in the filtered view.", "Use the strongest relationships as inputs for forecasting and scenario planning, but avoid assuming causation without broader operational context."),
        "Q6": build_insight_card("Q6 Executive Insight", "A scatter plot is the right choice for examining the relationship between two numeric variables while preserving store-level variation.", f"The fuel price to sales correlation is {fuel_corr:.2f}, showing whether sales move meaningfully with fuel cost changes in the selected slice.", "Monitor fuel-sensitive markets more closely and combine fuel trends with local demand signals when planning promotions and pricing support."),
        "Q7": build_insight_card("Q7 Executive Insight", "A histogram clearly shows the operating temperature range across the selected period without unnecessary clutter.", f"The average temperature in the filtered data is {df['Temperature'].mean():.1f}, helping management understand the environmental conditions under which sales were generated.", "Use temperature patterns to refine seasonal assortment, especially for weather-sensitive categories and regional inventory allocation."),
        "Q8": build_insight_card("Q8 Executive Insight", "A line chart is ideal for tracking CPI over time and identifying inflationary movement that may affect consumer behaviour.", f"Average CPI in the filtered view is {df['CPI'].mean():.2f}, with a visible trend that can influence purchasing power and basket composition.", "Track CPI alongside category performance to identify where inflation may require pricing, promotion, or pack-size adjustments."),
        "Q9": build_insight_card("Q9 Executive Insight", "A line chart helps executives see whether labour market conditions are improving or weakening over time.", f"Average unemployment stands at {df['Unemployment'].mean():.2f} in the filtered data, providing context for local consumer demand resilience.", "In markets with weaker employment conditions, prioritise value-led promotions and tighter demand forecasting."),
        "Q10": build_insight_card("Q10 Executive Insight", "A box plot provides a strong executive view of quarterly spread, median performance, and outliers.", f"{best_quarter} has the highest average weekly sales in the filtered view, indicating the strongest quarter for revenue generation.", "Use the strongest quarter as the benchmark for annual planning and prepare targeted interventions for weaker quarters."),
        "Q11": build_insight_card("Q11 Executive Insight", "A heatmap allows management to compare store-month combinations rapidly and spot concentration of sales intensity.", "The darkest cells identify the store-month combinations contributing the highest revenue, making seasonal and store-level peaks easy to detect.", "Prioritise inventory and promotional investment in the store-month combinations that repeatedly show the strongest demand."),
        "Q12": build_insight_card("Q12 Executive Insight", "A calendar-style heatmap is the closest Plotly-only representation for showing how sales intensity changes across the trading calendar.", "This view highlights which weeks within each month carry the strongest sales intensity, helping management identify timing effects beyond monthly totals.", "Use week-level intensity patterns to fine-tune campaign timing, replenishment cycles, and holiday readiness."),
    }


def generate_executive_summary(df: pd.DataFrame):
    """
    Purpose:
    Build the executive summary card with key observations and recommendations.

    Input:
    Filtered DataFrame.

    Output:
    Dash HTML component.
    """
    if df.empty:
        return html.Div("No executive summary is available because the selected filters returned no data.", style=SECTION_CARD_STYLE)

    store_sales = df.groupby("Store")["Weekly_Sales"].sum().sort_values(ascending=False)
    top_store = int(store_sales.index[0])

    monthly_sales = (
        df.groupby(["Year", "Month", "Month_Name"], as_index=False)["Weekly_Sales"]
        .sum()
        .sort_values(["Year", "Month"])
    )
    best_month = monthly_sales.loc[monthly_sales["Weekly_Sales"].idxmax()]

    holiday_avg = df[df["Holiday_Flag"] == 1]["Weekly_Sales"].mean()
    non_holiday_avg = df[df["Holiday_Flag"] == 0]["Weekly_Sales"].mean()
    holiday_impact = safe_percentage_change(
        0 if pd.isna(holiday_avg) else holiday_avg,
        0 if pd.isna(non_holiday_avg) else non_holiday_avg,
    )

    corr_df = df.select_dtypes(include=[np.number]).corr()
    fuel_corr = float(corr_df.loc["Fuel_Price", "Weekly_Sales"])
    temp_corr = float(corr_df.loc["Temperature", "Weekly_Sales"])
    cpi_corr = float(corr_df.loc["CPI", "Weekly_Sales"])
    unemployment_corr = float(corr_df.loc["Unemployment", "Weekly_Sales"])

    economic_driver = max([("CPI", abs(cpi_corr)), ("Unemployment", abs(unemployment_corr))], key=lambda item: item[1])[0]

    observations = [
        f"Top Performing Store: Store {top_store}",
        f"Highest Sales Month: {best_month['Month_Name']} {int(best_month['Year'])}",
        f"Holiday Impact: {holiday_impact:+.1f}% average sales change versus non-holiday weeks",
        f"Fuel Price Impact: correlation with sales = {fuel_corr:.2f}",
        f"Temperature Impact: correlation with sales = {temp_corr:.2f}",
        f"Economic Indicator Impact: {economic_driver} shows the stronger relationship with sales in the selected view",
    ]

    recommendations = [
        f"Scale best practices from Store {top_store} across the wider network.",
        f"Protect service levels and inventory availability during {best_month['Month_Name']} {int(best_month['Year'])}, the strongest sales month.",
        "Treat holiday weeks as priority trading periods with tighter execution controls.",
        "Incorporate macro indicators into forecasting to improve planning accuracy.",
        "Use week-level and month-level heatmaps to align promotions with demand peaks.",
    ]

    return html.Div(
        [
            html.H3("Executive Summary", style={"marginTop": 0, "color": DARK_BLUE, "fontSize": "24px"}),
            html.Div(
                [
                    html.Div([html.H4("Key Observations", style={"color": PRIMARY_BLUE}), html.Ul([html.Li(item) for item in observations])], style={"flex": "1", "paddingRight": "20px"}),
                    html.Div([html.H4("Management Recommendations", style={"color": PRIMARY_BLUE}), html.Ul([html.Li(item) for item in recommendations])], style={"flex": "1"}),
                ],
                style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
            ),
        ],
        style=SECTION_CARD_STYLE,
    )


################################################################################
# SECTION 11 - DASHBOARD LAYOUT
################################################################################

def create_kpi_card(label: str, value: str):
    """
    Purpose:
    Create a reusable KPI card.

    Input:
    KPI label and formatted value.

    Output:
    Dash HTML component.
    """
    return html.Div(
        [
            html.Div(label, style={"fontSize": "13px", "fontWeight": "bold", "color": MID_GREY, "marginBottom": "10px"}),
            html.Div(value, style={"fontSize": "28px", "fontWeight": "bold", "color": DARK_BLUE, "lineHeight": "1.2"}),
        ],
        style=KPI_CARD_STYLE,
    )


def create_section_title(title: str, subtitle: str = ""):
    """
    Purpose:
    Create a standard section title block.

    Input:
    Title and optional subtitle.

    Output:
    Dash HTML component.
    """
    return html.Div(
        [
            html.H2(title, style={"marginBottom": "6px", "color": DARK_BLUE, "fontSize": "28px"}),
            html.P(subtitle, style={"marginTop": 0, "marginBottom": "18px", "color": MID_GREY, "fontSize": "14px"}),
        ]
    )


def create_chart_block(graph_id: str, insight_id: str):
    """
    Purpose:
    Create a reusable chart and insight container.

    Input:
    Graph component ID and insight container ID.

    Output:
    Dash HTML component.
    """
    return html.Div(
        [
            html.Div(dcc.Graph(id=graph_id, config=CHART_CONFIG), style=SECTION_CARD_STYLE),
            html.Div(id=insight_id),
        ],
        style={"marginBottom": "24px"},
    )


app = Dash(__name__)
app.title = APP_TITLE

app.layout = html.Div(
    [
        html.Div(
            [
                html.H2(APP_TITLE, style={"fontSize": "24px", "marginBottom": "8px"}),
                html.P("DSM301 Assignment 2", style={"fontSize": "16px", "lineHeight": "1.5", "color": "#dbeafe"}),
                html.Hr(style={"borderColor": "rgba(255,255,255,0.2)"}),
                html.Div("Navigation", style={"fontWeight": "bold", "marginBottom": "12px", "fontSize": "14px"}),
                html.Ul(
                    [
                        html.Li(html.A("Data Preparation & Summary Statistics", href="#data-preparation", style={"color": "white", "textDecoration": "none"})),
                        html.Li(html.A("Exploratory Analysis", href="#exploratory-analysis", style={"color": "white", "textDecoration": "none"})),
                        html.Li(html.A("Advanced Analytics", href="#advanced-analytics", style={"color": "white", "textDecoration": "none"})),
                        html.Li(html.A("Executive Summary", href="#executive-summary", style={"color": "white", "textDecoration": "none"})),
                        html.Li(html.A("Input Data", href="#input-data", style={"color": "white", "textDecoration": "none"})),
                    ],
                    style={"paddingLeft": "18px", "lineHeight": "2"},
                ),
                html.Div(
                    [
                        html.H3("Penugonda Pavan Kalyan", style={"color": "#eee", "margin": "0", "fontSize": "18px"}),
                        html.Div("2504107029"),
                        html.Div("MSDSM 2025-27"),
                        html.Div("Batch 05"),
                    ],
                    style={"position": "absolute", "bottom": "24px", "left": "20px", "right": "20px", "fontSize": "14px", "color": "#bfdbfe"},
                ),
            ],
            style=SIDEBAR_STYLE,
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H1(APP_TITLE, style={"marginBottom": "6px", "color": DARK_BLUE, "fontSize": "34px"}),
                        html.P(APP_SUBTITLE, style={"marginTop": 0, "fontSize": "16px", "color": MID_GREY}),
                    ],
                    style=SECTION_CARD_STYLE,
                ),
                html.Div(
                    [
                        html.Div([html.Label("Store", style=FILTER_LABEL_STYLE), dcc.Dropdown(id="store-filter", options=store_options, value="All", clearable=False)], style={"flex": "1", "minWidth": "180px"}),
                        html.Div([html.Label("Year", style=FILTER_LABEL_STYLE), dcc.Dropdown(id="year-filter", options=year_options, value="All", clearable=False)], style={"flex": "1", "minWidth": "180px"}),
                        html.Div([html.Label("Quarter", style=FILTER_LABEL_STYLE), dcc.Dropdown(id="quarter-filter", options=quarter_options, value="All", clearable=False)], style={"flex": "1", "minWidth": "180px"}),
                        html.Div([html.Label("Holiday Flag", style=FILTER_LABEL_STYLE), dcc.Dropdown(id="holiday-filter", options=holiday_options, value="All", clearable=False)], style={"flex": "1", "minWidth": "180px"}),
                    ],
                    style={**SECTION_CARD_STYLE, "display": "flex", "gap": "16px", "flexWrap": "wrap"},
                ),
                html.Div(id="kpi-container", style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))", "gap": "16px", "marginBottom": "24px"}),
                html.Div(
                    id="data-preparation",
                    children=[
                        create_section_title("Section 1 - Data Preparation Summary", "Cleaned dataset diagnostics and descriptive statistics used for the dashboard."),
                        html.Div([html.Div(id="dataset-shape-card", style={"flex": "1"}), html.Div(id="duplicate-count-card", style={"flex": "1"})], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H3("Missing Values", style={"color": DARK_BLUE}),
                                        dash_table.DataTable(
                                            id="missing-values-table",
                                            columns=[{"name": "Column", "id": "Column"}, {"name": "Missing Values", "id": "Missing Values"}],
                                            data=prep_metadata["missing_values"].to_dict("records"),
                                            style_cell=TABLE_STYLE_CELL,
                                            style_header=TABLE_STYLE_HEADER,
                                            style_table={"overflowX": "auto"},
                                        ),
                                    ],
                                    style=SECTION_CARD_STYLE,
                                ),
                                html.Div(
                                    [
                                        html.H3("Summary Statistics", style={"color": DARK_BLUE}),
                                        dash_table.DataTable(
                                            id="summary-statistics-table",
                                            columns=[{"name": col, "id": col} for col in summary_statistics_df.columns],
                                            data=summary_statistics_df.to_dict("records"),
                                            style_cell=TABLE_STYLE_CELL,
                                            style_header=TABLE_STYLE_HEADER,
                                            style_table={"overflowX": "auto"},
                                        ),
                                    ],
                                    style=SECTION_CARD_STYLE,
                                ),
                            ]
                        ),
                    ],
                ),
                html.Div(
                    id="exploratory-analysis",
                    children=[
                        create_section_title("Section 2 - Exploratory Data Analysis", "visuals answering the questions with business context."),
                        create_chart_block("q1-graph", "q1-insight"),
                        create_chart_block("q2-graph", "q2-insight"),
                        create_chart_block("q3-graph", "q3-insight"),
                        create_chart_block("q4-graph", "q4-insight"),
                        create_chart_block("q5-graph", "q5-insight"),
                        create_chart_block("q6-graph", "q6-insight"),
                        create_chart_block("q7-graph", "q7-insight"),
                        create_chart_block("q8-graph", "q8-insight"),
                        create_chart_block("q9-graph", "q9-insight"),
                        create_chart_block("q10-graph", "q10-insight"),
                    ],
                ),
                html.Div(
                    id="advanced-analytics",
                    children=[
                        create_section_title("Section 3 - Advanced Visualization", "Heatmap-based views to reveal store-month concentration and calendar-level sales intensity."),
                        create_chart_block("q11-graph", "q11-insight"),
                        create_chart_block("q12-graph", "q12-insight"),
                    ],
                ),
                html.Div(
                    id="executive-summary",
                    children=[
                        create_section_title("Section 4 - Executive Summary", "A concise summary combining performance, drivers, and recommended actions."),
                        html.Div(id="executive-summary-card"),
                    ],
                ),
                html.Div(
                    id="input-data",
                    children=[
                        create_section_title("Input Data", "Processed dataset used for all visualisations, with interactive exploration controls."),
                        html.Div(
                            [
                                dash_table.DataTable(
                                    id="input-data-table",
                                    columns=[{"name": column, "id": column} for column in processed_df.columns],
                                    data=[],
                                    sort_action="native",
                                    filter_action="native",
                                    page_action="native",
                                    page_size=10,
                                    fixed_rows={"headers": True},
                                    style_table={"overflowX": "auto", "overflowY": "auto", "maxHeight": "500px"},
                                    style_cell=TABLE_STYLE_CELL,
                                    style_header=TABLE_STYLE_HEADER,
                                )
                            ],
                            style=SECTION_CARD_STYLE,
                        ),
                    ],
                ),
            ],
            style=CONTENT_STYLE,
        ),
    ]
)


################################################################################
# SECTION 12 - CALLBACKS
################################################################################

@app.callback(
    Output("kpi-container", "children"),
    Output("dataset-shape-card", "children"),
    Output("duplicate-count-card", "children"),
    Output("q1-graph", "figure"),
    Output("q2-graph", "figure"),
    Output("q3-graph", "figure"),
    Output("q4-graph", "figure"),
    Output("q5-graph", "figure"),
    Output("q6-graph", "figure"),
    Output("q7-graph", "figure"),
    Output("q8-graph", "figure"),
    Output("q9-graph", "figure"),
    Output("q10-graph", "figure"),
    Output("q11-graph", "figure"),
    Output("q12-graph", "figure"),
    Output("q1-insight", "children"),
    Output("q2-insight", "children"),
    Output("q3-insight", "children"),
    Output("q4-insight", "children"),
    Output("q5-insight", "children"),
    Output("q6-insight", "children"),
    Output("q7-insight", "children"),
    Output("q8-insight", "children"),
    Output("q9-insight", "children"),
    Output("q10-insight", "children"),
    Output("q11-insight", "children"),
    Output("q12-insight", "children"),
    Output("executive-summary-card", "children"),
    Output("input-data-table", "data"),
    Input("store-filter", "value"),
    Input("year-filter", "value"),
    Input("quarter-filter", "value"),
    Input("holiday-filter", "value"),
)
def update_dashboard(selected_store, selected_year, selected_quarter, selected_holiday):
    """
    Purpose:
    Update all dashboard components based on selected filters.

    Input:
    Filter values from the dashboard controls.

    Output:
    Updated KPI cards, figures, insights, executive summary, and input data table.
    """
    filtered_df = filter_dataset(processed_df, selected_store, selected_year, selected_quarter, selected_holiday)
    kpis = calculate_kpis(filtered_df)
    kpi_cards = [create_kpi_card(label, format_kpi_value(label, value)) for label, value in kpis.items()]

    dataset_shape_card = html.Div(
        [
            html.H3("Dataset Shape", style={"color": DARK_BLUE}),
            html.P(f"Rows: {filtered_df.shape[0]:,} | Columns: {filtered_df.shape[1]:,}", style={"fontSize": "22px", "fontWeight": "bold", "color": PRIMARY_BLUE}),
            html.P("This reflects the cleaned dataset after applying the active filters.", style={"color": MID_GREY}),
        ],
        style=SECTION_CARD_STYLE,
    )

    duplicate_count_card = html.Div(
        [
            html.H3("Duplicate Count Removed", style={"color": DARK_BLUE}),
            html.P(f"{prep_metadata['duplicate_count']:,}", style={"fontSize": "22px", "fontWeight": "bold", "color": PRIMARY_BLUE}),
            html.P("Duplicates were checked during preparation and removed before analysis.", style={"color": MID_GREY}),
        ],
        style=SECTION_CARD_STYLE,
    )

    insights = generate_chart_insights(filtered_df)

    if filtered_df.empty:
        empty_fig = create_empty_figure("No data available for the selected filters")
        return (
            kpi_cards,
            dataset_shape_card,
            duplicate_count_card,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            insights["Q1"],
            insights["Q2"],
            insights["Q3"],
            insights["Q4"],
            insights["Q5"],
            insights["Q6"],
            insights["Q7"],
            insights["Q8"],
            insights["Q9"],
            insights["Q10"],
            insights["Q11"],
            insights["Q12"],
            generate_executive_summary(filtered_df),
            filtered_df.to_dict("records"),
        )

    return (
        kpi_cards,
        dataset_shape_card,
        duplicate_count_card,
        create_weekly_sales_histogram(filtered_df),
        create_weekly_sales_by_store(filtered_df),
        create_monthly_sales_trend(filtered_df),
        create_holiday_sales_boxplot(filtered_df),
        create_correlation_heatmap(filtered_df),
        create_fuel_price_scatter(filtered_df),
        create_temperature_distribution(filtered_df),
        create_cpi_trend(filtered_df),
        create_unemployment_trend(filtered_df),
        create_quarter_sales_distribution(filtered_df),
        create_monthly_sales_heatmap(filtered_df),
        create_calendar_heatmap(filtered_df),
        insights["Q1"],
        insights["Q2"],
        insights["Q3"],
        insights["Q4"],
        insights["Q5"],
        insights["Q6"],
        insights["Q7"],
        insights["Q8"],
        insights["Q9"],
        insights["Q10"],
        insights["Q11"],
        insights["Q12"],
        generate_executive_summary(filtered_df),
        filtered_df.to_dict("records"),
    )


################################################################################
# SECTION 13 - RUN APPLICATION
################################################################################

if __name__ == "__main__":
    app.run(debug=True)
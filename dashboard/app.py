import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pathlib import Path
import json

import numpy as np
import pandas as pd
import rasterio
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from src.config import (
    PROJECT_ROOT,
    DATA_DIR,
    LULC_DIR,
    ECOLOGICAL_DIR,
    STATISTICS_DIR,
    OUTPUT_DIR,
    MAPS_DIR,
    CHARTS_DIR,
    REPORTS_DIR,
    YEARS,
    LULC_CLASSES,
    CLASS_COLORS
)


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NMMC | LULC & Ecological Impact",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. DASHBOARD CONFIGURATION
# =========================================================

LULC_FILES = {
    1994: LULC_DIR / "NMMC_LULC_1994_PRELIMINARY.tif",
    2004: LULC_DIR / "NMMC_LULC_2004_PRELIMINARY.tif",
    2014: LULC_DIR / "NMMC_LULC_2014_PRELIMINARY.tif",
    2024: LULC_DIR / "NMMC_2024_LULC_SCIENTIFIC_FINAL.tif",
}

OPTICAL_FILES = {
    year: ECOLOGICAL_DIR / "optical_indices" /
    f"NMMC_Optical_Indices_{year}.tif"
    for year in YEARS
}

THERMAL_FILES = {
    year: ECOLOGICAL_DIR / "thermal_indicators" /
    f"NMMC_Thermal_Indicators_{year}.tif"
    for year in YEARS
}

ANNUAL_LULC_FILE = (
    REPORTS_DIR / "NMMC_LULC_annual_area_statistics.csv"
)

TRANSITION_FILE = (
    REPORTS_DIR / "NMMC_LULC_all_transition_details.csv"
)

CLASS_CHANGE_FILE = (
    REPORTS_DIR / "NMMC_LULC_class_gains_losses.csv"
)

ECOLOGICAL_FILE = (
    REPORTS_DIR / "ecological_annual_raster_statistics.csv"
)

ECOLOGICAL_CHANGE_FILE = (
    REPORTS_DIR / "ecological_temporal_changes_1994_2024.csv"
)

ECOLOGICAL_TREND_FILE = (
    REPORTS_DIR / "ecological_temporal_trends.csv"
)

VALIDATION_SUMMARY_FILE = (
    REPORTS_DIR / "NMMC_validation_summary.json"
)

VALIDATION_RESULTS_FILE = (
    REPORTS_DIR / "NMMC_validation_results.csv"
)

TRANSITION_PERIODS = [
    "1994–2004",
    "2004–2014",
    "2014–2024",
    "1994–2024",
]

PERIOD_YEARS = {
    "1994–2004": (1994, 2004),
    "2004–2014": (2004, 2014),
    "2014–2024": (2014, 2024),
    "1994–2024": (1994, 2024),
}


# =========================================================
# 3. CUSTOM STYLING
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 750;
        margin-bottom: 0.1rem;
    }

    .subtitle {
        color: #8b949e;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 1rem;
        margin-bottom: 0.7rem;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.07);
        padding: 15px;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 4. DATA LOADING FUNCTIONS
# =========================================================

@st.cache_data(show_spinner=False)
def load_csv(path):

    path = Path(path)

    if not path.exists():
        return None

    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path):

    path = Path(path)

    if not path.exists():
        return None

    with open(path, "r") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def load_raster(path):

    path = Path(path)

    if not path.exists():
        return None

    with rasterio.open(path) as src:

        data = src.read(1, masked=True)

        metadata = {
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "resolution": src.res,
            "nodata": src.nodata,
            "bounds": tuple(src.bounds),
            "transform": tuple(src.transform),
        }

    return data, metadata


@st.cache_data(show_spinner=False)
def load_transition_raster(start_year, end_year):

    path = (
        MAPS_DIR
        / f"NMMC_TRANSITION_{start_year}_{end_year}.tif"
    )

    return load_raster(path)


# =========================================================
# 5. COMMON HELPER FUNCTIONS
# =========================================================

def format_number(value, decimals=2):

    if pd.isna(value):
        return "N/A"

    return f"{value:,.{decimals}f}"


def section_heading(title, description=None):

    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )

    if description:
        st.caption(description)


def show_missing_data(message):

    st.info(message)


def make_class_legend():

    handles = []

    for code, name in LULC_CLASSES.items():

        handles.append(
            Patch(
                facecolor=CLASS_COLORS[code],
                label=name,
            )
        )

    return handles


def raster_map_figure(data, title):

    values = np.ma.asarray(data)

    fig, ax = plt.subplots(figsize=(7, 8))

    cmap = ListedColormap([
        CLASS_COLORS[code]
        for code in sorted(LULC_CLASSES)
    ])

    cmap.set_bad(color="white")

    norm = BoundaryNorm(
        np.arange(-0.5, len(LULC_CLASSES) + 0.5, 1),
        cmap.N,
    )

    ax.imshow(
        values,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Raster Column")
    ax.set_ylabel("Raster Row")

    ax.legend(
        handles=make_class_legend(),
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=8,
    )

    fig.tight_layout()

    return fig


def show_chart(fig):

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =========================================================
# 6. SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🌍 NMMC Analytics")

    st.caption(
        "Multitemporal Land Use/Land Cover "
        "& Ecological Impact Study"
    )

    st.divider()

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "LULC Analysis",
            "Change Detection",
            "Ecological Analysis",
            "Data Validation",
        ],
    )

    st.divider()

    st.markdown("**Study Information**")

    st.write("📍 Navi Mumbai Municipal Corporation")
    st.write("📅 1994 – 2024")
    st.write("🛰️ Landsat 5, 7, 8 and 9")
    st.write("📐 30 m optical resolution")

    st.divider()

    st.caption("NMMC Research Project")


# =========================================================
# 7. DASHBOARD HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    'Navi Mumbai: Land Use, Land Cover & Ecological Impact'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Multitemporal remote sensing analysis | 1994–2024'
    '</div>',
    unsafe_allow_html=True,
)


# =========================================================
# 8. OVERVIEW PAGE
# =========================================================

def render_overview():

    section_heading(
        "Project Overview",
        "A 30-year assessment of land-cover transformation "
        "and ecological indicators across NMMC.",
    )

    annual = load_csv(ANNUAL_LULC_FILE)
    ecological = load_csv(ECOLOGICAL_FILE)
    validation = load_json(VALIDATION_SUMMARY_FILE)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Study Period", "30 Years")
    col2.metric("Study Years", "4")
    col3.metric("LULC Classes", "5")
    col4.metric("Spatial Resolution", "30 m")

    st.divider()

    section_heading("Study Framework")

    st.markdown(
        """
        This project examines land-use and land-cover changes
        across Navi Mumbai Municipal Corporation using
        multitemporal Landsat satellite imagery.

        The analysis integrates:

        - Land Use/Land Cover classification
        - Multitemporal change detection
        - Vegetation and water indices
        - Land Surface Temperature
        - Urban thermal variation (UTFVI)
        - Automated data quality validation
        """
    )

    st.divider()

    section_heading("Key Findings: 1994–2024")

    if annual is not None:

        start = annual[
            annual["year"] == 1994
        ]

        end = annual[
            annual["year"] == 2024
        ]

        if not start.empty and not end.empty:

            merged = start.merge(
                end,
                on="class_code",
                suffixes=("_1994", "_2024"),
            )

            cols = st.columns(5)

            for index, row in merged.iterrows():

                col = cols[index % 5]

                change = (
                    row["percentage_2024"]
                    - row["percentage_1994"]
                )

                col.metric(
                    LULC_CLASSES[int(row["class_code"])],
                    f"{row['percentage_2024']:.1f}%",
                    f"{change:+.1f} pp",
                )

    else:

        show_missing_data(
            "Annual LULC statistics are not available."
        )

    st.divider()

    section_heading("Ecological Indicators")

    if ecological is not None:

        latest = ecological[
            ecological["year"] == 2024
        ]

        columns = st.columns(4)

        for index, indicator in enumerate(
            ["NDVI", "MNDWI", "LST_C", "UTFVI"]
        ):

            row = latest[
                latest["indicator"] == indicator
            ]

            if not row.empty:

                columns[index].metric(
                    indicator,
                    format_number(row.iloc[0]["mean"], 4),
                )

    if validation is not None:

        st.divider()

        section_heading("Data Quality Status")

        summary = validation.get("summary", {})

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Validation Status",
            validation.get("overall_status", "N/A"),
        )

        c2.metric(
            "Checks Passed",
            summary.get("PASS", 0),
        )

        c3.metric(
            "Failures",
            summary.get("FAIL", 0),
        )

    st.caption(
        "Note: Historical classification outputs are preliminary "
        "and require independent accuracy assessment."
    )


# =========================================================
# 9. LULC ANALYSIS PAGE
# =========================================================

def render_lulc_analysis():

    section_heading(
        "Land Use / Land Cover Analysis",
        "Explore classified land-cover maps and area "
        "distribution for each study year.",
    )

    year = st.selectbox(
        "Select Study Year",
        YEARS,
        index=3,
    )

    raster_result = load_raster(
        LULC_FILES[year]
    )

    annual = load_csv(ANNUAL_LULC_FILE)

    if raster_result is None:

        show_missing_data(
            f"LULC raster for {year} is missing."
        )

        return

    data, metadata = raster_result

    col1, col2 = st.columns([1.4, 1])

    with col1:

        section_heading("Classified LULC Map")

        fig = raster_map_figure(
            data,
            f"NMMC LULC Classification — {year}",
        )

        st.pyplot(fig)
        plt.close(fig)

    with col2:

        section_heading("Raster Information")

        st.write(f"**CRS:** {metadata['crs']}")
        st.write(
            f"**Dimensions:** "
            f"{metadata['width']} × {metadata['height']}"
        )

        st.write(
            f"**Resolution:** {metadata['resolution'][0]} m"
        )

        st.write(
            f"**Valid pixels:** "
            f"{data.count():,}"
        )

        st.divider()

        section_heading("Class Distribution")

        if annual is not None:

            subset = annual[
                annual["year"] == year
            ].sort_values("class_code")

            if not subset.empty:

                fig = px.pie(
                    subset,
                    names="class_name",
                    values="percentage",
                    color="class_name",
                    color_discrete_map={
                        name: CLASS_COLORS[code]
                        for code, name in LULC_CLASSES.items()
                    },
                    hole=0.35,
                )

                fig.update_layout(
                    height=400,
                    margin=dict(l=10, r=10, t=30, b=10),
                )

                show_chart(fig)

    st.divider()

    section_heading("Annual LULC Statistics")

    if annual is not None:

        table = annual[
            annual["year"] == year
        ][
            [
                "class_name",
                "pixel_count",
                "area_km2",
                "percentage",
            ]
        ].copy()

        table.columns = [
            "Class",
            "Pixel Count",
            "Area (km²)",
            "Percentage (%)",
        ]

        st.dataframe(
            table.round(3),
            use_container_width=True,
            hide_index=True,
        )

    else:

        show_missing_data(
            "Annual LULC statistics CSV is missing."
        )

    st.divider()

    if annual is not None:

        section_heading(
            "LULC Area Distribution Across Years"
        )

        fig = px.line(
            annual,
            x="year",
            y="area_km2",
            color="class_name",
            markers=True,
            color_discrete_map={
                name: CLASS_COLORS[code]
                for code, name in LULC_CLASSES.items()
            },
        )

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Area (km²)",
            hovermode="x unified",
        )

        show_chart(fig)


# =========================================================
# 10. CHANGE DETECTION PAGE
# =========================================================

def render_change_detection():

    section_heading(
        "LULC Change Detection",
        "Investigate land-cover transitions between "
        "1994, 2004, 2014 and 2024.",
    )

    transitions = load_csv(TRANSITION_FILE)
    changes = load_csv(CLASS_CHANGE_FILE)

    if transitions is None or changes is None:

        show_missing_data(
            "Transition analysis reports are missing. "
            "Run src.change_detection first."
        )

        return

    period = st.selectbox(
        "Select Comparison Period",
        TRANSITION_PERIODS,
        index=3,
    )

    start_year, end_year = PERIOD_YEARS[period]

    subset = transitions[
        (transitions["start_year"] == start_year)
        & (transitions["end_year"] == end_year)
    ]

    change_subset = changes[
        (changes["start_year"] == start_year)
        & (changes["end_year"] == end_year)
    ]

    if subset.empty:

        show_missing_data(
            "No transition data available for this period."
        )

        return

    section_heading("Transition Matrix")

    matrix = subset.pivot(
        index="from_class",
        columns="to_class",
        values="area_km2",
    )

    matrix = matrix.reindex(
        index=[
            LULC_CLASSES[c]
            for c in LULC_CLASSES
        ],
        columns=[
            LULC_CLASSES[c]
            for c in LULC_CLASSES
        ],
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale="YlGnBu",
            colorbar=dict(
                title="Area (km²)"
            ),
            text=np.round(matrix.values, 2),
            texttemplate="%{text}",
            hovertemplate=(
                "From: %{y}<br>"
                "To: %{x}<br>"
                "Area: %{z:.3f} km²"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        xaxis_title="Destination Class",
        yaxis_title="Original Class",
        height=550,
    )

    show_chart(fig)

    st.caption(
        "Diagonal cells represent unchanged classes. "
        "Off-diagonal cells represent mapped class transitions."
    )

    st.divider()

    section_heading("Class Gains and Losses")

    fig = go.Figure()

    for column, label in [
        ("gain_km2", "Gain"),
        ("loss_km2", "Loss"),
    ]:

        fig.add_trace(
            go.Bar(
                name=label,
                x=change_subset["class_name"],
                y=change_subset[column],
            )
        )

    fig.update_layout(
        barmode="group",
        xaxis_title="Land-cover Class",
        yaxis_title="Area (km²)",
        height=430,
    )

    show_chart(fig)

    section_heading("Net Class Change")

    net = change_subset.copy()

    fig = px.bar(
        net,
        x="class_name",
        y="net_change_km2",
        color="net_change_km2",
        color_continuous_scale="RdYlGn",
        labels={
            "class_name": "Class",
            "net_change_km2": "Net Change (km²)",
        },
    )

    fig.add_hline(y=0, line_dash="dash")

    show_chart(fig)

    st.dataframe(
        change_subset[
            [
                "class_name",
                "gain_km2",
                "loss_km2",
                "net_change_km2",
            ]
        ].round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    section_heading("Spatial Transition Map")

    transition_result = load_transition_raster(
        start_year,
        end_year,
    )

    if transition_result is not None:

        transition_data, metadata = transition_result

        values = np.asarray(transition_data.data)

        valid = (
            ~np.ma.getmaskarray(transition_data)
            & (values != 255)
        )

        codes = sorted(
            np.unique(values[valid]).tolist()
        )

        if codes:

            colors = plt.get_cmap("tab20")(
                np.linspace(0, 1, len(codes))
            )

            lookup = np.full(
                values.shape,
                -1,
                dtype=np.int16,
            )

            for index, code in enumerate(codes):

                lookup[values == code] = index

            lookup = np.ma.masked_where(
                lookup < 0,
                lookup,
            )

            fig, ax = plt.subplots(figsize=(7, 9))

            cmap = ListedColormap(colors)
            cmap.set_bad("white")

            ax.imshow(
                lookup,
                cmap=cmap,
                interpolation="nearest",
            )

            ax.set_title(
                f"Spatial LULC Transitions: {period}"
            )

            ax.set_xlabel("Raster Column")
            ax.set_ylabel("Raster Row")

            legend = []

            for index, code in enumerate(codes):

                from_code = code // 10
                to_code = code % 10

                label = (
                    f"{LULC_CLASSES[from_code]} → "
                    f"{LULC_CLASSES[to_code]}"
                )

                legend.append(
                    Patch(
                        facecolor=colors[index],
                        label=label,
                    )
                )

            ax.legend(
                handles=legend,
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                fontsize=7,
            )

            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

    else:

        show_missing_data(
            "Transition GeoTIFF is missing. "
            "Run src.change_detection first."
        )


# =========================================================
# 11. ECOLOGICAL ANALYSIS PAGE
# =========================================================

def render_ecological_analysis():

    section_heading(
        "Ecological Analysis",
        "Explore vegetation, water, built-up and thermal "
        "indicators derived from Landsat imagery.",
    )

    annual = load_csv(ECOLOGICAL_FILE)
    changes = load_csv(ECOLOGICAL_CHANGE_FILE)
    trends = load_csv(ECOLOGICAL_TREND_FILE)

    if annual is None:

        show_missing_data(
            "Ecological statistics are missing. "
            "Run src.ecological_analysis first."
        )

        return

    indicator = st.selectbox(
        "Select Ecological Indicator",
        [
            "NDVI",
            "NDBI",
            "NDMI",
            "NDWI",
            "MNDWI",
            "LST_C",
            "UTFVI",
        ],
    )

    indicator_data = annual[
        annual["indicator"] == indicator
    ].sort_values("year")

    if indicator_data.empty:

        show_missing_data(
            f"No data found for {indicator}."
        )

        return

    is_temperature = indicator == "LST_C"

    unit = "°C" if is_temperature else "Index Value"

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=indicator_data["year"],
            y=indicator_data["mean"],
            mode="lines+markers",
            name="Mean",
            line=dict(width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=indicator_data["year"],
            y=indicator_data["mean"] + indicator_data["std"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=indicator_data["year"],
            y=indicator_data["mean"] - indicator_data["std"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(100,150,200,0.18)",
            line=dict(width=0),
            name="±1 Standard Deviation",
        )
    )

    fig.update_layout(
        title=f"{indicator} Temporal Variation (1994–2024)",
        xaxis_title="Year",
        yaxis_title=unit,
        xaxis=dict(
            tickmode="array",
            tickvals=YEARS,
        ),
        hovermode="x unified",
        height=480,
    )

    show_chart(fig)

    st.divider()

    section_heading("Annual Indicator Statistics")

    st.dataframe(
        indicator_data,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    section_heading("Spatial Indicator Raster")

    selected_year = st.selectbox(
        "Select Raster Year",
        YEARS,
        index=3,
        key="ecological_raster_year",
    )

    raster_path = None
    band_name = None

    if indicator in ["LST_C", "UTFVI"]:

        raster_path = THERMAL_FILES[selected_year]
        band_name = indicator

    else:

        raster_path = OPTICAL_FILES[selected_year]
        band_name = indicator

    raster_result = load_raster(raster_path)

    if raster_result is not None:

        raster_data, metadata = raster_result

        with rasterio.open(raster_path) as src:

            band_names = [
                description
                for description in src.descriptions
            ]

            if band_name in band_names:

                band_index = (
                    band_names.index(band_name) + 1
                )

                raster_data = src.read(
                    band_index,
                    masked=True,
                )

            else:

                raster_data = None

        if raster_data is not None:

            fig, ax = plt.subplots(figsize=(8, 6))

            image = ax.imshow(
                raster_data,
                cmap=(
                    "RdYlGn"
                    if indicator == "NDVI"
                    else "coolwarm"
                    if indicator == "LST_C"
                    else "viridis"
                ),
            )

            ax.set_title(
                f"{indicator} — {selected_year}"
            )

            ax.set_xlabel("Raster Column")
            ax.set_ylabel("Raster Row")

            fig.colorbar(
                image,
                ax=ax,
                label=unit,
                shrink=0.75,
            )

            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

        else:

            show_missing_data(
                f"Band {indicator} was not found in the raster."
            )

    else:

        show_missing_data(
            f"Ecological raster missing for {selected_year}."
        )

    st.divider()

    if changes is not None:

        section_heading(
            "Long-Term Change: 1994–2024"
        )

        change_row = changes[
            changes["indicator"] == indicator
        ]

        if not change_row.empty:

            row = change_row.iloc[0]

            st.metric(
                "Mean Change",
                format_number(row["mean_change"], 4),
                help="Difference between endpoint means on common valid pixels.",
            )

            st.write(
                f"Common valid pixels: "
                f"{int(row['common_valid_pixels']):,}"
            )

    if trends is not None:

        trend_row = trends[
            trends["indicator"] == indicator
        ]

        if not trend_row.empty:

            row = trend_row.iloc[0]

            st.metric(
                "Descriptive Slope per Decade",
                format_number(row["slope_per_decade"], 4),
            )

            st.caption(
                "Descriptive linear trend using four observation years. "
                "This is not a statistical significance test."
            )

    if indicator in ["NDBI", "NDMI"]:

        st.warning(
            "In this implementation, NDBI and NDMI are mathematical "
            "negatives of one another because they use the same two "
            "bands in reverse order. Interpret them as one independent "
            "spectral signal, not two separate indicators."
        )

    if indicator == "LST_C":

        st.info(
            "LST values represent January–April seasonal composite "
            "surface temperatures, not annual mean air temperatures."
        )

    if indicator == "UTFVI":

        st.info(
            "UTFVI represents spatial thermal variation relative to "
            "the mean LST. It should not be interpreted directly as "
            "a human thermal-comfort measurement."
        )


# =========================================================
# 12. DATA VALIDATION PAGE
# =========================================================

def render_validation():

    section_heading(
        "Data Validation & Quality Control",
        "Automated checks for raster integrity, analytical "
        "consistency and transition calculations.",
    )

    summary = load_json(VALIDATION_SUMMARY_FILE)
    results = load_csv(VALIDATION_RESULTS_FILE)

    if summary is None:

        show_missing_data(
            "Validation summary is missing. "
            "Run python3 -m src.validation."
        )

        return

    counts = summary.get("summary", {})

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Overall Status",
        summary.get("overall_status", "N/A"),
    )

    c2.metric(
        "Passed",
        counts.get("PASS", 0),
    )

    c3.metric(
        "Warnings",
        counts.get("WARNING", 0),
    )

    c4.metric(
        "Failures",
        counts.get("FAIL", 0),
    )

    st.divider()

    section_heading("Validation Distribution")

    chart_data = pd.DataFrame({
        "Status": ["PASS", "WARNING", "FAIL", "INFO"],
        "Count": [
            counts.get("PASS", 0),
            counts.get("WARNING", 0),
            counts.get("FAIL", 0),
            counts.get("INFO", 0),
        ],
    })

    fig = px.pie(
        chart_data,
        names="Status",
        values="Count",
        hole=0.4,
        color="Status",
        color_discrete_map={
            "PASS": "#2ca02c",
            "WARNING": "#f0ad4e",
            "FAIL": "#d62728",
            "INFO": "#7f8c8d",
        },
    )

    show_chart(fig)

    st.divider()

    section_heading("Validation Details")

    if results is not None:

        status_filter = st.multiselect(
            "Filter by status",
            ["PASS", "WARNING", "FAIL", "INFO"],
            default=["WARNING", "FAIL"],
        )

        filtered = results[
            results["status"].isin(status_filter)
        ]

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
        )

        if not filtered.empty:

            csv_data = filtered.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download filtered validation results",
                data=csv_data,
                file_name="NMMC_validation_results_filtered.csv",
                mime="text/csv",
            )

    st.divider()

    section_heading("Methodological Limitations")

    st.warning(
        """
        - Historical classifications use a 2024-trained model transferred
          to earlier Landsat imagery.
        - Independent historical accuracy assessment is still required.
        - NDBI and NDMI are redundant in the current implementation.
        - Local full-pixel area estimates and GEE boundary-based area
          estimates use different area/grid treatments.
        """
    )

    st.caption(
        summary.get("validation_scope", "")
    )


# =========================================================
# 13. PAGE ROUTING
# =========================================================

if page == "Overview":

    render_overview()

elif page == "LULC Analysis":

    render_lulc_analysis()

elif page == "Change Detection":

    render_change_detection()

elif page == "Ecological Analysis":

    render_ecological_analysis()

elif page == "Data Validation":

    render_validation()


# =========================================================
# 14. FOOTER
# =========================================================

st.divider()

st.caption(
    "NMMC Multitemporal LULC & Ecological Impact Analysis | "
    "1994–2024 | Remote Sensing & Geospatial Analytics"
)

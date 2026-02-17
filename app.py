import json
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Volcanoes of the World", layout="wide")
st.title("Volcanoes of the World")

DATA_PATH = "data/volcano_ds_pop.csv"
GEOJSON_PATH = "data/countries.geojson"


@st.cache_data(show_spinner="Loading volcano dataset...")
def load_volcano_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


@st.cache_data(show_spinner="Loading GeoJSON...")
def load_geojson(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_unique_sorted(series: pd.Series) -> List[str]:
    vals = [x for x in series.dropna().unique().tolist() if str(x).strip() != ""]
    return sorted(vals)


def coerce_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series([np.nan] * len(df))


def find_best_geojson_key_for_countries(
    geojson_obj: dict,
    countries: pd.Series,
    candidate_keys: Optional[List[str]] = None,
) -> Tuple[Optional[str], float]:
    if not geojson_obj or "features" not in geojson_obj or not geojson_obj["features"]:
        return None, 0.0

    if candidate_keys is None:
        candidate_keys = ["ADMIN", "NAME", "name", "Country", "COUNTRY", "SOVEREIGNT", "ISO_A3", "SOV_A3"]

    ds_countries = set([str(x).strip() for x in countries.dropna().unique().tolist() if str(x).strip()])
    if not ds_countries:
        return None, 0.0

    features = geojson_obj["features"][:100]
    available_keys = set()
    for feat in features:
        available_keys.update(feat.get("properties", {}).keys())

    keys_to_try = [k for k in candidate_keys if k in available_keys] or list(available_keys)

    best_key, best_score = None, 0.0
    for key in keys_to_try:
        gj_vals = set()
        for feat in geojson_obj["features"]:
            v = feat.get("properties", {}).get(key, None)
            if v is None:
                continue
            gj_vals.add(str(v).strip())

        if not gj_vals:
            continue

        score = len(ds_countries.intersection(gj_vals)) / max(1, len(ds_countries))
        if score > best_score:
            best_score, best_key = score, key

    return best_key, best_score


def activity_weight_from_status(status: str) -> float:
    """
    You can tune these later.
    Your dataset shows values like: Holocene, Historical, Tephrochronology, Unknown.
    """
    s = str(status).strip().lower()
    if s in {"historical"}:
        return 1.0
    if s in {"holocene"}:
        return 0.7
    if s in {"tephrochronology"}:
        return 0.45
    if s in {"unknown", "nan"}:
        return 0.35
    return 0.5


def activity_weight_from_last_known(last_known: str) -> float:
    """
    Your dataset has values like: Unknown, U, D1, D2, D6.
    Not sure what the codebook is, so keep this conservative.
    """
    lk = str(last_known).strip().upper()
    if lk in {"U", "UNKNOWN", "NAN"}:
        return 0.6
    if lk.startswith("D"):
        # D1 more recent than D6 (assumption). Tune if you have a legend.
        try:
            n = int(lk[1:])
            return float(np.clip(1.0 - (n - 1) * 0.08, 0.55, 1.0))
        except Exception:
            return 0.7
    return 0.7


def build_map(
    df: pd.DataFrame,
    geojson_obj: Optional[dict],
    choropleth_mode: str,
    choropleth_opacity: float,
    color_mode: str,
) -> go.Figure:
    fig = go.Figure()

    # Choropleth layer
    if choropleth_mode != "Off" and geojson_obj is not None and "Country" in df.columns and len(df) > 0:
        key, score = find_best_geojson_key_for_countries(geojson_obj, df["Country"])

        if key is not None and score >= 0.20:
            if choropleth_mode == "Volcano count":
                agg = df.groupby("Country", dropna=True).size().reset_index(name="Metric")
                z_title = "Volcanoes"
            else:
                agg = df.groupby("Country", dropna=True)["Risk score"].sum().reset_index(name="Metric")
                z_title = "Risk sum"

            fig.add_trace(
                go.Choroplethmapbox(
                    geojson=geojson_obj,
                    locations=agg["Country"],
                    z=agg["Metric"],
                    featureidkey=f"properties.{key}",
                    colorscale="YlOrRd",
                    marker_opacity=choropleth_opacity,
                    marker_line_width=0.2,
                    hovertemplate="<b>%{location}</b><br>" + z_title + ": %{z:.2f}<extra></extra>",
                    name=z_title,
                    showscale=True,
                )
            )
        else:
            st.info(
                "Choropleth not shown: GeoJSON country names did not match your dataset well enough. "
                "If you can add ISO3 codes (or a consistent country key), it will be reliable."
            )

    # Point layer settings
    color_col = "Status" if color_mode == "Status" else "Type"
    if color_col not in df.columns:
        color_col = None

    # Marker size: Elev by default
    elev = coerce_numeric(df, "Elev")
    if elev.notna().any():
        e = elev.fillna(elev.min())
        e_min, e_max = float(e.min()), float(e.max())
        size = 6 + 12 * (e - e_min) / max(1e-9, (e_max - e_min))
    else:
        size = pd.Series([8.0] * len(df), index=df.index)

    hover_cols = [c for c in ["Number", "Country", "Region", "Type", "Status", "Last Known", "Elev", "Population (2020)", "Risk score"]
                  if c in df.columns]

    if color_col is None:
        customdata = df[hover_cols].to_numpy()
        hovertemplate = "<b>%{text}</b><br>" + "<br>".join(
            [f"{c}: %{{customdata[{i}]}}" for i, c in enumerate(hover_cols)]
        ) + "<extra></extra>"

        fig.add_trace(
            go.Scattermapbox(
                lat=df["Latitude"],
                lon=df["Longitude"],
                mode="markers",
                marker=dict(size=size, color="#1f77b4", opacity=0.85),
                text=df["Volcano Name"] if "Volcano Name" in df.columns else df["Number"] if "Number" in df.columns else None,
                customdata=customdata,
                hovertemplate=hovertemplate,
                name="Volcanoes",
            )
        )
    else:
        categories = safe_unique_sorted(df[color_col])
        palette = px.colors.qualitative.Plotly

        for i, cat in enumerate(categories):
            sub = df[df[color_col] == cat]
            if sub.empty:
                continue

            customdata = sub[hover_cols].to_numpy()
            hovertemplate = "<b>%{text}</b><br>" + "<br>".join(
                [f"{c}: %{{customdata[{j}]}}" for j, c in enumerate(hover_cols)]
            ) + "<extra></extra>"

            fig.add_trace(
                go.Scattermapbox(
                    lat=sub["Latitude"],
                    lon=sub["Longitude"],
                    mode="markers",
                    marker=dict(size=size.loc[sub.index], color=palette[i % len(palette)], opacity=0.85),
                    text=sub["Volcano Name"] if "Volcano Name" in sub.columns else sub["Number"],
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                    name=str(cat),
                )
            )

    fig.update_layout(
        title=dict(text="<b>Volcanoes of the World</b>", x=0.5, xanchor="center"),
        height=850,
        margin=dict(l=0, r=0, t=70, b=0),
        mapbox=dict(style="open-street-map", zoom=1.2, center=dict(lat=10, lon=0)),
        legend=dict(bgcolor="rgba(255,255,255,0.8)"),
    )
    return fig


# ----------------------------
# Load and prep data
# ----------------------------
volcano_df = load_volcano_data(DATA_PATH)
volcano_geojson = load_geojson(GEOJSON_PATH)

# Coerce numeric fields you actually have
volcano_df["Elev"] = coerce_numeric(volcano_df, "Elev")
volcano_df["Population (2020)"] = coerce_numeric(volcano_df, "Population (2020)")

# Basic cleaning
volcano_df = volcano_df.dropna(subset=["Latitude", "Longitude"]).copy()

# Build a risk score (simple proxy)
status_w = volcano_df["Status"].apply(activity_weight_from_status) if "Status" in volcano_df.columns else 0.6
lk_w = volcano_df["Last Known"].apply(activity_weight_from_last_known) if "Last Known" in volcano_df.columns else 0.7

# Exposure: log scale to avoid single huge countries dominating
pop = volcano_df["Population (2020)"].fillna(0.0)
exposure = np.log10(pop + 10)  # +10 prevents log(0)

volcano_df["Risk score"] = exposure * status_w * lk_w


# ----------------------------
# Sidebar filters
# ----------------------------
with st.sidebar:
    st.header("Filters")

    q = st.text_input("Search volcano name", value="")

    selected_types = None
    if "Type" in volcano_df.columns:
        types = safe_unique_sorted(volcano_df["Type"])
        selected_types = st.multiselect("Type", options=types, default=types)

    selected_status = None
    if "Status" in volcano_df.columns:
        statuses = safe_unique_sorted(volcano_df["Status"])
        selected_status = st.multiselect("Status", options=statuses, default=statuses)

    selected_countries = None
    if "Country" in volcano_df.columns:
        countries = safe_unique_sorted(volcano_df["Country"])
        selected_countries = st.multiselect("Country", options=countries, default=[])

    selected_regions = None
    if "Region" in volcano_df.columns:
        regions = safe_unique_sorted(volcano_df["Region"])
        selected_regions = st.multiselect("Region", options=regions, default=[])

    # Population slider
    pop_series = volcano_df["Population (2020)"]
    if pop_series.notna().any():
        pmin, pmax = int(pop_series.min()), int(pop_series.max())
        pop_range = st.slider("Population (2020) range", min_value=pmin, max_value=pmax, value=(pmin, pmax))
    else:
        pop_range = None

    # Elev slider
    elev_series = volcano_df["Elev"]
    if elev_series.notna().any():
        emin, emax = int(elev_series.min()), int(elev_series.max())
        elev_range = st.slider("Elevation range (m)", min_value=emin, max_value=emax, value=(emin, emax))
    else:
        elev_range = None

    st.divider()
    color_mode = st.radio("Color points by", options=["Status", "Type"], horizontal=True)
    choropleth_mode = st.selectbox("Choropleth", options=["Off", "Volcano count", "Risk sum"], index=1)
    choropleth_opacity = st.slider("Choropleth opacity", 0.10, 0.70, 0.35, 0.05)


# Apply filters
df = volcano_df.copy()

if q and "Volcano Name" in df.columns:
    df = df[df["Volcano Name"].astype(str).str.contains(q, case=False, na=False)]

if selected_types is not None and "Type" in df.columns:
    df = df[df["Type"].isin(selected_types)]

if selected_status is not None and "Status" in df.columns:
    df = df[df["Status"].isin(selected_status)]

if selected_countries and "Country" in df.columns:
    df = df[df["Country"].isin(selected_countries)]

if selected_regions and "Region" in df.columns:
    df = df[df["Region"].isin(selected_regions)]

if pop_range is not None:
    df = df[df["Population (2020)"].between(pop_range[0], pop_range[1], inclusive="both")]

if elev_range is not None:
    df = df[df["Elev"].between(elev_range[0], elev_range[1], inclusive="both")]


# ----------------------------
# KPI row
# ----------------------------
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Volcanoes", f"{len(df):,}")
c2.metric("Countries", f"{df['Country'].nunique():,}" if "Country" in df.columns else "n/a")
c3.metric("Types", f"{df['Type'].nunique():,}" if "Type" in df.columns else "n/a")
c4.metric("Statuses", f"{df['Status'].nunique():,}" if "Status" in df.columns else "n/a")
c5.metric("Total risk (sum)", f"{df['Risk score'].sum():.2f}" if "Risk score" in df.columns else "n/a")

st.caption("All views reflect current filters.")


# ----------------------------
# Tabs
# ----------------------------
tab_map, tab_insights, tab_data = st.tabs(["Map", "Insights", "Data"])

with tab_map:
    fig = build_map(
        df=df,
        geojson_obj=volcano_geojson,
        choropleth_mode=choropleth_mode,
        choropleth_opacity=choropleth_opacity,
        color_mode=color_mode,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

with tab_insights:
    left, right = st.columns(2)

    if "Country" in df.columns and len(df) > 0:
        top_countries = (
            df.groupby("Country", dropna=True)["Risk score"]
            .sum()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        with left:
            st.plotly_chart(
                px.bar(top_countries, x="Risk score", y="Country", orientation="h",
                       title="Top countries by risk sum (filtered)"),
                use_container_width=True,
            )

    if "Type" in df.columns and len(df) > 0:
        by_type = (
            df.groupby("Type", dropna=True)["Risk score"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        with right:
            st.plotly_chart(
                px.bar(by_type.head(15), x="Risk score", y="Type", orientation="h",
                       title="Top types by risk sum (filtered)"),
                use_container_width=True,
            )

    st.subheader("Highest risk volcanoes (filtered)")
    cols = [c for c in ["Number", "Volcano Name", "Country", "Region", "Type", "Status", "Last Known", "Elev", "Population (2020)", "Risk score"]
            if c in df.columns]
    st.dataframe(df.sort_values("Risk score", ascending=False)[cols].head(25), use_container_width=True)

with tab_data:
    st.subheader("Filtered dataset")
    default_cols = [c for c in ["Number", "Volcano Name", "Country", "Region", "Latitude", "Longitude",
                               "Elev", "Type", "Status", "Last Known", "Population (2020)", "Risk score"]
                    if c in df.columns]
    shown_cols = st.multiselect("Columns to show", options=df.columns.tolist(), default=default_cols or df.columns.tolist())
    st.dataframe(df[shown_cols], use_container_width=True, height=520)

    csv_bytes = df[shown_cols].to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", data=csv_bytes, file_name="volcanoes_filtered.csv", mime="text/csv")


with st.expander("Notes"):
    st.write(
        """
- Risk score here is a simple proxy: log10(population) × status weight × last-known weight.
- If you know what Last Known codes (D1, D2, D6, etc.) mean, you should tune the weighting.
- Choropleth depends on GeoJSON country identifiers matching your `Country` strings.
"""
    )

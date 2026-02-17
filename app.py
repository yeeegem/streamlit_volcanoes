import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from urllib.request import urlopen
import json

st.set_page_config(page_title="Volcanoes of the World", layout="wide")

st.title("Volcanoes of the World")

@st.cache_data(show_spinner="Loading volcano dataset...")
def load_volcano_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)

@st.cache_data(show_spinner="Loading GeoJSON...")
def load_geojson(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Optional: cache the figure too (useful if it is expensive to build)
@st.cache_data(show_spinner="Building map...")
def make_figure(df: pd.DataFrame):
    fig = px.scatter_map(
        df,
        lat="Latitude",
        lon="Longitude",
        color="Type",
        hover_name="Volcano Name",
        hover_data=["Type", "Country", "Region", "Status"],
        zoom=1.5,
        title="<b>Volcanoes of the World</b>",
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )

    fig.update_layout(
        title={"font_size": 20, "xanchor": "center", "x": 0.38, "yanchor": "bottom", "y": 0.95},
        title_font=dict(size=24, color="Black", family="Arial, sans-serif"),
        height=900,
        autosize=True,
        hovermode="closest",
        map=dict(style="open-street-map"),
        legend_title_text="Volcano Type",
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig

volcano_df = load_volcano_data("data/volcano_ds_pop.csv")
volcano_geojson = load_geojson("data/countries.geojson")  # loaded and cached, not used in this chart yet

fig = make_figure(volcano_df)

st.plotly_chart(fig, use_container_width=True)

with st.expander("Data preview"):
    st.dataframe(volcano_df.head(25), use_container_width=True)
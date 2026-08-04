"""AgriQ interactive dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path("data/processed")


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    st.set_page_config(page_title="AgriQ Dashboard", layout="wide")
    st.title("AgriQ — Smart Crop Planning")
    st.caption("AI + Quantum-Inspired Optimization for Syrian Agriculture")

    col1, col2, col3 = st.columns(3)
    parcels = load_csv("parcels.csv")
    suitability = load_csv("suitability_matrix.csv")
    prices = load_csv("price_forecast.csv")

    col1.metric("Parcels", len(parcels))
    col2.metric("Crops", suitability["crop"].nunique() if not suitability.empty else 0)
    col3.metric("Price forecasts", len(prices))

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Suitability", "Allocations", "Forecasts", "Parcels"]
    )

    with tab1:
        st.subheader("Crop Suitability Matrix")
        if not suitability.empty:
            pivot = suitability.pivot(
                index="parcel_id", columns="crop", values="suitability_score"
            )
            st.dataframe(pivot, use_container_width=True)
        else:
            st.info("Run the pipeline to generate suitability data.")

    with tab2:
        st.subheader("Optimizer Comparison")
        sa = load_csv("allocation_sa.csv")
        sb = load_csv("allocation_sb.csv")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Simulated Annealing**")
            st.dataframe(sa, use_container_width=True)
            if not sa.empty:
                st.metric("Total profit (USD)", f"{sa['expected_profit_usd'].sum():,.0f}")
        with c2:
            st.markdown("**Simulated Bifurcation**")
            st.dataframe(sb, use_container_width=True)
            if not sb.empty:
                st.metric("Total profit (USD)", f"{sb['expected_profit_usd'].sum():,.0f}")

    with tab3:
        st.subheader("Price & Demand Forecasts")
        demand = load_csv("demand_forecast.csv")
        if not prices.empty:
            st.line_chart(prices.set_index("year")["forecast_price_usd"])
        if not demand.empty:
            st.bar_chart(demand.set_index("crop")["demand_tonnes"])

    with tab4:
        st.subheader("Parcel Overview")
        st.dataframe(parcels, use_container_width=True)
        if not parcels.empty and {"lat", "lon"}.issubset(parcels.columns):
            st.map(parcels.rename(columns={"lat": "latitude", "lon": "longitude"}))


if __name__ == "__main__":
    main()

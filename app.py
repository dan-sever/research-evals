"""Streamlit dashboard for browsing eval runs and comparing providers.

    streamlit run app.py

The seven tabs each live in `ui/tabs/`. This file is the entrypoint —
config + DB init + one-shot housekeeping (log prune) + tab dispatch.
"""

from __future__ import annotations

import streamlit as st

from benchmarks import launcher, storage
from ui.tabs import compare as compare_tab
from ui.tabs import dashboard as dashboard_tab
from ui.tabs import export as export_tab
from ui.tabs import finance_search as finance_search_tab
from ui.tabs import insights as insights_tab
from ui.tabs import inspect as inspect_tab
from ui.tabs import launch as launch_tab
from ui.tabs import tier as tier_tab

st.set_page_config(page_title="Research Benchmarks", layout="wide")
st.title("Research benchmark runs")

storage.init_db()

# One-shot log prune at app boot. Keeps last 100 logs, skips anything
# touched in the last hour so in-flight runs are never disturbed.
if "_logs_pruned" not in st.session_state:
    try:
        launcher.prune_logs(keep_n=100, min_age_seconds=3600)
    except Exception:
        pass
    st.session_state["_logs_pruned"] = True


(tab_launch, tab_finance, tab_inspect, tab_compare, tab_tier,
 tab_dashboard, tab_insights, tab_export) = st.tabs(
    ["Launch run", "Finance search", "Single run inspector",
     "Provider comparison", "Tier analysis", "Dashboard", "Insights",
     "Export data"]
)

with tab_launch:
    launch_tab.render()

with tab_finance:
    finance_search_tab.render()

with tab_inspect:
    inspect_tab.render()

with tab_compare:
    compare_tab.render()

with tab_tier:
    tier_tab.render()

with tab_dashboard:
    dashboard_tab.render()

with tab_insights:
    insights_tab.render()

with tab_export:
    export_tab.render()

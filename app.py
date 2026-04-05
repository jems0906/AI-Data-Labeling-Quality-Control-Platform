from __future__ import annotations

import streamlit as st

from src.labeling_qc.auth import ensure_storage
from src.labeling_qc.core import PlatformConfig, build_demo_results
from src.labeling_qc.ui import (
    init_session_state,
    inject_styles,
    login_view,
    render_account_settings,
    render_admin_console,
    render_client_snapshot,
    render_header,
    render_labeler_ops,
    render_matching,
    render_overview,
    render_sidebar,
)

st.set_page_config(
    page_title="AI Labeling Quality Control Platform",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_results(num_tasks: int, num_labelers: int, seed: int):
    return build_demo_results(
        PlatformConfig(num_tasks=num_tasks, num_labelers=num_labelers, seed=seed)
    )


def main() -> None:
    ensure_storage()
    init_session_state()

    if not st.session_state.authenticated:
        login_view()
        return

    inject_styles(st.session_state.get("theme", "Light"))
    num_tasks, num_labelers, seed, page = render_sidebar()
    results = load_results(num_tasks=num_tasks, num_labelers=num_labelers, seed=seed)
    render_header(results["summary"])

    if page == "Overview":
        render_overview(results)
    elif page == "Labeler Ops":
        render_labeler_ops(results)
    elif page == "Smart Matching":
        render_matching(results)
    elif page == "Client Snapshot":
        render_client_snapshot(results)
    elif page == "Admin Console":
        render_admin_console(results)
    elif page == "Account Settings":
        render_account_settings()


main()

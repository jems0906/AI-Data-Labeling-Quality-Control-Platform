"""Streamlit UI views and presentation helpers for the labeling QC platform."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from .auth import (
    DEFAULT_USERS,
    ROLES,
    USER_DB_PATH,
    append_audit_event,
    authenticate_user,
    build_user_record,
    hash_password,
    load_audit_log,
    load_users,
    normalize_username,
    now_iso,
    save_users,
    set_authenticated_user,
    verify_password,
)


def inject_styles(theme: str = "Light") -> None:
    is_dark = theme == "Dark"
    app_background = "linear-gradient(180deg, #0b1220 0%, #111827 100%)" if is_dark else "linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%)"
    main_text = "#e5eefc" if is_dark else "#0f172a"
    sidebar_background = "linear-gradient(180deg, #020617 0%, #0f172a 100%)" if is_dark else "linear-gradient(180deg, #0f172a 0%, #111827 100%)"
    card_background = "#111827" if is_dark else "#ffffff"
    card_border = "#334155" if is_dark else "#dbeafe"
    muted_text = "#cbd5e1" if is_dark else "#475569"
    hero_gradient = "linear-gradient(135deg, #020617 0%, #1d4ed8 100%)" if is_dark else "linear-gradient(135deg, #111827 0%, #2563eb 100%)"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {app_background};
            color: {main_text};
        }}
        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 1.5rem;
        }}
        [data-testid="stSidebar"] {{
            background: {sidebar_background};
        }}
        [data-testid="stSidebar"] * {{
            color: #f8fafc;
        }}
        .hero-card {{
            background: {hero_gradient};
            border-radius: 18px;
            padding: 1.4rem 1.6rem;
            color: white;
            box-shadow: 0 18px 45px rgba(37, 99, 235, 0.18);
            margin-bottom: 1rem;
        }}
        .hero-card h1 {{
            margin: 0;
            font-size: 2rem;
        }}
        .hero-card p {{
            margin: 0.45rem 0 0 0;
            opacity: 0.92;
        }}
        .pill {{
            display: inline-block;
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.22);
            border-radius: 999px;
            padding: 0.22rem 0.65rem;
            font-size: 0.78rem;
            margin-bottom: 0.6rem;
        }}
        .login-card {{
            background: {card_background};
            border: 1px solid {card_border};
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
        }}
        .mini-note {{
            font-size: 0.9rem;
            color: {muted_text};
        }}
        div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"], .stMarkdown, .stCaption {{
            color: {main_text};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    defaults = {
        "authenticated": False,
        "username": None,
        "role": None,
        "display_name": None,
        "theme": "Light",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def logout() -> None:
    actor = st.session_state.get("username") or "unknown"
    append_audit_event(actor, "Signed out", "Success")
    for key in ("authenticated", "username", "role", "display_name"):
        st.session_state[key] = False if key == "authenticated" else None


def login_view() -> None:
    inject_styles(st.session_state.get("theme", "Light"))
    st.markdown(
        """
        <div class="hero-card">
            <div class="pill">Secure Local Authentication Enabled</div>
            <h1>AI Data Labeling Quality Control Platform</h1>
            <p>Sign in to access fraud analytics, labeler ops dashboards, and administrator controls.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns((1.15, 0.95))
    with left:
        st.subheader("What’s new")
        st.markdown(
            """
            - **Persistent local users** stored in `data/users.json`
            - **Salted password hashes** using PBKDF2-SHA256
            - **Admin CRUD tools** for creating, editing, disabling, and deleting users
            """
        )
        st.info("Use a starter account below, or create more from the Admin Console after signing in as `admin`.")
        with st.expander("Starter accounts", expanded=True):
            starter_users = pd.DataFrame(
                [
                    {"username": username, "password": config["password"], "role": config["role"]}
                    for username, config in DEFAULT_USERS.items()
                ]
            )
            st.dataframe(starter_users, width="stretch", hide_index=True)

        quick_cols = st.columns(3)
        if quick_cols[0].button("Use admin demo", width="stretch"):
            user = load_users()["admin"]
            append_audit_event("admin", "Signed in via demo quick access", "Success")
            set_authenticated_user(user)
            st.rerun()
        if quick_cols[1].button("Use ops demo", width="stretch"):
            user = load_users()["ops"]
            append_audit_event("ops", "Signed in via demo quick access", "Success")
            set_authenticated_user(user)
            st.rerun()
        if quick_cols[2].button("Use client demo", width="stretch"):
            user = load_users()["client"]
            append_audit_event("client", "Signed in via demo quick access", "Success")
            set_authenticated_user(user)
            st.rerun()

    with right:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.subheader("🔐 Sign in")
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="admin")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Sign in", width="stretch")

        st.caption("Demo credentials: `admin / admin123`, `ops / ops123`, `client / client123`.")
        st.markdown('</div>', unsafe_allow_html=True)

        if submitted:
            success, error_message, user = authenticate_user(username, password)
            if success and user:
                set_authenticated_user(user)
                st.rerun()
            elif error_message:
                st.error(error_message)


def render_header(summary: dict[str, float | int | bool]) -> None:
    display_name = st.session_state.display_name or st.session_state.username
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="pill">Signed in as {display_name} · {st.session_state.role}</div>
            <h1>Scale-Style Labeler Performance Hub</h1>
            <p>{summary['tasks_processed']:,} tasks processed in {summary['processing_time_seconds']}s with {summary['matching_speedup']}x faster smart matching.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[int, int, int, str]:
    role = st.session_state.role or "Viewer"
    if role == "Administrator":
        pages = ["Overview", "Labeler Ops", "Smart Matching", "Admin Console", "Account Settings"]
    elif role == "Customer Success":
        pages = ["Overview", "Smart Matching", "Client Snapshot", "Account Settings"]
    else:
        pages = ["Overview", "Labeler Ops", "Smart Matching", "Account Settings"]

    with st.sidebar:
        st.markdown("## 🛡️ Control Center")
        st.caption(f"User: `{st.session_state.username}`")
        st.caption(f"Role: **{role}**")
        st.caption(f"Local user store: `{USER_DB_PATH.name}`")
        page = st.radio("Workspace view", pages)
        st.divider()
        st.markdown("### Preferences")
        dark_mode = st.toggle("Dark mode", value=st.session_state.get("theme", "Light") == "Dark")
        st.session_state.theme = "Dark" if dark_mode else "Light"
        st.divider()
        st.markdown("### Demo controls")
        num_tasks = st.slider("Mock tasks", min_value=10_000, max_value=50_000, value=50_000, step=5_000)
        num_labelers = st.slider("Simulated labelers", min_value=250, max_value=1_000, value=1_000, step=50)
        seed = st.number_input("Random seed", min_value=1, max_value=999, value=42, step=1)
        st.divider()
        if st.button("Log out", width="stretch"):
            logout()
            st.rerun()

    return num_tasks, num_labelers, int(seed), page


def render_overview(results: dict[str, Any]) -> None:
    summary = results["summary"]
    metrics = results["labeler_metrics"].copy()
    tasks = results["tasks"].copy()
    growth = results["growth"].copy()

    st.markdown("## 🚀 Executive overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fraudulent labelers detected", f"{summary['fraud_detected_pct']}%")
    col2.metric("Accuracy prediction lift", f"{summary['accuracy_prediction_lift_pct']}%")
    col3.metric("Matching speedup", f"{summary['matching_speedup']}x")
    col4.metric("Throughput", f"{summary['tasks_processed']:,} / {summary['processing_time_seconds']}s")

    st.info(
        f"Precision on flagged contributors: **{summary['fraud_precision_pct']}%** | "
        f"Model AUC improved from **{summary['baseline_auc']}** to **{summary['enhanced_auc']}**."
    )

    left, right = st.columns((1.15, 0.95))
    with left:
        scatter = px.scatter(
            metrics,
            x="quality_score",
            y="fraud_risk",
            color=metrics["fraud_flag"].map({True: "Flagged", False: "Trusted"}),
            hover_data=["labeler_id", "accuracy", "speed_score", "consistency_score", "tasks_completed"],
            title="Contributor quality vs fraud risk",
            labels={"color": "Contributor status"},
            color_discrete_map={"Flagged": "#ef4444", "Trusted": "#10b981"},
        )
        scatter.update_layout(height=400)
        st.plotly_chart(scatter, width="stretch")

    with right:
        top_labelers = results["top_labelers"].head(10).sort_values("quality_score")
        leaderboard = px.bar(
            top_labelers,
            x="quality_score",
            y=top_labelers["labeler_id"].astype(str),
            color="primary_domain",
            orientation="h",
            title="Top labeler leaderboard",
        )
        leaderboard.update_layout(height=400, yaxis_title="Labeler ID")
        st.plotly_chart(leaderboard, width="stretch")

    left, right = st.columns(2)
    with left:
        quality_curve = px.histogram(
            tasks,
            x="predicted_task_quality",
            nbins=30,
            color="task_value",
            title="Predicted task quality distribution",
            barmode="overlay",
        )
        quality_curve.update_layout(height=340)
        st.plotly_chart(quality_curve, width="stretch")

    with right:
        growth["week"] = pd.to_datetime(growth["week"])
        growth_chart = px.line(
            growth,
            x="week",
            y=["cumulative_labelers", "tasks_processed"],
            title="Contributor growth and weekly throughput",
        )
        growth_chart.update_layout(height=340, legend_title_text="Metric")
        st.plotly_chart(growth_chart, width="stretch")


def render_labeler_ops(results: dict[str, Any]) -> None:
    metrics = results["labeler_metrics"].copy()
    suspicious = results["suspicious_labelers"].copy()

    st.markdown("## 🔎 Labeler operations")
    ops_tab, fraud_tab = st.tabs(["Quality signals", "Fraud review queue"])

    with ops_tab:
        left, right = st.columns(2)
        with left:
            quality_by_domain = (
                metrics.groupby("primary_domain", as_index=False)[["quality_score", "accuracy"]]
                .mean()
                .sort_values("quality_score", ascending=False)
            )
            fig = px.bar(
                quality_by_domain,
                x="primary_domain",
                y=["quality_score", "accuracy"],
                barmode="group",
                title="Average quality by specialty domain",
            )
            fig.update_layout(height=340, legend_title_text="Signal")
            st.plotly_chart(fig, width="stretch")

        with right:
            risk_hist = px.histogram(
                metrics,
                x="fraud_risk",
                color=metrics["fraud_flag"].map({True: "Flagged", False: "Trusted"}),
                nbins=25,
                title="Fraud risk distribution",
                color_discrete_map={"Flagged": "#ef4444", "Trusted": "#3b82f6"},
            )
            risk_hist.update_layout(height=340, showlegend=True)
            st.plotly_chart(risk_hist, width="stretch")

    with fraud_tab:
        review_stats = st.columns(3)
        review_stats[0].metric("Flagged accounts", int(metrics["fraud_flag"].sum()))
        review_stats[1].metric("Avg. flagged risk", f"{suspicious['fraud_risk'].mean():.2f}")
        review_stats[2].metric("Avg. identical-answer ratio", f"{suspicious['identical_answer_ratio'].mean():.2f}")
        st.dataframe(suspicious, width="stretch", hide_index=True)


def render_matching(results: dict[str, Any]) -> None:
    summary = results["summary"]
    match_preview = results["matching_sample"].copy()
    open_tasks = results["open_tasks"].copy()

    st.markdown("## 🧠 Smart matching workspace")
    stat_cols = st.columns(3)
    stat_cols[0].metric("Naive matching time", f"{summary['naive_matching_seconds']}s")
    stat_cols[1].metric("Optimized matching time", f"{summary['optimized_matching_seconds']}s")
    stat_cols[2].metric("SLA target met", "Yes" if summary["processing_target_met"] else "No")

    left, right = st.columns((1.05, 0.95))
    with left:
        task_mix = (
            open_tasks.groupby(["domain", "task_value"], as_index=False)
            .size()
            .rename(columns={"size": "open_tasks"})
        )
        fig = px.bar(
            task_mix,
            x="domain",
            y="open_tasks",
            color="task_value",
            title="Open customer tasks by value tier",
            barmode="stack",
        )
        fig.update_layout(height=340)
        st.plotly_chart(fig, width="stretch")

    with right:
        quality_mix = px.scatter(
            match_preview,
            x="priority",
            y="assigned_quality_score",
            color="domain",
            size="task_value_num",
            title="Assignment quality by task priority",
        )
        quality_mix.update_layout(height=340)
        st.plotly_chart(quality_mix, width="stretch")

    match_preview["assigned_quality_score"] = match_preview["assigned_quality_score"].round(3)
    match_preview["assigned_fraud_risk"] = match_preview["assigned_fraud_risk"].round(3)
    st.dataframe(match_preview, width="stretch", hide_index=True)


def render_client_snapshot(results: dict[str, Any]) -> None:
    tasks = results["tasks"].copy()
    customer_summary = (
        tasks.groupby("task_value", as_index=False)["predicted_task_quality"]
        .mean()
        .sort_values("predicted_task_quality", ascending=False)
    )

    st.markdown("## 🤝 Client quality snapshot")
    st.success("This view is optimized for customer-facing reporting and SLA confidence checks.")
    fig = px.bar(
        customer_summary,
        x="task_value",
        y="predicted_task_quality",
        color="task_value",
        title="Predicted quality by customer task tier",
    )
    fig.update_layout(height=360, showlegend=False)
    st.plotly_chart(fig, width="stretch")


def render_account_settings() -> None:
    users = load_users()
    username = st.session_state.username
    current_user = users.get(username)

    if current_user is None:
        st.error("Unable to load the current account.")
        return

    st.markdown("## 👤 Account settings")
    left, right = st.columns((1, 1))

    with left:
        with st.container(border=True):
            st.subheader("Profile")
            st.write(f"**Username:** `{current_user['username']}`")
            st.write(f"**Role:** {current_user['role']}")
            st.write(f"**Theme:** {st.session_state.get('theme', 'Light')}")
            with st.form("profile_update_form"):
                display_name = st.text_input("Display name", value=current_user.get("display_name", ""))
                save_profile = st.form_submit_button("Save profile", width="stretch")
            if save_profile:
                current_user["display_name"] = display_name.strip() or username
                current_user["updated_at"] = now_iso()
                users[username] = current_user
                save_users(users)
                st.session_state.display_name = current_user["display_name"]
                append_audit_event(username, "Updated profile display name", "Success")
                st.success("Profile updated.")
                st.rerun()

    with right:
        with st.container(border=True):
            st.subheader("Change password")
            with st.form("password_change_form", clear_on_submit=True):
                current_password = st.text_input("Current password", type="password")
                new_password = st.text_input("New password", type="password")
                confirm_password = st.text_input("Confirm new password", type="password")
                change_password = st.form_submit_button("Update password", width="stretch")

            if change_password:
                if not verify_password(current_password, current_user["password_hash"], current_user["salt"]):
                    st.error("Current password is incorrect.")
                elif len(new_password) < 8:
                    st.error("New password must be at least 8 characters.")
                elif new_password != confirm_password:
                    st.error("New passwords do not match.")
                else:
                    password_hash, salt = hash_password(new_password)
                    current_user["password_hash"] = password_hash
                    current_user["salt"] = salt
                    current_user["updated_at"] = now_iso()
                    users[username] = current_user
                    save_users(users)
                    append_audit_event(username, "Changed account password", "Success")
                    st.success("Password updated successfully.")


def render_user_management() -> None:
    users = load_users()
    user_rows = []
    for user in users.values():
        user_rows.append(
            {
                "username": user["username"],
                "display_name": user.get("display_name", ""),
                "role": user["role"],
                "active": user.get("active", True),
                "updated_at": user.get("updated_at", ""),
            }
        )

    st.markdown("### User management CRUD")
    st.dataframe(pd.DataFrame(user_rows), width="stretch", hide_index=True)

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("Create user")
            with st.form("create_user_form", clear_on_submit=True):
                username = st.text_input("Username")
                display_name = st.text_input("Display name")
                role = st.selectbox("Role", ROLES)
                password = st.text_input("Temporary password", type="password")
                active = st.checkbox("Active", value=True)
                create_user = st.form_submit_button("Create user", width="stretch")

            if create_user:
                normalized = normalize_username(username)
                if not normalized:
                    st.error("Username is required.")
                elif normalized in users:
                    st.error("That username already exists.")
                elif len(password) < 8:
                    st.error("Password must be at least 8 characters.")
                else:
                    users[normalized] = build_user_record(
                        username=normalized,
                        display_name=display_name or normalized,
                        role=role,
                        password=password,
                        active=active,
                    )
                    save_users(users)
                    append_audit_event(st.session_state.username or "admin", f"Created user '{normalized}'", "Success")
                    st.success(f"User `{normalized}` created.")
                    st.rerun()

    with right:
        with st.container(border=True):
            st.subheader("Edit or delete user")
            selected_username = st.selectbox("Select user", options=sorted(users))
            selected_user = users[selected_username]

            with st.form("update_user_form"):
                display_name = st.text_input("Display name", value=selected_user.get("display_name", ""))
                role = st.selectbox("Role", ROLES, index=ROLES.index(selected_user["role"]))
                active = st.checkbox("Active", value=selected_user.get("active", True))
                new_password = st.text_input("Reset password (optional)", type="password")
                save_changes = st.form_submit_button("Save changes", width="stretch")

            if save_changes:
                if selected_username == st.session_state.username and role != "Administrator":
                    st.error("You cannot remove your own administrator access while signed in.")
                elif selected_username == st.session_state.username and not active:
                    st.error("You cannot disable the account currently in use.")
                elif new_password and len(new_password) < 8:
                    st.error("New password must be at least 8 characters.")
                else:
                    selected_user["display_name"] = display_name.strip() or selected_username
                    selected_user["role"] = role
                    selected_user["active"] = active
                    selected_user["updated_at"] = now_iso()
                    if new_password:
                        password_hash, salt = hash_password(new_password)
                        selected_user["password_hash"] = password_hash
                        selected_user["salt"] = salt
                    users[selected_username] = selected_user
                    save_users(users)
                    append_audit_event(
                        st.session_state.username or "admin",
                        f"Updated user '{selected_username}'",
                        "Success",
                    )
                    st.success(f"User `{selected_username}` updated.")
                    st.rerun()

            delete_disabled = selected_username == st.session_state.username
            if st.button("Delete selected user", width="stretch", disabled=delete_disabled):
                users.pop(selected_username, None)
                save_users(users)
                append_audit_event(
                    st.session_state.username or "admin",
                    f"Deleted user '{selected_username}'",
                    "Success",
                )
                st.success(f"User `{selected_username}` deleted.")
                st.rerun()

            if delete_disabled:
                st.caption("You cannot delete the account currently in use.")


def render_admin_console(results: dict[str, Any]) -> None:
    suspicious = results["suspicious_labelers"].copy()
    metrics = results["labeler_metrics"].copy()
    audit_log = load_audit_log()

    st.markdown("## 🔐 Admin console")
    st.warning("Administrator-only workspace for policy controls, reviewer operations, audit visibility, and user management.")

    left, right = st.columns((1, 1))
    with left:
        with st.container(border=True):
            st.subheader("Policy controls")
            auto_block = st.toggle("Auto-block high-risk contributors", value=True)
            manual_review = st.toggle("Require review for critical tasks", value=True)
            anomaly_alerts = st.toggle("Send anomaly alerts to ops", value=True)
            if st.button("Save policy changes", width="stretch"):
                append_audit_event(
                    st.session_state.username or "admin",
                    f"Updated policy settings (auto_block={auto_block}, review={manual_review}, alerts={anomaly_alerts})",
                    "Success",
                )
                st.success("Policy changes saved.")

    with right:
        with st.container(border=True):
            st.subheader("Platform health")
            st.metric("Trusted contributors", int((~metrics["fraud_flag"]).sum()))
            st.metric("Fraud review queue", int(metrics["fraud_flag"].sum()))
            st.metric("Average quality score", f"{metrics['quality_score'].mean():.2f}")

    admin_tab1, admin_tab2, admin_tab3 = st.tabs(["User Management", "Audit Log", "Risk Queue"])

    with admin_tab1:
        render_user_management()

    with admin_tab2:
        st.markdown("### Recent admin and auth events")
        st.dataframe(audit_log.head(25), width="stretch", hide_index=True)

    with admin_tab3:
        st.markdown("### High-risk accounts pending action")
        st.dataframe(suspicious.head(10), width="stretch", hide_index=True)

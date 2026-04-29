import ast
import html
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from questionnaire import render_questionnaire


DEV_MODE = False


st.set_page_config(
    page_title="Nature Intelligence Platform",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if not DEV_MODE:
    st.markdown("""
    <style>
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"] {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4f4f4f;
        margin-bottom: 1.4rem;
    }
    .card {
        padding: 1rem 1rem 0.85rem 1rem;
        border-radius: 14px;
        border: 1px solid #e7e9ee;
        background-color: #ffffff;
        box-shadow: 0 4px 14px rgba(15,23,42,0.05);
        margin-bottom: 1rem;
    }
    .dashboard-card {
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid #e7e9ee;
        background: #ffffff;
        box-shadow: 0 4px 14px rgba(15,23,42,0.05);
        margin-bottom: 1rem;
    }
    .dashboard-title {
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.65rem;
    }
    .input-row {
        border-bottom: 1px solid #f0f2f5;
        padding: 0.48rem 0;
    }
    .input-label {
        color: #64748b;
        font-size: 0.78rem;
        margin-bottom: 0.08rem;
    }
    .input-value {
        color: #111827;
        font-size: 0.92rem;
        font-weight: 600;
    }
    .badge {
        display: inline-block;
        padding: 0.22rem 0.55rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        margin: 0.15rem 0.2rem 0.15rem 0;
        border: 1px solid transparent;
    }
    .badge-green {
        background: #ecfdf3;
        color: #087443;
        border-color: #bbf7d0;
    }
    .badge-blue {
        background: #eff6ff;
        color: #1d4ed8;
        border-color: #bfdbfe;
    }
    .badge-orange {
        background: #fff7ed;
        color: #c2410c;
        border-color: #fed7aa;
    }
    .metric-box {
        padding: 0.85rem;
        border-radius: 12px;
        background: #f8fafc;
        border: 1px solid #edf2f7;
        min-height: 92px;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin-top: 0.8rem;
    }
    .metric-label {
        color: #64748b;
        font-size: 0.78rem;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        color: #0f172a;
        font-size: 1.18rem;
        font-weight: 800;
    }
    .section-note {
        color: #64748b;
        font-size: 0.86rem;
    }
    .score-row {
        margin-bottom: 0.42rem;
    }
    .score-head {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 0.75rem;
        margin-bottom: 0.08rem;
    }
    .score-name {
        font-weight: 800;
        color: #1f2937;
        font-size: 0.88rem;
    }
    .score-points {
        color: #334155;
        font-weight: 700;
        font-size: 0.86rem;
        white-space: nowrap;
    }
    .score-explain {
        color: #6b7280;
        font-size: 0.76rem;
        line-height: 1.18;
        margin-bottom: 0.16rem;
    }
    .score-track {
        height: 5px;
        background: #e8eef7;
        border-radius: 999px;
        overflow: hidden;
    }
    .score-fill {
        height: 100%;
        background: #2f80ed;
        border-radius: 999px;
    }
    .score-divider {
        border-top: 1px solid #e7e9ee;
        margin: 0.75rem 0 0.65rem 0;
    }
    .score-total {
        color: #0f172a;
        font-size: 1.45rem;
        font-weight: 800;
    }
    .option-card {
        display: grid;
        grid-template-columns: 56px 1fr auto;
        gap: 0.9rem;
        align-items: center;
        padding: 0.9rem 1rem;
        border: 1px solid #e1e7ef;
        border-radius: 14px;
        box-shadow: 0 3px 10px rgba(15,23,42,0.04);
        background: #ffffff;
        margin-bottom: 0.75rem;
    }
    .option-icon {
        width: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        background: #eff6ff;
        color: #1d4ed8;
        font-size: 1.45rem;
        font-weight: 800;
    }
    .option-title {
        color: #0f172a;
        font-weight: 800;
        font-size: 1.02rem;
        margin-bottom: 0.1rem;
    }
    .option-subtitle {
        color: #64748b;
        font-size: 0.9rem;
    }
    .option-score {
        color: #0f172a;
        font-size: 1.4rem;
        font-weight: 800;
        white-space: nowrap;
    }
    .small-note {
        font-size: 0.9rem;
        color: #6b6b6b;
    }
    div.stButton > button {
        min-height: 48px;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 14px;
    }
</style>
""", unsafe_allow_html=True)


def load_app_2_functions():
    """
    Reuse app_2 helper and recommendation functions without executing its UI.
    app_2.py is intentionally left untouched as the current working prototype.
    """
    source_path = Path("app_2.py")
    source = source_path.read_text()
    module = ast.parse(source, filename=str(source_path))
    function_nodes = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    reusable_module = ast.Module(body=function_nodes, type_ignores=[])
    ast.fix_missing_locations(reusable_module)

    namespace = {
        "st": st,
        "pd": pd,
        "np": np,
        "Path": Path,
    }
    exec(compile(reusable_module, str(source_path), "exec"), namespace)
    return namespace


APP_2 = load_app_2_functions()

build_recommendation = APP_2["build_recommendation"]
ensure_columns = APP_2["ensure_columns"]
find_first_existing = APP_2["find_first_existing"]
format_currency = APP_2["format_currency"]
get_financial_view = APP_2["get_financial_view"]


PROJECT_COLUMNS = [
    "project_id",
    "project_name",
    "country",
    "continent",
    "status",
    "project_summary",
    "intervention_rationale",
    "budget_usd",
    "budget_band",
    "start_date",
    "end_date",
    "duration_years",
    "time_band",
    "local_problems_perceived",
    "nature_based_solutions_implemented",
    "main_pressure_tags",
    "material_topic_tags",
    "intervention_type_tags",
    "risk_concern_tags",
    "business_objective_tags",
    "sbtn_outcome_tags",
    "biodiversity_benefit",
    "water_benefit",
    "climate_adaptation_benefit",
    "climate_mitigation_benefit",
    "food_security_benefit",
    "social_benefit",
    "stakeholders_involved",
    "match_explanation",
]


@st.cache_data
def load_data_for_app_3():
    base = Path(".")
    company_path = base / "company_exposure.csv"
    project_path = base / "nbs_projects_matching_ready.csv"
    finance_path = base / "project_financial_output.csv"

    if not company_path.exists():
        return None, None, None, "Missing company_exposure.csv."
    if not project_path.exists():
        return None, None, None, "Missing nbs_projects_matching_ready.csv."

    df_company = pd.read_csv(company_path)
    df_projects = pd.read_csv(project_path)
    df_finance = pd.read_csv(finance_path) if finance_path.exists() else pd.DataFrame()
    data_note = "Using nbs_projects_matching_ready.csv as the main project supply database."

    df_company = ensure_columns(df_company, [
        "company_name",
        "hotspots",
        "risk_exposure",
        "sbtn_targets",
        "business_priorities",
        "company_summary",
    ])

    df_projects = ensure_columns(df_projects, PROJECT_COLUMNS)

    if not df_finance.empty:
        df_finance = ensure_columns(df_finance, [
            "project_id",
            "project_name",
            "country",
            "intervention_type",
            "estimated_cost_usd",
        ])

    return df_company, df_projects, df_finance, data_note


def build_projects_from_finance(df_finance):
    finance = df_finance.copy()
    project_id_col = find_first_existing(finance, ["project_id", "id"])
    project_name_col = find_first_existing(finance, ["project_name", "name"])
    country_col = find_first_existing(finance, ["country", "region"])
    intervention_col = find_first_existing(finance, ["intervention_type", "intervention"])
    status_col = find_first_existing(finance, ["status"])
    stage_col = find_first_existing(finance, ["stage"])

    rows = []
    for _, row in finance.drop_duplicates(subset=[project_id_col] if project_id_col else None).iterrows():
        project_name = row.get(project_name_col, "Unknown project") if project_name_col else "Unknown project"
        region = row.get(country_col, "N/A") if country_col else "N/A"
        intervention = row.get(intervention_col, "N/A") if intervention_col else "N/A"
        status = row.get(status_col, "") if status_col else ""
        stage = row.get(stage_col, "") if stage_col else ""

        tag_text = ";".join(
            str(value).replace("_", " ")
            for value in [region, intervention, status, stage]
            if pd.notna(value) and str(value).strip()
        )

        rows.append({
            "project_id": row.get(project_id_col, "") if project_id_col else "",
            "project_name": project_name,
            "region": region,
            "ecosystem": "N/A",
            "intervention_type": intervention,
            "project_summary": (
                f"{project_name} is available in the finance dataset and is shown here "
                "as a fallback project record for the questionnaire prototype."
            ),
            "hotspot_tags": tag_text,
            "risk_tags": tag_text,
            "sbtn_tags": tag_text,
            "business_tags": tag_text,
        })

    return pd.DataFrame(rows)


def split_tags(value):
    if pd.isna(value) or str(value).strip() == "":
        return set()
    return {
        part.strip()
        for part in str(value).replace("|", ";").split(";")
        if part.strip()
    }


def first_tag(value, default="N/A"):
    tags = sorted(split_tags(value))
    return tags[0] if tags else default


def meaningful_answer(value):
    if value is None:
        return ""
    value = str(value).strip()
    if value.lower() in {"", "unknown", "not sure yet", "not yet defined", "not sure / prefer not to say"}:
        return ""
    return value


def single_value_fit(answer, tags, neutral=0.5):
    answer = meaningful_answer(answer)
    if not answer:
        return neutral, set()
    tag_set = split_tags(tags)
    if not tag_set:
        return neutral, set()
    overlap = {answer} if answer in tag_set else set()
    return (1.0 if overlap else 0.0), overlap


def multi_value_fit(answers, tags, neutral=0.5):
    selected = {meaningful_answer(answer) for answer in (answers or [])}
    selected.discard("")
    if not selected:
        return neutral, set()
    tag_set = split_tags(tags)
    if not tag_set:
        return neutral, set()
    overlap = selected.intersection(tag_set)
    return len(overlap) / len(selected), overlap


def geography_fit(geography, detail, row, neutral=0.5):
    location_input = " ".join(
        part.lower()
        for part in [geography, detail]
        if part and str(part).strip()
    )
    if not location_input:
        return neutral, set()

    location_values = [
        row.get("country", ""),
        row.get("continent", ""),
    ]
    matches = {
        str(value).strip()
        for value in location_values
        if value and str(value).strip() and str(value).strip().lower() in location_input
    }
    return (1.0 if matches else 0.0), matches


def budget_fit(answer, budget_band, neutral=0.5):
    answer = meaningful_answer(answer)
    budget_band = meaningful_answer(budget_band)
    if not answer or not budget_band:
        return neutral
    return 1.0 if answer == budget_band else 0.0


def time_fit(answer, time_band, neutral=0.5):
    answer = meaningful_answer(answer)
    time_band = meaningful_answer(time_band)
    if not answer or not time_band:
        return neutral
    return 1.0 if answer == time_band else 0.0


def build_project_finance_info(rec, df_finance=None):
    finance_info = get_financial_view(rec, df_finance)
    if finance_info is not None:
        return finance_info

    budget = pd.to_numeric(rec.get("budget_usd"), errors="coerce")
    if pd.isna(budget):
        return None

    company_contribution = float(budget) * 0.25
    effort = "Low"
    if company_contribution >= 3_000_000:
        effort = "High"
    elif company_contribution >= 1_000_000:
        effort = "Medium"

    horizon = rec.get("duration_years", "N/A")
    return {
        "total_cost": format_currency(budget),
        "company_contribution": format_currency(company_contribution),
        "time_horizon": horizon if str(horizon).strip() != "" else "N/A",
        "effort": effort,
    }


def build_questionnaire_recommendation(company_name, company_profile, df_projects, df_finance=None):
    if df_projects is None or df_projects.empty:
        return None, "No matching-ready NbS projects found."

    scored_rows = []

    for _, row in df_projects.iterrows():
        geo_score, geography_overlap = geography_fit(
            company_profile.get("geography"),
            company_profile.get("geography_detail"),
            row,
        )
        materiality_score, materiality_overlap = multi_value_fit(
            company_profile.get("material_topics"),
            row.get("material_topic_tags"),
        )
        pressure_score, pressure_overlap = single_value_fit(
            company_profile.get("main_pressure"),
            row.get("main_pressure_tags"),
        )
        business_score, business_overlap = single_value_fit(
            company_profile.get("business_objective"),
            row.get("business_objective_tags"),
        )
        risk_score, risk_overlap = single_value_fit(
            company_profile.get("key_risk_concern"),
            row.get("risk_concern_tags"),
        )
        budget_score = budget_fit(
            company_profile.get("budget_level"),
            row.get("budget_band"),
        )
        time_score = time_fit(
            company_profile.get("preferred_time_horizon"),
            row.get("time_band"),
        )

        weighted_score = (
            geo_score * 0.20
            + materiality_score * 0.20
            + pressure_score * 0.20
            + business_score * 0.15
            + risk_score * 0.10
            + budget_score * 0.10
            + time_score * 0.05
        )

        scored_rows.append({
            **row.to_dict(),
            "region": row.get("country", "N/A"),
            "ecosystem": row.get("sbtn_outcome_tags", "N/A"),
            "intervention_type": first_tag(row.get("intervention_type_tags"), "Nature-based solutions"),
            "geography_fit": geo_score,
            "materiality_fit": materiality_score,
            "pressure_fit": pressure_score,
            "business_fit": business_score,
            "risk_fit": risk_score,
            "budget_fit": budget_score,
            "time_fit": time_score,
            "raw_score": weighted_score,
            "fit_score": int(round(weighted_score * 100)),
            "hotspot_overlap": pressure_overlap.union(geography_overlap),
            "risk_overlap": risk_overlap,
            "sbtn_overlap": materiality_overlap,
            "business_overlap": business_overlap,
            "geography_overlap": geography_overlap,
            "hotspot_count": len(pressure_overlap.union(geography_overlap)),
            "risk_count": len(risk_overlap),
            "sbtn_count": len(materiality_overlap),
            "business_count": len(business_overlap),
        })

    result_df = pd.DataFrame(scored_rows).sort_values(
        by=["fit_score", "materiality_fit", "pressure_fit", "business_fit", "project_name"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    top = result_df.iloc[0].copy()
    why_text = top.get("match_explanation")
    if pd.isna(why_text) or str(why_text).strip() == "":
        why_text = (
            f"This project is recommended because it has the strongest combined questionnaire fit "
            f"across geography, materiality, pressure, business objective, risk, budget, and time horizon."
        )

    rec = {
        "company_name": company_name,
        "project_id": top.get("project_id", ""),
        "project_name": top.get("project_name", "Unknown project"),
        "region": top.get("country", "N/A"),
        "ecosystem": top.get("sbtn_outcome_tags", "N/A"),
        "intervention_type": top.get("intervention_type", "Nature-based solutions"),
        "fit_score": int(top.get("fit_score", 0)),
        "why_text": why_text,
        "project_summary": top.get("project_summary", ""),
        "budget_usd": top.get("budget_usd", ""),
        "duration_years": top.get("duration_years", "N/A"),
        "hotspot_overlap": top.get("hotspot_overlap", set()),
        "risk_overlap": top.get("risk_overlap", set()),
        "sbtn_overlap": top.get("sbtn_overlap", set()),
        "business_overlap": top.get("business_overlap", set()),
        "geography_overlap": top.get("geography_overlap", set()),
        "full_ranking": result_df,
    }
    rec["finance_info"] = build_project_finance_info(rec, df_finance)

    return rec, None


def clean_display(value, default="N/A"):
    if value is None or pd.isna(value) or str(value).strip() == "":
        return default
    return str(value)


def format_money_compact(value):
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value):
        return "To be assessed"
    value = float(value)
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}m"
    if value >= 1_000:
        return f"${value / 1_000:.0f}k"
    return f"${value:,.0f}"


def derive_investment_metrics(top_row):
    budget = pd.to_numeric(top_row.get("budget_usd"), errors="coerce")
    budget_band = clean_display(top_row.get("budget_band"), "Unknown")
    duration = pd.to_numeric(top_row.get("duration_years"), errors="coerce")
    time_band = clean_display(top_row.get("time_band"), "")

    if pd.isna(budget):
        estimated_cost = "To be assessed"
        contribution = "To be assessed"
    else:
        estimated_cost = format_money_compact(budget)
        contribution = format_money_compact(float(budget) * 0.25)

    if not pd.isna(duration):
        time_horizon = f"{duration:.1f} years"
    elif time_band and time_band != "N/A":
        time_horizon = time_band
    else:
        time_horizon = "To be assessed"

    effort_map = {
        "Low: < $1.5M": "Low",
        "Medium: $1.5M – $3M": "Medium",
        "High: > $3M": "High",
        "Unknown": "To be assessed",
    }

    return {
        "estimated_cost": estimated_cost,
        "company_contribution": contribution,
        "time_horizon": time_horizon,
        "financial_effort": effort_map.get(budget_band, "To be assessed"),
    }


def get_top_ranked_row(rec):
    ranking = rec.get("full_ranking")
    if ranking is None or ranking.empty:
        return pd.Series(dtype=object)
    return ranking.iloc[0]


def render_score_line(label, score, max_points):
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.5
    score = max(0.0, min(score, 1.0))
    achieved = score * max_points

    left, right = st.columns([2.5, 1])
    with left:
        st.write(f"**{label}**")
        st.progress(score)
    with right:
        st.write(f"{achieved:.1f} / {max_points}")


def safe_html(value, default="To be assessed"):
    return html.escape(clean_display(value, default))


def html_fragment(markup):
    return "\n".join(line.strip() for line in textwrap.dedent(markup).strip().splitlines())


def render_input_row(label, value):
    st.markdown(
        html_fragment(f"""
        <div class="input-row">
            <div class="input-label">{html.escape(label)}</div>
            <div class="input-value">{safe_html(value)}</div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def render_metric_box(label, value):
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">{html.escape(label)}</div>
            <div class="metric-value">{safe_html(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_box_html(label, value):
    return html_fragment(f"""
    <div class="metric-box">
        <div class="metric-label">{html.escape(label)}</div>
        <div class="metric-value">{safe_html(value)}</div>
    </div>
    """)


def build_match_badges(top_row):
    badge_specs = []
    if pd.to_numeric(top_row.get("materiality_fit"), errors="coerce") >= 0.75:
        badge_specs.append(("Strong materiality fit", "green"))
    if pd.to_numeric(top_row.get("pressure_fit"), errors="coerce") >= 0.75:
        badge_specs.append(("Strong pressure fit", "green"))
    if pd.to_numeric(top_row.get("budget_fit"), errors="coerce") >= 0.75:
        badge_specs.append(("Budget fit", "blue"))
    if pd.to_numeric(top_row.get("geography_fit"), errors="coerce") < 0.5:
        badge_specs.append(("Outside selected geography", "orange"))
    if pd.to_numeric(top_row.get("time_fit"), errors="coerce") >= 0.75:
        badge_specs.append(("Time horizon fit", "blue"))
    return badge_specs


def render_badges(badge_specs):
    if not badge_specs:
        st.markdown('<span class="badge badge-blue">Needs further review</span>', unsafe_allow_html=True)
        return

    html_badges = " ".join(
        f'<span class="badge badge-{color}">{html.escape(label)}</span>'
        for label, color in badge_specs
    )
    st.markdown(html_badges, unsafe_allow_html=True)


def badges_html(badge_specs):
    if not badge_specs:
        return '<span class="badge badge-blue">Needs further review</span>'
    return " ".join(
        f'<span class="badge badge-{color}">{html.escape(label)}</span>'
        for label, color in badge_specs
    )


def score_row_html(label, explanation, score, max_points):
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.5
    score = max(0.0, min(score, 1.0))
    achieved = score * max_points
    width = int(round(score * 100))
    return html_fragment(f"""
    <div class="score-row">
        <div class="score-head">
            <div class="score-name">{html.escape(label)}</div>
            <div class="score-points">{achieved:.0f} / {max_points}</div>
        </div>
        <div class="score-explain">{html.escape(explanation)}</div>
        <div class="score-track"><div class="score-fill" style="width: {width}%;"></div></div>
    </div>
    """)


def render_score_detail(label, explanation, score, max_points):
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.5
    score = max(0.0, min(score, 1.0))
    achieved = score * max_points

    left, right = st.columns([3, 1])
    with left:
        st.write(f"**{label}**")
        st.caption(explanation)
        st.progress(score)
    with right:
        st.write(f"{achieved:.0f} / {max_points}")


def render_indicative_investment(top_row):
    investment = derive_investment_metrics(top_row)

    st.markdown("## Indicative investment")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Estimated project cost", investment["estimated_cost"])
    with col2:
        st.metric("Suggested company contribution", investment["company_contribution"])
    with col3:
        st.metric("Time horizon", investment["time_horizon"])
    with col4:
        st.metric("Financial effort", investment["financial_effort"])

    st.caption(
        "Suggested company contribution, time horizon, and financial effort are indicative values derived from estimated project cost and MVP assumptions."
    )
    st.markdown('</div>', unsafe_allow_html=True)


def tag_match_text(selected, tags, selected_label, project_label):
    selected_value = clean_display(selected, "to be assessed")
    project_tags = split_tags(tags)
    if selected_value != "to be assessed" and selected_value in project_tags:
        return f"Selected {selected_label}: {selected_value}. The recommended project is tagged for the same {project_label}, making the fit direct."
    if project_tags:
        return f"Selected {selected_label}: {selected_value}. The project is tagged for {', '.join(sorted(project_tags)[:3])}, so this connection should be reviewed in context."
    return f"Selected {selected_label}: {selected_value}. Project evidence for this dimension is to be assessed."


def materiality_text(company_profile, top_row):
    selected_topics = [
        topic for topic in company_profile.get("material_topics", [])
        if meaningful_answer(topic)
    ]
    project_topics = split_tags(top_row.get("material_topic_tags"))
    outcomes = split_tags(top_row.get("sbtn_outcome_tags"))
    overlap = set(selected_topics).intersection(project_topics)

    if overlap:
        return (
            f"The project aligns with selected material topic(s): {', '.join(sorted(overlap))}. "
            f"Its outcome tags point to {', '.join(sorted(outcomes)[:3]) if outcomes else 'nature outcomes to be assessed'}."
        )
    if selected_topics and project_topics:
        return (
            f"The selected topics are {', '.join(selected_topics)}. The project is tagged for "
            f"{', '.join(sorted(project_topics)[:3])}, so materiality alignment should be assessed further."
        )
    return "Material topic alignment is to be assessed because either questionnaire topics or project outcome tags are incomplete."


def business_relevance_text(company_profile, top_row):
    objective = clean_display(company_profile.get("business_objective"), "to be assessed")
    risk = clean_display(company_profile.get("key_risk_concern"), "to be assessed")
    objective_tags = split_tags(top_row.get("business_objective_tags"))
    risk_tags = split_tags(top_row.get("risk_concern_tags"))

    objective_match = objective in objective_tags
    risk_match = risk in risk_tags
    if objective_match and risk_match:
        return f"The project supports the stated objective and risk concern: {objective} and {risk}."
    if objective_match:
        return f"The project supports the stated objective: {objective}. The risk connection is to be assessed against {risk}."
    if risk_match:
        return f"The project responds to the selected risk concern: {risk}. The business objective connection is to be assessed against {objective}."
    return f"The company selected {objective} and {risk}; project evidence for these priorities is to be assessed."


def investment_feasibility_text(company_profile, top_row):
    selected_budget = clean_display(company_profile.get("budget_level"), "to be assessed")
    selected_horizon = clean_display(company_profile.get("preferred_time_horizon"), "to be assessed")
    project_budget = clean_display(top_row.get("budget_band"), "to be assessed")
    project_time = clean_display(top_row.get("time_band"), "to be assessed")
    investment = derive_investment_metrics(top_row)

    budget_phrase = "fits" if selected_budget == project_budget else "should be compared with"
    time_phrase = "fits" if selected_horizon == project_time else "should be compared with"
    return (
        f"The project budget {budget_phrase} the selected budget range ({project_budget}); estimated cost is "
        f"{investment['estimated_cost']}. Its implementation horizon {time_phrase} the preferred horizon ({project_time})."
    )


def render_explanation_card(title, body, evidence):
    st.markdown(
        html_fragment(f"""
        <div class="card">
            <div class="dashboard-title">{html.escape(title)}</div>
            <p>{safe_html(body)}</p>
            <div class="section-note">Evidence used: {safe_html(evidence)}.</div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def render_recommendation_fit_explanation(company_profile, rec, top_row):
    st.markdown("## Why this recommendation fits")

    pressure_body = tag_match_text(
        company_profile.get("main_pressure"),
        top_row.get("main_pressure_tags"),
        "pressure",
        "pressure",
    )
    match_explanation = clean_display(top_row.get("match_explanation"), "")
    if match_explanation:
        pressure_body = f"{pressure_body} {match_explanation}"

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        render_explanation_card(
            "Materiality alignment",
            materiality_text(company_profile, top_row),
            "selected material topics, project material tags, and outcome tags",
        )
    with row1_col2:
        render_explanation_card(
            "Pressure match",
            pressure_body,
            "selected pressure, project pressure tags, and project summary",
        )

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        render_explanation_card(
            "Business relevance",
            business_relevance_text(company_profile, top_row),
            "selected business objective, selected risk concern, and project tags",
        )
    with row2_col2:
        render_explanation_card(
            "Investment feasibility",
            investment_feasibility_text(company_profile, top_row),
            "selected budget, preferred horizon, project budget band, and project duration",
        )

    st.caption(
        "This explanation is generated from the questionnaire answers, project tags, and the rule-based score breakdown."
    )


def render_company_inputs_dashboard(company_profile):
    topics = company_profile.get("material_topics") or []
    rows = [
        ("Industry", company_profile.get("industry")),
        ("Asset geography", company_profile.get("geography")),
        ("Material topics", ", ".join(topics) if topics else "To be assessed"),
        ("Main pressure", company_profile.get("main_pressure")),
        ("Risk concern", company_profile.get("key_risk_concern")),
        ("Budget", company_profile.get("budget_level")),
        ("Horizon", company_profile.get("preferred_time_horizon")),
    ]
    rows_html = "".join(
        html_fragment(f"""
        <div class="input-row">
            <div class="input-label">{html.escape(label)}</div>
            <div class="input-value">{safe_html(value)}</div>
        </div>
        """)
        for label, value in rows
    )
    st.markdown(
        html_fragment(f"""
        <div class="dashboard-card">
            <div class="dashboard-title">Company inputs</div>
            {rows_html}
        </div>
        """),
        unsafe_allow_html=True,
    )


def render_best_match_dashboard(rec, top_row):
    summary = clean_display(rec.get("project_summary"), "")
    summary_html = ""
    if summary:
        summary_html = html_fragment(f"""
        <div class="section-note" style="margin-top: 0.85rem;">Project context</div>
        <p>{safe_html(summary)}</p>
        """)

    st.markdown(
        html_fragment(f"""
        <div class="dashboard-card">
            <span class="badge badge-green">Best match</span>
            <h2 style="margin: 0.8rem 0 0.4rem 0;">{safe_html(rec.get('project_name'), 'Recommended project to be assessed')}</h2>
            <div class="metric-label">Fit score</div>
            <div style="font-size: 2rem; font-weight: 800; color: #0f172a; margin-bottom: 0.7rem;">{int(rec.get('fit_score', 0))} / 100</div>
            <div>{badges_html(build_match_badges(top_row))}</div>
            <p>{safe_html(rec.get('why_text'), 'Project fit is to be assessed.')}</p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-top: 0.7rem;">
                <div>
                    <p><strong>Country / region:</strong> {safe_html(rec.get('region'))}</p>
                    <p><strong>Intervention type:</strong> {safe_html(rec.get('intervention_type'))}</p>
                </div>
                <div>
                    <p><strong>Ecosystem or material topic tags:</strong> {safe_html(top_row.get('material_topic_tags'))}</p>
                </div>
            </div>
            {summary_html}
        </div>
        """),
        unsafe_allow_html=True,
    )


def render_investment_snapshot_dashboard(top_row):
    investment = derive_investment_metrics(top_row)
    st.markdown(
        html_fragment(f"""
        <div class="dashboard-card">
            <div class="dashboard-title">Investment snapshot</div>
            <div class="metric-grid">
                {metric_box_html("Estimated project cost", investment["estimated_cost"])}
                {metric_box_html("Suggested contribution", investment["company_contribution"])}
                {metric_box_html("Time horizon", investment["time_horizon"])}
                {metric_box_html("Financial effort", investment["financial_effort"])}
            </div>
            <div class="section-note" style="margin-top: 0.65rem;">
                Suggested company contribution, time horizon, and financial effort are indicative values derived from estimated project cost and MVP assumptions.
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def render_score_breakdown_dashboard(rec, top_row):
    rows = [
        score_row_html("Geography fit", "Location alignment with asset or sourcing geography.", top_row.get("geography_fit", 0.5), 20),
        score_row_html("Materiality fit", "Project outcomes aligned with material nature topics.", top_row.get("materiality_fit", 0.5), 20),
        score_row_html("Pressure fit", "Project response to the selected nature pressure.", top_row.get("pressure_fit", 0.5), 20),
        score_row_html("Business fit", "Support for the stated business objective.", top_row.get("business_fit", 0.5), 15),
        score_row_html("Risk fit", "Relevance to the selected risk concern.", top_row.get("risk_fit", 0.5), 10),
        score_row_html("Budget fit", "Fit with the selected investment range.", top_row.get("budget_fit", 0.5), 10),
        score_row_html("Time fit", "Fit with the preferred implementation horizon.", top_row.get("time_fit", 0.5), 5),
    ]
    st.markdown(
        html_fragment(f"""
        <div class="dashboard-card">
            <div class="dashboard-title">How the score is built</div>
            {''.join(rows)}
            <div class="score-divider"></div>
            <div class="score-total">Total: {int(rec.get('fit_score', 0))} / 100</div>
            <div class="section-note">Transparent rule-based scoring model. Weights can be adjusted in future versions.</div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def render_other_options_table(rec):
    other_options = rec["full_ranking"].iloc[1:3]
    if other_options.empty:
        return

    st.markdown("## Other relevant options")
    for _, row in other_options.iterrows():
        subtitle_bits = [
            clean_display(row.get("country"), "To be assessed"),
            clean_display(row.get("material_topic_tags"), "To be assessed"),
            clean_display(row.get("intervention_type"), "To be assessed"),
        ]
        subtitle = " | ".join(bit for bit in subtitle_bits if bit != "To be assessed")
        icon = "≈" if "Water" in clean_display(row.get("material_topic_tags"), "") else "◌"
        st.markdown(
            html_fragment(f"""
            <div class="option-card">
                <div class="option-icon">{html.escape(icon)}</div>
                <div>
                    <div class="option-title">{safe_html(row.get('project_name'))}</div>
                    <div class="option-subtitle">{safe_html(subtitle, 'Details to be assessed')}</div>
                </div>
                <div class="option-score">{int(row.get('fit_score', 0))} / 100</div>
            </div>
            """),
            unsafe_allow_html=True,
        )


def render_next_steps():
    st.markdown("## Recommended next steps")
    steps = [
        ("Review project evidence", "Explore supporting documents, outcomes, and data sources."),
        ("Validate geography relevance", "Assess the implications if the project is outside the selected geography."),
        ("Request due diligence", "Engage with the project team for deeper evaluation."),
    ]
    cols = st.columns(3)
    for col, (title, body) in zip(cols, steps):
        with col:
            st.markdown(
                html_fragment(f"""
                <div class="dashboard-card">
                    <div class="dashboard-title">{html.escape(title)}</div>
                    <p>{safe_html(body)}</p>
                </div>
                """),
                unsafe_allow_html=True,
            )


def render_profile_summary(company_profile):
    st.markdown("## Company context summary")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write(f"**Industry:** {company_profile.get('industry', 'N/A')}")
        st.write(f"**Asset geography:** {company_profile.get('geography') or 'N/A'}")
    with col2:
        topics = company_profile.get("material_topics") or []
        st.write(f"**Material nature topics:** {', '.join(topics) if topics else 'N/A'}")
        st.write(f"**Main pressure:** {company_profile.get('main_pressure', 'N/A')}")
    with col3:
        st.write(f"**Key nature-related risk:** {company_profile.get('key_risk_concern', 'N/A')}")
        st.write(f"**Business objective:** {company_profile.get('business_objective', 'N/A')}")
    with col4:
        st.write(f"**Budget level:** {company_profile.get('budget_level', 'N/A')}")
        st.write(f"**Preferred horizon:** {company_profile.get('preferred_time_horizon', 'N/A')}")

    st.markdown('</div>', unsafe_allow_html=True)
    st.write(
        "The recommendation below is generated by combining your questionnaire inputs with the prototype matching logic."
    )


def render_recommendation_outputs(rec, company_profile, show_top3, show_debug):
    top_row = get_top_ranked_row(rec)

    left, center, right = st.columns([1.1, 2.6, 1.4])
    with left:
        render_company_inputs_dashboard(company_profile)
        if st.button("Edit questionnaire", use_container_width=True, key="edit_questionnaire_company_inputs"):
            reset_to_questionnaire()
            st.rerun()
    with center:
        render_best_match_dashboard(rec, top_row)
        render_investment_snapshot_dashboard(top_row)
    with right:
        render_score_breakdown_dashboard(rec, top_row)

    render_recommendation_fit_explanation(company_profile, rec, top_row)
    render_other_options_table(rec)
    render_next_steps()

    if show_debug:
        st.markdown("## Scoring logic table")
        st.caption(
            "Geography fit: 20% — matches project location with company asset / sourcing geography. "
            "Materiality fit: 20% — matches project outcomes with material nature-related topics. "
            "Pressure fit: 20% — matches project intervention with the company’s selected pressure. "
            "Business fit: 15% — supports the company’s stated business objective. "
            "Risk fit: 10% — responds to the company’s key nature-related risk concern. "
            "Budget fit: 10% — fits the selected investment range. "
            "Time fit: 5% — fits the preferred implementation horizon."
        )
        debug_df = rec["full_ranking"][[
            "project_name",
            "country",
            "geography_fit",
            "materiality_fit",
            "pressure_fit",
            "business_fit",
            "risk_fit",
            "budget_fit",
            "time_fit",
            "raw_score",
            "fit_score",
        ]]
        debug_df = debug_df.rename(columns={
            "project_name": "Project",
            "country": "Country",
            "geography_fit": "Geography fit",
            "materiality_fit": "Materiality fit",
            "pressure_fit": "Pressure fit",
            "business_fit": "Business fit",
            "risk_fit": "Risk fit",
            "budget_fit": "Budget fit",
            "time_fit": "Time fit",
            "raw_score": "Weighted score",
            "fit_score": "Fit score",
        })
        st.dataframe(debug_df, use_container_width=True)


def reset_to_questionnaire():
    st.session_state["app_3_page"] = "questionnaire"
    st.session_state.pop("recommendation_app_3", None)


df_company, df_projects, df_finance, data_note = load_data_for_app_3()

if df_company is None or df_projects is None:
    st.error(
        "The demo data is not available right now. Please check the local setup before running the recommendation."
    )
    st.stop()

company_col = find_first_existing(df_company, ["company_name", "company", "name"])
company_list = sorted(df_company[company_col].dropna().astype(str).unique().tolist())
default_company = "Adidas" if "Adidas" in company_list else company_list[0]
selected_company = default_company
show_top3 = False
show_debug = False

if DEV_MODE:
    st.sidebar.header("Developer settings")
    selected_company = st.sidebar.selectbox(
        "Company record for internal context",
        options=company_list,
        index=company_list.index(default_company),
        key="selected_company_app_3",
    )
    show_top3 = st.sidebar.checkbox("Show top 3 projects", value=False, key="show_top3_app_3")
    show_debug = st.sidebar.checkbox("Show scoring table", value=False, key="show_debug_app_3")
    if show_debug and data_note:
        st.sidebar.caption(f"Developer note: {data_note}")

if "app_3_page" not in st.session_state:
    st.session_state["app_3_page"] = "questionnaire"

if st.session_state["app_3_page"] == "questionnaire":
    st.markdown('<div class="main-title">Nature Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">From company context to credible NbS investment options.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("## Company context questionnaire")

    company_profile = render_questionnaire()
    if company_profile is not None:
        st.session_state["company_profile"] = company_profile
        st.session_state["app_3_page"] = "results"
        st.rerun()

else:
    company_profile = st.session_state.get("company_profile", {})

    st.markdown('<div class="main-title">Nature Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">From company context to credible NbS investment options.</div>',
        unsafe_allow_html=True,
    )

    recommendation, error = build_questionnaire_recommendation(
        selected_company,
        company_profile,
        df_projects,
        df_finance,
    )

    if error:
        st.error(
            "We could not generate a recommendation from the available demo data yet. "
            "Your questionnaire answers have been saved."
        )
        if show_debug:
            st.caption(f"Developer note: {error}")
    else:
        st.session_state["recommendation_app_3"] = recommendation
        render_recommendation_outputs(recommendation, company_profile, show_top3, show_debug)

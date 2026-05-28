import streamlit as st


CSRD_STATUS_OPTIONS = [
    "Yes, completed",
    "In progress",
    "Not yet",
    "Not sure / prefer not to say",
]

MATERIAL_TOPIC_OPTIONS = [
    "Water",
    "Biodiversity",
    "Land use / ecosystem degradation",
    "Climate adaptation",
    "Pollution",
    "Resource use / circularity",
    "Community / social impact",
    "Not yet defined",
]

MAIN_PRESSURE_OPTIONS = [
    "Water withdrawal / water stress",
    "Land-use change",
    "Pollution / nutrient runoff",
    "Habitat degradation",
    "Climate physical risk",
    "Supply chain dependency",
    "Regulatory / reporting pressure",
]

INDUSTRY_OPTIONS = [
    "Food & beverage",
    "Textiles & apparel",
    "Mining & materials",
    "Consumer goods",
    "Infrastructure / real estate",
    "Energy & utilities",
    "Finance",
    "Other",
]

RISK_CONCERN_OPTIONS = [
    "Operational disruption",
    "Supply chain risk",
    "Regulatory / reporting risk",
    "Reputational risk",
    "Cost increase",
    "Loss of ecosystem services",
    "Not sure yet",
]

BUSINESS_OBJECTIVE_OPTIONS = [
    "Reduce nature-related risk",
    "Support CSRD / ESRS / TNFD disclosure",
    "Identify credible NbS investment opportunities",
    "Improve supply chain resilience",
    "Support SBTN-aligned action planning",
    "Build a nature-positive investment case",
    "Explore partnership opportunities with WWF",
]

BUDGET_LEVEL_OPTIONS = [
    "Low: < $1.5M",
    "Medium: $1.5M – $3M",
    "High: > $3M",
    "Not sure / prefer not to say",
]
TIME_HORIZON_OPTIONS = [
    "Short term: 0–2 years",
    "Medium term: 2–5 years",
    "Long term: 5+ years",
    "Not sure yet",
]

QUESTIONNAIRE_WIDGET_KEYS = [
    "q_csrd_status",
    "q_material_topics",
    "q_main_pressure",
    "q_industry",
    "q_geography",
    "q_geography_detail",
    "q_key_risk_concern",
    "q_business_objective",
    "q_budget_level",
    "q_preferred_time_horizon",
]

DEFAULT_QUESTIONNAIRE_ANSWERS = {
    "csrd_status": CSRD_STATUS_OPTIONS[0],
    "material_topics": [],
    "main_pressure": MAIN_PRESSURE_OPTIONS[0],
    "industry": INDUSTRY_OPTIONS[0],
    "geography": "",
    "geography_detail": "",
    "key_risk_concern": RISK_CONCERN_OPTIONS[0],
    "business_objective": BUSINESS_OBJECTIVE_OPTIONS[0],
    "budget_level": BUDGET_LEVEL_OPTIONS[0],
    "preferred_time_horizon": TIME_HORIZON_OPTIONS[0],
}


def ensure_questionnaire_answers():
    saved_answers = st.session_state.setdefault("questionnaire_answers", {})
    for field, default_value in DEFAULT_QUESTIONNAIRE_ANSWERS.items():
        saved_answers.setdefault(
            field,
            default_value.copy() if isinstance(default_value, list) else default_value,
        )
    return saved_answers


def option_index(options, saved_value):
    if saved_value in options:
        return options.index(saved_value)
    return 0


def valid_multiselect_values(options, saved_values):
    return [value for value in (saved_values or []) if value in options]


def reset_questionnaire_answers():
    st.session_state["questionnaire_answers"] = {
        field: default.copy() if isinstance(default, list) else default
        for field, default in DEFAULT_QUESTIONNAIRE_ANSWERS.items()
    }
    for key in QUESTIONNAIRE_WIDGET_KEYS:
        st.session_state.pop(key, None)
    st.session_state.pop("company_profile", None)
    st.session_state.pop("recommendation_app_3", None)


def render_questionnaire():
    """
    Render the company-context questionnaire.

    Returns:
        dict | None: company_profile after form submission; otherwise None.
    """
    st.write(
        "This short questionnaire helps translate your company context into a relevant NbS recommendation. "
        "It does not ask for confidential financial or supply-chain data."
    )

    saved_answers = ensure_questionnaire_answers()

    with st.form("company_context_questionnaire"):
        st.markdown("### Section 1 — Sustainability readiness")

        csrd_status = st.selectbox(
            "Q1. Has your company already completed a CSRD / ESRS materiality assessment?",
            CSRD_STATUS_OPTIONS,
            index=option_index(CSRD_STATUS_OPTIONS, saved_answers.get("csrd_status")),
            key="q_csrd_status",
        )
        material_topics = st.multiselect(
            "Q2. Which nature-related topics are material for your company? Multi-select.",
            MATERIAL_TOPIC_OPTIONS,
            default=valid_multiselect_values(
                MATERIAL_TOPIC_OPTIONS,
                saved_answers.get("material_topics"),
            ),
            key="q_material_topics",
        )
        main_pressure = st.selectbox(
            "Q3. What is the main pressure your company wants to address?",
            MAIN_PRESSURE_OPTIONS,
            index=option_index(MAIN_PRESSURE_OPTIONS, saved_answers.get("main_pressure")),
            key="q_main_pressure",
        )

        st.divider()
        st.markdown("### Section 2 — Company and risk profile")

        industry = st.selectbox(
            "Q4. What is your company’s industry?",
            INDUSTRY_OPTIONS,
            index=option_index(INDUSTRY_OPTIONS, saved_answers.get("industry")),
            key="q_industry",
        )
        geography = st.text_input(
            "Q5. Where are the relevant assets, operations, or sourcing regions located?",
            value=saved_answers.get("geography", ""),
            placeholder="Example: Spain, Chile, USA, Mediterranean region",
            key="q_geography",
        )
        geography_detail = st.text_input(
            "Optional basin / city / production area",
            value=saved_answers.get("geography_detail", ""),
            placeholder="Example: Ebro basin, Santiago, cotton sourcing area",
            key="q_geography_detail",
        )
        key_risk_concern = st.selectbox(
            "Q6. What is your key nature-related risk concern?",
            RISK_CONCERN_OPTIONS,
            index=option_index(RISK_CONCERN_OPTIONS, saved_answers.get("key_risk_concern")),
            key="q_key_risk_concern",
        )
        business_objective = st.selectbox(
            "Q7. What is your main business objective?",
            BUSINESS_OBJECTIVE_OPTIONS,
            index=option_index(BUSINESS_OBJECTIVE_OPTIONS, saved_answers.get("business_objective")),
            key="q_business_objective",
        )

        st.divider()
        st.markdown("### Section 3 — Investment preferences")

        budget_level = st.selectbox(
            "Q8. What is your indicative budget level?",
            BUDGET_LEVEL_OPTIONS,
            index=option_index(BUDGET_LEVEL_OPTIONS, saved_answers.get("budget_level")),
            key="q_budget_level",
        )
        preferred_time_horizon = st.selectbox(
            "Q9. What is your preferred time horizon?",
            TIME_HORIZON_OPTIONS,
            index=option_index(TIME_HORIZON_OPTIONS, saved_answers.get("preferred_time_horizon")),
            key="q_preferred_time_horizon",
        )

        submitted = st.form_submit_button(
            "Find relevant NbS opportunities",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return None

    company_profile = {
        "csrd_status": csrd_status,
        "material_topics": material_topics,
        "main_pressure": main_pressure,
        "industry": industry,
        "geography": geography,
        "geography_detail": geography_detail,
        "key_risk_concern": key_risk_concern,
        "business_objective": business_objective,
        "budget_level": budget_level,
        "preferred_time_horizon": preferred_time_horizon,
    }
    st.session_state["questionnaire_answers"] = company_profile.copy()
    return company_profile

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

    with st.form("company_context_questionnaire"):
        st.markdown("### Section 1 — Sustainability readiness")

        csrd_status = st.selectbox(
            "Q1. Has your company already completed a CSRD / ESRS materiality assessment?",
            CSRD_STATUS_OPTIONS,
            key="q_csrd_status",
        )
        material_topics = st.multiselect(
            "Q2. Which nature-related topics are material for your company? Multi-select.",
            MATERIAL_TOPIC_OPTIONS,
            key="q_material_topics",
        )
        main_pressure = st.selectbox(
            "Q3. What is the main pressure your company wants to address?",
            MAIN_PRESSURE_OPTIONS,
            key="q_main_pressure",
        )

        st.divider()
        st.markdown("### Section 2 — Company and risk profile")

        industry = st.selectbox(
            "Q4. What is your company’s industry?",
            INDUSTRY_OPTIONS,
            key="q_industry",
        )
        geography = st.text_input(
            "Q5. Where are the relevant assets, operations, or sourcing regions located?",
            placeholder="Example: Spain, Chile, USA, Mediterranean region",
            key="q_geography",
        )
        geography_detail = st.text_input(
            "Optional basin / city / production area",
            placeholder="Example: Ebro basin, Santiago, cotton sourcing area",
            key="q_geography_detail",
        )
        key_risk_concern = st.selectbox(
            "Q6. What is your key nature-related risk concern?",
            RISK_CONCERN_OPTIONS,
            key="q_key_risk_concern",
        )
        business_objective = st.selectbox(
            "Q7. What is your main business objective?",
            BUSINESS_OBJECTIVE_OPTIONS,
            key="q_business_objective",
        )

        st.divider()
        st.markdown("### Section 3 — Investment preferences")

        budget_level = st.selectbox(
            "Q8. What is your indicative budget level?",
            BUDGET_LEVEL_OPTIONS,
            key="q_budget_level",
        )
        preferred_time_horizon = st.selectbox(
            "Q9. What is your preferred time horizon?",
            TIME_HORIZON_OPTIONS,
            key="q_preferred_time_horizon",
        )

        submitted = st.form_submit_button(
            "Find relevant NbS opportunities",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return None

    return {
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

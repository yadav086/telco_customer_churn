# ============================================================
# RETAINIQ - CUSTOMER CHURN INTELLIGENCE
# Production Streamlit Application
# ============================================================
from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="RetainIQ | Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS + ERROR LOGGING
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "model" / "Telco_customer_churn.pkl"

# All application errors are written here.
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "retainiq_errors.log"

logger = logging.getLogger("retainiq")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
    )
    logger.addHandler(file_handler)


# ============================================================
# MODEL INPUT CONTRACT
#
# This is intentionally explicit.
# Do NOT infer numeric/categorical status from the fitted
# ColumnTransformer. The uploaded model has:
#
#   Numeric:
#       Tenure_Months
#       Monthly_Charges
#       Churn_Score
#
#   Categorical:
#       all remaining fields below
#
# In particular, Senior_Citizen is a categorical "No"/"Yes"
# value in the supplied PKL.
# ============================================================
MODEL_FEATURES = [
    "City",
    "Gender",
    "Senior_Citizen",
    "Partner",
    "Dependents",
    "Phone_Service",
    "Multiple_Lines",
    "Internet_Service",
    "Online_Security",
    "Online_Backup",
    "Device_Protection",
    "Tech_Support",
    "Streaming_TV",
    "Contract",
    "Paperless_Billing",
    "Payment_Method",
    "Tenure_Months",
    "Monthly_Charges",
    "Churn_Score",
]

CATEGORICAL_COLUMNS = [
    "City",
    "Gender",
    "Senior_Citizen",
    "Partner",
    "Dependents",
    "Phone_Service",
    "Multiple_Lines",
    "Internet_Service",
    "Online_Security",
    "Online_Backup",
    "Device_Protection",
    "Tech_Support",
    "Streaming_TV",
    "Contract",
    "Paperless_Billing",
    "Payment_Method",
]

NUMERIC_COLUMNS = [
    "Tenure_Months",
    "Monthly_Charges",
    "Churn_Score",
]

# Values learned by the TargetEncoder in the supplied model.
# They are used for validation and friendly dropdowns.
MODEL_CATEGORIES = {
    "Gender": ["Female", "Male"],
    "Senior_Citizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "Phone_Service": ["No", "Yes"],
    "Multiple_Lines": ["No", "No phone service", "Yes"],
    "Internet_Service": ["DSL", "Fiber optic", "No"],
    "Online_Security": ["No", "No internet service", "Yes"],
    "Online_Backup": ["No", "No internet service", "Yes"],
    "Device_Protection": ["No", "No internet service", "Yes"],
    "Tech_Support": ["No", "No internet service", "Yes"],
    "Streaming_TV": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "Paperless_Billing": ["No", "Yes"],
    "Payment_Method": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}

# The model contains many city categories. We intentionally do
# not reproduce the model's complete schema in the customer UI.
# City is accepted as text. The TargetEncoder handles it.
KNOWN_DEFAULTS = {
    "City": "Pune",
    "Gender": "Female",
    "Senior_Citizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "Phone_Service": "Yes",
    "Multiple_Lines": "No",
    "Internet_Service": "Fiber optic",
    "Online_Security": "No",
    "Online_Backup": "No",
    "Device_Protection": "No",
    "Tech_Support": "No",
    "Streaming_TV": "No",
    "Contract": "Month-to-month",
    "Paperless_Billing": "Yes",
    "Payment_Method": "Electronic check",
    "Tenure_Months": 12,
    "Monthly_Charges": 70.0,
    "Churn_Score": 50,
}


# ============================================================
# CSS
# ============================================================
st.markdown(
    """
<style>
.stApp {
    background:
        radial-gradient(circle at 0% 0%, rgba(79,70,229,.12), transparent 28rem),
        radial-gradient(circle at 100% 0%, rgba(8,145,178,.10), transparent 28rem),
        #f8fafc;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#111827 0%,#1e1b4b 55%,#312e81 100%);
}
[data-testid="stSidebar"] * { color: #f8fafc; }
.hero {
    padding: 2.5rem;
    border-radius: 24px;
    margin-bottom: 1.5rem;
    color: white;
    background: linear-gradient(120deg,#4338ca,#7c3aed 50%,#0891b2);
    box-shadow: 0 20px 50px rgba(79,70,229,.22);
}
.hero h1 { margin:0; font-size:2.6rem; font-weight:800; letter-spacing:-.05em; }
.hero p { margin-top:.7rem; font-size:1.05rem; opacity:.92; }
.eyebrow {
    font-size:.72rem;
    font-weight:800;
    text-transform:uppercase;
    letter-spacing:.14em;
}
.card {
    background:white;
    border:1px solid #e2e8f0;
    border-radius:18px;
    padding:1.35rem;
    box-shadow:0 8px 25px rgba(15,23,42,.06);
}
.risk-card {
    background:linear-gradient(135deg,#fff1f2,#ffe4e6);
    border:1px solid #fecdd3;
    border-radius:18px;
    padding:1.35rem;
}
.safe-card {
    background:linear-gradient(135deg,#ecfdf5,#cffafe);
    border:1px solid #a7f3d0;
    border-radius:18px;
    padding:1.35rem;
}
.confidence-card {
    background:linear-gradient(135deg,#eef2ff,#e0e7ff);
    border:1px solid #c7d2fe;
    border-radius:18px;
    padding:1.35rem;
}
.big-value {
    font-size:2.35rem;
    line-height:1;
    font-weight:850;
    color:#0f172a;
}
.small-text { color:#64748b; margin-top:.4rem; }
.section-title { font-size:1.35rem; font-weight:800; color:#0f172a; }
div.stButton > button, div.stFormSubmitButton > button {
    min-height:3rem;
    border:none;
    border-radius:12px;
    color:white;
    font-weight:750;
    background:linear-gradient(90deg,#4f46e5,#7c3aed,#0891b2);
    box-shadow:0 8px 22px rgba(79,70,229,.22);
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div { border-radius:10px; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================
def frontend_label(column: str) -> str:
    return str(column).replace("_", " ")


def log_exception(message: str, exc: Exception) -> None:
    logger.error("%s | %s", message, exc)
    logger.error(traceback.format_exc())


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_category(column: str, value: Any) -> str:
    """
    Normalize categorical values before they reach TargetEncoder.

    Important:
      Senior_Citizen -> "No"/"Yes"
      0/1, True/False, "0"/"1" are accepted and converted.
    """
    if column == "Senior_Citizen":
        if pd.isna(value):
            return "No"

        text = str(value).strip().lower()

        if text in {"1", "1.0", "yes", "y", "true", "senior", "senior citizen"}:
            return "Yes"

        if text in {"0", "0.0", "no", "n", "false", "not senior", "non-senior"}:
            return "No"

        return str(value).strip()

    return clean_text(value)


def numeric_value(column: str, value: Any) -> float:
    if pd.isna(value) or str(value).strip() == "":
        raise ValueError(
            f"{frontend_label(column)} is required and must contain a numeric value."
        )

    # Handles Excel/CSV values such as "70", "70.0", "1,250".
    text = str(value).strip().replace(",", "")

    try:
        number = float(text)
    except (TypeError, ValueError):
        raise ValueError(
            f"{frontend_label(column)} must contain a numeric value. "
            f"Received: '{value}'."
        )

    if not np.isfinite(number):
        raise ValueError(
            f"{frontend_label(column)} contains an invalid numeric value."
        )

    return number


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept common variants such as:
      Senior Citizen
      Senior_Citizen
      senior citizen
    while keeping the model's internal names.
    """
    result = df.copy()

    rename = {}
    for col in result.columns:
        normalized = (
            str(col)
            .strip()
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )
        normalized = " ".join(normalized.split())

        for target in MODEL_FEATURES:
            target_norm = (
                target.lower()
                .replace("_", " ")
            )
            target_norm = " ".join(target_norm.split())

            if normalized == target_norm:
                rename[col] = target
                break

    return result.rename(columns=rename)


def prepare_input_dataframe(
    data: pd.DataFrame,
    required_columns: list[str],
) -> pd.DataFrame:
    """
    Prepare input using the known training contract.

    IMPORTANT:
    We never ask sklearn to decide whether a field is numeric.
    That decision is explicit here.
    """
    df = canonicalize_columns(data)

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            "Some required customer information is missing: "
            + ", ".join(frontend_label(c) for c in missing)
        )

    df = df[required_columns].copy()

    # Explicit categorical handling.
    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(
                lambda x, c=column: normalize_category(c, x)
            ).astype(object)

    # Explicit numeric handling.
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(
                lambda x, c=column: numeric_value(c, x)
            ).astype(float)

    # Preserve exact model feature order.
    return df[required_columns].copy()


# ============================================================
# MODEL
# ============================================================
@st.cache_resource(show_spinner="Loading the RetainIQ prediction engine...")
def load_model(path: str):
    model_file = Path(path)

    if not model_file.exists():
        raise FileNotFoundError(
            f"Prediction model was not found at: {model_file}"
        )

    loaded = joblib.load(model_file)

    if not hasattr(loaded, "predict"):
        raise TypeError("The prediction model does not support predictions.")

    if not hasattr(loaded, "predict_proba"):
        raise TypeError(
            "The prediction model does not provide probability estimates."
        )

    return loaded


try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    log_exception("Model loading failure", exc)
    st.error(
        "The RetainIQ prediction service could not be started. "
        "Please verify that the trained model file is available."
    )
    st.stop()


# ============================================================
# PREDICTION
# ============================================================
def calculate_probabilities(
    model: Any,
    probabilities: np.ndarray,
):
    classes = list(getattr(model, "classes_", [0, 1]))

    if probabilities.ndim != 2 or probabilities.shape[1] == 0:
        raise ValueError("The prediction engine returned an invalid probability result.")

    # For this trained Telco model:
    # 0 = churn
    # 1 = retain
    if 0 in classes:
        churn_index = classes.index(0)
    else:
        churn_index = int(np.argmin(classes))

    if 1 in classes:
        retain_index = classes.index(1)
    else:
        retain_index = 1 if probabilities.shape[1] > 1 else 0

    churn = probabilities[:, churn_index].astype(float)
    retain = probabilities[:, retain_index].astype(float)
    confidence = probabilities.max(axis=1).astype(float)

    return churn, retain, confidence


def predict_dataframe(model: Any, customer_df: pd.DataFrame):
    prepared = prepare_input_dataframe(
        customer_df,
        MODEL_FEATURES,
    )

    try:
        prediction = model.predict(prepared)
        probabilities = model.predict_proba(prepared)
    except Exception as exc:
        log_exception("Prediction failure", exc)

        # Convert the low-level sklearn message into a useful
        # application-level error without exposing model internals.
        raise RuntimeError(
            "The customer information could not be processed by the "
            "prediction engine. Please verify the customer values and "
            "try again."
        ) from exc

    churn, retain, confidence = calculate_probabilities(
        model,
        probabilities,
    )

    return prepared, prediction, churn, retain, confidence


def predict_single(customer: dict[str, Any]):
    raw = pd.DataFrame([customer])

    prepared, prediction, churn, retain, confidence = predict_dataframe(
        model,
        raw,
    )

    return {
        "prediction": int(prediction[0]),
        "churn_probability": float(churn[0]),
        "retain_probability": float(retain[0]),
        "confidence": float(confidence[0]),
        "model_input": prepared,
    }


def predict_bulk(uploaded_df: pd.DataFrame):
    prepared, prediction, churn, retain, confidence = predict_dataframe(
        model,
        uploaded_df,
    )

    results = uploaded_df.copy()

    results["Prediction"] = [
        "Churn" if int(x) == 0 else "Retain"
        for x in prediction
    ]
    results["Churn Probability"] = np.round(churn * 100, 2)
    results["Retention Probability"] = np.round(retain * 100, 2)
    results["Confidence Score"] = np.round(confidence * 100, 2)

    results["Risk Band"] = np.select(
        [
            churn >= 0.70,
            churn >= 0.40,
        ],
        [
            "High",
            "Moderate",
        ],
        default="Low",
    )

    results["Recommended Action"] = np.select(
        [
            churn >= 0.70,
            churn >= 0.40,
        ],
        [
            "Immediate retention outreach",
            "Proactive customer engagement",
        ],
        default="Continue normal engagement",
    )

    return results


# ============================================================
# BUSINESS MESSAGES
# ============================================================
def risk_band(churn: float) -> str:
    if churn >= 0.70:
        return "High Risk"
    if churn >= 0.40:
        return "Moderate Risk"
    return "Low Risk"


def confidence_label(confidence: float) -> str:
    if confidence >= 0.80:
        return "Strong confidence"
    if confidence >= 0.60:
        return "Good confidence"
    return "Review recommended"


def business_explanation(
    churn: float,
    retain: float,
    confidence: float,
) -> tuple[str, str]:
    risk = risk_band(churn)

    if churn >= 0.70:
        headline = "Immediate retention attention is recommended."
        body = (
            f"The assessment indicates a {churn*100:.1f}% likelihood of churn "
            f"versus {retain*100:.1f}% likelihood of retention. "
            f"This places the customer in the {risk.lower()} category. "
            "Consider prioritising this customer for a timely retention conversation, "
            "service review, personalised offer, or issue-resolution activity."
        )
    elif churn >= 0.40:
        headline = "Proactive engagement is recommended."
        body = (
            f"The assessment indicates a {churn*100:.1f}% likelihood of churn "
            f"and {retain*100:.1f}% likelihood of retention. "
            "The customer is showing a meaningful level of churn risk, so "
            "targeted engagement and service monitoring can help reduce future risk."
        )
    else:
        headline = "The customer currently appears relatively stable."
        body = (
            f"The assessment indicates a {churn*100:.1f}% likelihood of churn "
            f"and {retain*100:.1f}% likelihood of retention. "
            "Continue normal customer engagement while monitoring for changes "
            "in usage, service experience, tenure, or account behaviour."
        )

    body += (
        f" The prediction confidence score is {confidence*100:.1f}% "
        f"({confidence_label(confidence).lower()})."
    )

    return headline, body


# ============================================================
# CHARTS
# ============================================================
def gauge(probability: float, title: str):
    pct = probability * 100

    if pct >= 70:
        bar = "#e11d48"
    elif pct >= 40:
        bar = "#d97706"
    else:
        bar = "#059669"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 42}},
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": bar},
                "steps": [
                    {"range": [0, 40], "color": "#d1fae5"},
                    {"range": [40, 70], "color": "#fef3c7"},
                    {"range": [70, 100], "color": "#ffe4e6"},
                ],
            },
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def confidence_gauge(confidence: float):
    pct = confidence * 100

    if pct >= 80:
        bar = "#059669"
    elif pct >= 60:
        bar = "#d97706"
    else:
        bar = "#e11d48"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 42}},
            title={"text": "Prediction Confidence Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": bar},
                "steps": [
                    {"range": [0, 60], "color": "#ffe4e6"},
                    {"range": [60, 80], "color": "#fef3c7"},
                    {"range": [80, 100], "color": "#d1fae5"},
                ],
            },
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ============================================================
# UI FIELD
# ============================================================
def get_default(column: str):
    return KNOWN_DEFAULTS.get(column, "")


def render_field(column: str, key_prefix: str):
    label = frontend_label(column)
    key = f"{key_prefix}_{column}"

    if column in MODEL_CATEGORIES:
        options = MODEL_CATEGORIES[column]
        default = str(get_default(column))

        try:
            index = options.index(default)
        except ValueError:
            index = 0

        return st.selectbox(
            label,
            options,
            index=index,
            key=key,
        )

    if column == "City":
        return st.text_input(
            label,
            value=str(get_default(column)),
            key=key,
            help="Enter the customer's city. City is treated as categorical information.",
        )

    if column == "Tenure_Months":
        return st.number_input(
            label,
            min_value=0,
            max_value=120,
            value=int(get_default(column)),
            step=1,
            key=key,
        )

    if column == "Monthly_Charges":
        return st.number_input(
            label,
            min_value=0.0,
            max_value=10000.0,
            value=float(get_default(column)),
            step=1.0,
            key=key,
        )

    if column == "Churn_Score":
        return st.number_input(
            label,
            min_value=0,
            max_value=100,
            value=int(get_default(column)),
            step=1,
            key=key,
        )

    return st.text_input(
        label,
        value=str(get_default(column)),
        key=key,
    )


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🚀 RetainIQ")
    st.caption("Customer Churn Intelligence Platform")
    st.divider()

    st.markdown(
        """
### Assessment Workflow

**01** Customer information

**02** Churn risk assessment

**03** Probability analysis

**04** Confidence score

**05** Retention recommendation
"""
    )

    st.divider()

    st.success("Prediction Engine Ready")
    st.caption("Single customer assessment")
    st.caption("Bulk portfolio analytics")
    st.caption("Confidence scoring")
    st.caption("Automatic error logging")

    st.divider()

    st.markdown("### Business Risk Bands")
    st.markdown(
        """
🔴 **High Risk**  
70% or higher churn probability

🟠 **Moderate Risk**  
40%–69.9% churn probability

🟢 **Low Risk**  
Below 40% churn probability
"""
    )


# ============================================================
# HERO
# ============================================================
st.markdown(
    """
<div class="hero">
<div class="eyebrow">Customer Intelligence Platform</div>
<h1>Predict Churn. Protect Customers.</h1>
<p>
Turn customer information into practical retention intelligence.
Assess individual customers or analyse an entire customer portfolio
with probability, confidence and recommended next actions.
</p>
</div>
""",
    unsafe_allow_html=True,
)


a, b, c, d = st.columns(4)
a.metric("Engine Status", "Ready")
b.metric("Customer Features", "19")
c.metric("Prediction Mode", "Real Time")
d.metric("Analytics", "Enabled")


# ============================================================
# TABS
# ============================================================
single_tab, bulk_tab = st.tabs(
    ["👤 Single Customer", "📂 Bulk Customer Analytics"]
)


# ============================================================
# SINGLE CUSTOMER
# ============================================================
with single_tab:
    st.markdown("### Customer Churn Assessment")
    st.caption(
        "Provide the customer's current profile and account information "
        "to receive a probability-based churn assessment."
    )

    with st.form("customer_prediction_form"):
        customer = {}

        left, right = st.columns(2, gap="large")
        midpoint = (len(MODEL_FEATURES) + 1) // 2

        with left:
            st.markdown("#### 👤 Customer Profile")
            for column in MODEL_FEATURES[:midpoint]:
                customer[column] = render_field(column, "left")

        with right:
            st.markdown("#### ⚙ Services & Account")
            for column in MODEL_FEATURES[midpoint:]:
                customer[column] = render_field(column, "right")

        submitted = st.form_submit_button(
            "🚀 Run Churn Assessment",
            width="stretch",
        )

    if submitted:
        try:
            with st.spinner("Analysing customer risk and preparing the assessment..."):
                result = predict_single(customer)

            prediction = result["prediction"]
            churn = result["churn_probability"]
            retain = result["retain_probability"]
            confidence = result["confidence"]

            is_churn = prediction == 0
            churn_pct = churn * 100
            retain_pct = retain * 100
            confidence_pct = confidence * 100

            st.divider()
            st.markdown("### 🎯 Customer Risk Assessment")

            r1, r2, r3, r4 = st.columns(4)

            with r1:
                css = "risk-card" if is_churn else "safe-card"
                outcome = "CHURN" if is_churn else "RETAIN"
                st.markdown(
                    f"""
<div class="{css}">
<div class="eyebrow">Prediction</div>
<div class="big-value">{outcome}</div>
<div class="small-text">Most likely customer outcome</div>
</div>
""",
                    unsafe_allow_html=True,
                )

            with r2:
                st.markdown(
                    f"""
<div class="risk-card">
<div class="eyebrow">Churn Probability</div>
<div class="big-value">{churn_pct:.1f}%</div>
<div class="small-text">Estimated likelihood of churn</div>
</div>
""",
                    unsafe_allow_html=True,
                )

            with r3:
                st.markdown(
                    f"""
<div class="safe-card">
<div class="eyebrow">Retention Probability</div>
<div class="big-value">{retain_pct:.1f}%</div>
<div class="small-text">Estimated likelihood of retention</div>
</div>
""",
                    unsafe_allow_html=True,
                )

            with r4:
                st.markdown(
                    f"""
<div class="confidence-card">
<div class="eyebrow">Confidence Score</div>
<div class="big-value">{confidence_pct:.1f}%</div>
<div class="small-text">{confidence_label(confidence)}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

            g1, g2 = st.columns(2)

            with g1:
                st.plotly_chart(
                    gauge(churn, "Churn Risk"),
                    width="stretch",
                    config={"displayModeBar": False},
                )

            with g2:
                st.plotly_chart(
                    confidence_gauge(confidence),
                    width="stretch",
                    config={"displayModeBar": False},
                )

            chart1, chart2 = st.columns(2)

            with chart1:
                probability_df = pd.DataFrame(
                    {
                        "Outcome": ["Churn", "Retain"],
                        "Probability": [churn_pct, retain_pct],
                    }
                )

                fig = px.pie(
                    probability_df,
                    names="Outcome",
                    values="Probability",
                    hole=0.60,
                    title="Customer Outcome Probability",
                    color="Outcome",
                    color_discrete_map={
                        "Churn": "#e11d48",
                        "Retain": "#059669",
                    },
                )
                fig.update_traces(textinfo="label+percent")
                st.plotly_chart(fig, width="stretch")

            with chart2:
                fig = px.bar(
                    probability_df,
                    x="Outcome",
                    y="Probability",
                    text="Probability",
                    title="Probability Comparison",
                    color="Outcome",
                    color_discrete_map={
                        "Churn": "#e11d48",
                        "Retain": "#059669",
                    },
                )
                fig.update_traces(
                    texttemplate="%{text:.1f}%",
                    textposition="outside",
                )
                fig.update_yaxes(range=[0, 100])
                st.plotly_chart(fig, width="stretch")

            st.markdown("### 💡 What This Assessment Means")

            headline, explanation = business_explanation(
                churn,
                retain,
                confidence,
            )

            if churn >= 0.70:
                st.error(f"🔴 **{headline}**\n\n{explanation}")
            elif churn >= 0.40:
                st.warning(f"🟠 **{headline}**\n\n{explanation}")
            else:
                st.success(f"🟢 **{headline}**\n\n{explanation}")

            st.markdown("### 📌 Recommended Next Step")

            if churn >= 0.70:
                st.markdown(
                    """
**Priority 1 — Retention intervention**

- Contact the customer promptly.
- Review recent service experience.
- Consider a personalised retention offer.
- Investigate unresolved support or billing issues.
- Monitor the account after the intervention.
"""
                )
            elif churn >= 0.40:
                st.markdown(
                    """
**Priority 2 — Proactive engagement**

- Review the customer's current services.
- Identify opportunities for better service fit.
- Send a targeted engagement communication.
- Monitor the account for increasing risk.
"""
                )
            else:
                st.markdown(
                    """
**Priority 3 — Maintain relationship**

- Continue normal engagement.
- Maintain service quality.
- Monitor future account behaviour.
- Reassess if customer circumstances change.
"""
                )

            with st.expander("View Customer Assessment Inputs"):
                display = pd.DataFrame([customer])
                display.columns = [frontend_label(c) for c in display.columns]
                st.dataframe(
                    display,
                    width="stretch",
                    hide_index=True,
                )

        except ValueError as exc:
            log_exception("Customer input validation error", exc)

            st.error(
                "### We need a small correction before completing the assessment"
            )
            st.write(
                str(exc)
            )
            st.info(
                "Please review the highlighted customer information and "
                "make sure categorical fields contain valid selections and "
                "numeric fields contain numbers."
            )

        except Exception as exc:
            log_exception("Customer-facing prediction error", exc)

            st.error(
                "### We could not complete the customer assessment"
            )
            st.write(
                "The customer information was received, but the prediction "
                "engine was unable to complete the assessment."
            )
            st.info(
                "Please review the entered customer details and try again. "
                "If the same issue continues, the technical error has been "
                "recorded automatically for troubleshooting."
            )


# ============================================================
# BULK
# ============================================================
with bulk_tab:
    st.markdown("### 📂 Bulk Customer Churn Analytics")
    st.caption(
        "Upload a CSV containing customer records. The application will "
        "validate and normalize the data before generating portfolio-level "
        "risk, probability and confidence analytics."
    )

    sample = pd.DataFrame(
        [{c: KNOWN_DEFAULTS.get(c, "") for c in MODEL_FEATURES}]
    )

    st.download_button(
        "⬇ Download Sample CSV",
        data=sample.to_csv(index=False).encode("utf-8"),
        file_name="customer_churn_sample.csv",
        mime="text/csv",
        width="stretch",
    )

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            uploaded_df = canonicalize_columns(uploaded_df)

            st.success(
                f"Successfully loaded {len(uploaded_df):,} customer records."
            )

            missing = [
                c for c in MODEL_FEATURES
                if c not in uploaded_df.columns
            ]

            if missing:
                st.error(
                    "The uploaded file is missing: "
                    + ", ".join(frontend_label(c) for c in missing)
                )
            else:
                st.markdown("### Data Preview")

                preview = uploaded_df.head(10).copy()
                preview.columns = [
                    frontend_label(c) for c in preview.columns
                ]

                st.dataframe(
                    preview,
                    width="stretch",
                    hide_index=True,
                )

                if st.button(
                    "🚀 Run Bulk Churn Assessment",
                    width="stretch",
                ):
                    try:
                        with st.spinner(
                            "Analysing the customer portfolio..."
                        ):
                            results = predict_bulk(uploaded_df)

                        st.session_state["prediction_results"] = results
                        st.success(
                            "Bulk assessment completed successfully."
                        )

                    except ValueError as exc:
                        log_exception(
                            "Bulk input validation error",
                            exc,
                        )
                        st.error(
                            "The uploaded customer data needs correction."
                        )
                        st.write(str(exc))

                    except Exception as exc:
                        log_exception(
                            "Bulk prediction error",
                            exc,
                        )
                        st.error(
                            "The bulk assessment could not be completed. "
                            "The technical error has been recorded automatically."
                        )

        except Exception as exc:
            log_exception("CSV reading error", exc)
            st.error(
                "The uploaded CSV could not be read. "
                "Please verify that it is a valid CSV file."
            )

    # ========================================================
    # RESULTS
    # ========================================================
    if "prediction_results" in st.session_state:
        results = st.session_state["prediction_results"]

        st.divider()
        st.markdown("### 📊 Portfolio Risk Overview")

        total = len(results)
        churn_count = int((results["Prediction"] == "Churn").sum())
        retain_count = total - churn_count

        churn_rate = (
            churn_count / total * 100
            if total
            else 0
        )

        avg_confidence = results["Confidence Score"].mean()
        avg_churn = results["Churn Probability"].mean()
        avg_retain = results["Retention Probability"].mean()

        high = int((results["Risk Band"] == "High").sum())
        moderate = int((results["Risk Band"] == "Moderate").sum())
        low = int((results["Risk Band"] == "Low").sum())

        k1, k2, k3, k4, k5, k6 = st.columns(6)

        k1.metric("Customers", f"{total:,}")
        k2.metric("Predicted Churn", f"{churn_count:,}")
        k3.metric("Predicted Retain", f"{retain_count:,}")
        k4.metric("Churn Rate", f"{churn_rate:.1f}%")
        k5.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        k6.metric("High Risk", f"{high:,}")

        # ----------------------------------------------------
        # Chart 1
        # ----------------------------------------------------
        c1, c2 = st.columns(2)

        with c1:
            distribution = pd.DataFrame(
                {
                    "Outcome": ["Churn", "Retain"],
                    "Customers": [churn_count, retain_count],
                }
            )
            fig = px.pie(
                distribution,
                names="Outcome",
                values="Customers",
                hole=0.58,
                title="Churn vs Retain Distribution",
                color="Outcome",
                color_discrete_map={
                    "Churn": "#e11d48",
                    "Retain": "#059669",
                },
            )
            fig.update_traces(textinfo="label+percent")
            st.plotly_chart(fig, width="stretch")

        with c2:
            risk_df = pd.DataFrame(
                {
                    "Risk Band": ["High", "Moderate", "Low"],
                    "Customers": [high, moderate, low],
                }
            )
            fig = px.bar(
                risk_df,
                x="Risk Band",
                y="Customers",
                text="Customers",
                title="Customer Risk Distribution",
                color="Risk Band",
                color_discrete_map={
                    "High": "#e11d48",
                    "Moderate": "#d97706",
                    "Low": "#059669",
                },
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, width="stretch")

        # ----------------------------------------------------
        # Chart 2
        # ----------------------------------------------------
        c3, c4 = st.columns(2)

        with c3:
            fig = px.histogram(
                results,
                x="Churn Probability",
                nbins=20,
                title="Churn Probability Distribution",
            )
            fig.add_vline(
                x=40,
                line_dash="dash",
                annotation_text="40% Moderate",
            )
            fig.add_vline(
                x=70,
                line_dash="dash",
                annotation_text="70% High",
            )
            st.plotly_chart(fig, width="stretch")

        with c4:
            fig = px.histogram(
                results,
                x="Confidence Score",
                nbins=20,
                title="Prediction Confidence Distribution",
            )
            fig.add_vline(
                x=60,
                line_dash="dash",
                annotation_text="60% Review",
            )
            fig.add_vline(
                x=80,
                line_dash="dash",
                annotation_text="80% Strong",
            )
            st.plotly_chart(fig, width="stretch")

        # ----------------------------------------------------
        # Chart 3
        # ----------------------------------------------------
        c5, c6 = st.columns(2)

        with c5:
            portfolio_metrics = pd.DataFrame(
                {
                    "Metric": [
                        "Churn Probability",
                        "Retention Probability",
                        "Confidence Score",
                    ],
                    "Percentage": [
                        avg_churn,
                        avg_retain,
                        avg_confidence,
                    ],
                }
            )
            fig = px.bar(
                portfolio_metrics,
                x="Metric",
                y="Percentage",
                text="Percentage",
                title="Portfolio Average Assessment",
            )
            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
            )
            fig.update_yaxes(range=[0, 100])
            st.plotly_chart(fig, width="stretch")

        with c6:
            risk_share = pd.DataFrame(
                {
                    "Risk Band": ["High", "Moderate", "Low"],
                    "Customers": [high, moderate, low],
                }
            )
            fig = px.pie(
                risk_share,
                names="Risk Band",
                values="Customers",
                hole=0.55,
                title="Portfolio Risk Share",
                color="Risk Band",
                color_discrete_map={
                    "High": "#e11d48",
                    "Moderate": "#d97706",
                    "Low": "#059669",
                },
            )
            fig.update_traces(textinfo="label+percent")
            st.plotly_chart(fig, width="stretch")

        # ----------------------------------------------------
        # Risk / confidence matrix
        # ----------------------------------------------------
        st.markdown("### 🔎 Risk vs Confidence Analysis")

        hover_columns = [
            c for c in [
                "Prediction",
                "Churn Probability",
                "Retention Probability",
                "Confidence Score",
                "Risk Band",
                "Recommended Action",
            ]
            if c in results.columns
        ]

        scatter = px.scatter(
            results,
            x="Churn Probability",
            y="Confidence Score",
            color="Risk Band",
            hover_data=hover_columns,
            title="Customer Churn Risk vs Prediction Confidence",
            color_discrete_map={
                "High": "#e11d48",
                "Moderate": "#d97706",
                "Low": "#059669",
            },
        )

        scatter.add_vline(x=40, line_dash="dash")
        scatter.add_vline(x=70, line_dash="dash")
        scatter.add_hline(y=60, line_dash="dash")
        scatter.update_layout(height=500)

        st.plotly_chart(scatter, width="stretch")

        # ----------------------------------------------------
        # Portfolio interpretation
        # ----------------------------------------------------
        st.markdown("### 💡 Portfolio Interpretation")

        if churn_rate >= 40:
            st.error(
                f"🔴 **Elevated portfolio churn exposure:** "
                f"{churn_rate:.1f}% of assessed customers are predicted to churn. "
                "Prioritise high-risk customers for retention campaigns."
            )
        elif churn_rate >= 20:
            st.warning(
                f"🟠 **Meaningful churn exposure:** "
                f"{churn_rate:.1f}% of assessed customers are predicted to churn. "
                "Proactive engagement should focus on the highest-risk accounts."
            )
        else:
            st.success(
                f"🟢 **Relatively stable portfolio:** "
                f"{churn_rate:.1f}% of assessed customers are predicted to churn. "
                "Continue monitoring and maintain customer engagement."
            )

        st.markdown("### 👥 Customer-Level Results")

        display_df = results.copy()
        display_df.columns = [
            frontend_label(c) for c in display_df.columns
        ]

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "⬇ Download Churn Prediction Results",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="customer_churn_predictions.csv",
            mime="text/csv",
            width="stretch",
        )


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    f"""
<div style="
    text-align:center;
    padding:2rem 0 1rem;
    color:#64748b;
    font-size:.85rem;
">
RetainIQ • Customer Churn Intelligence Platform
<br>
Application errors are recorded in: {LOG_FILE}
</div>
""",
    unsafe_allow_html=True,
)

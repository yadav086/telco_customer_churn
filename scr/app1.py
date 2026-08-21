# ============================================================
# RETAINIQ - CUSTOMER CHURN PREDICTION
# Production Streamlit Application
# ============================================================

from __future__ import annotations

import sys
import logging
import traceback
from pathlib import Path
from copy import deepcopy
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
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("retainiq")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR.parent
    / "model"
    / "Telco_customer_churn.pkl"
)

EXCEL_DEFAULT_PATH = (
    BASE_DIR
    / "config"
    / "customer_input_values.xlsx"
)


# ============================================================
# FALLBACK FEATURES
# ============================================================

DEFAULT_FEATURES = [
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
    "Paperless_Billing",
    "Payment_Method",
    "Contract",
    "Tenure_Months",
    "Monthly_Charges",
    "Churn_Score",
]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(
            circle at 0% 0%,
            rgba(79,70,229,.12),
            transparent 28rem
        ),
        radial-gradient(
            circle at 100% 0%,
            rgba(8,145,178,.10),
            transparent 28rem
        ),
        #f8fafc;
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #111827 0%,
            #1e1b4b 55%,
            #312e81 100%
        );
}

[data-testid="stSidebar"] * {
    color: #f8fafc;
}

.hero {
    padding: 2.5rem;
    border-radius: 24px;
    margin-bottom: 1.5rem;
    color: white;
    background:
        linear-gradient(
            120deg,
            #4338ca,
            #7c3aed 50%,
            #0891b2
        );
    box-shadow:
        0 20px 50px rgba(79,70,229,.22);
}

.hero h1 {
    margin: 0;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -.05em;
}

.hero p {
    margin-top: .7rem;
    font-size: 1.05rem;
    opacity: .92;
}

.eyebrow {
    font-size: .72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .14em;
}

.card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 1.35rem;
    box-shadow:
        0 8px 25px rgba(15,23,42,.06);
}

.risk-card {
    background:
        linear-gradient(
            135deg,
            #fff1f2,
            #ffe4e6
        );
    border: 1px solid #fecdd3;
    border-radius: 18px;
    padding: 1.35rem;
}

.safe-card {
    background:
        linear-gradient(
            135deg,
            #ecfdf5,
            #cffafe
        );
    border: 1px solid #a7f3d0;
    border-radius: 18px;
    padding: 1.35rem;
}

.big-value {
    font-size: 2.5rem;
    line-height: 1;
    font-weight: 850;
    color: #0f172a;
}

.small-text {
    color: #64748b;
    margin-top: .4rem;
}

div.stButton > button,
div.stFormSubmitButton > button {
    min-height: 3rem;
    border: none;
    border-radius: 12px;
    color: white;
    font-weight: 750;
    background:
        linear-gradient(
            90deg,
            #4f46e5,
            #7c3aed,
            #0891b2
        );
    box-shadow:
        0 8px 22px rgba(79,70,229,.22);
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 10px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# FRONTEND LABEL
# ============================================================

def frontend_label(column: str) -> str:
    return str(column).replace("_", " ")


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading churn model...")
def load_churn_model(path: str):

    model_file = Path(path)

    if not model_file.exists():
        raise FileNotFoundError(
            f"""
Model file was not found.

Expected:

{model_file}
"""
        )

    model = joblib.load(model_file)

    if not hasattr(model, "predict"):
        raise TypeError(
            "Saved model does not contain predict()."
        )

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Saved model does not contain predict_proba()."
        )

    logger.info(
        "Loaded model: %s",
        model.__class__.__name__,
    )

    return model


# ============================================================
# RECURSIVE OBJECT SEARCH
# ============================================================

def find_objects_by_class(
    model: Any,
    class_names: set[str],
):

    found = []
    visited = set()

    def search(obj):

        if obj is None:
            return

        if id(obj) in visited:
            return

        visited.add(id(obj))

        class_name = obj.__class__.__name__

        if class_name in class_names:
            found.append(obj)

        # Pipeline
        if hasattr(obj, "named_steps"):
            try:
                for step in obj.named_steps.values():
                    search(step)
            except Exception:
                pass

        # ColumnTransformer
        if hasattr(obj, "transformers_"):
            try:
                for _, transformer, _ in obj.transformers_:
                    if transformer not in (
                        "drop",
                        "passthrough",
                    ):
                        search(transformer)
            except Exception:
                pass

        # Nested transformers
        if hasattr(obj, "transformers"):
            try:
                for _, transformer, _ in obj.transformers:
                    if transformer not in (
                        "drop",
                        "passthrough",
                    ):
                        search(transformer)
            except Exception:
                pass

    search(model)

    return found


# ============================================================
# FIND COLUMN TRANSFORMER
# ============================================================

def find_column_transformer(model: Any):

    transformers = find_objects_by_class(
        model,
        {"ColumnTransformer"},
    )

    return transformers[0] if transformers else None


# ============================================================
# FIND TARGET ENCODERS
# ============================================================

def find_target_encoders(model: Any):

    return find_objects_by_class(
        model,
        {"TargetEncoder"},
    )


def find_target_encoder(model: Any):

    encoders = find_target_encoders(model)

    if not encoders:
        return None

    return encoders[0]


# ============================================================
# MODEL FEATURES
# ============================================================

def get_model_features(model: Any):

    features = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if features is not None:
        return list(features)

    if hasattr(model, "named_steps"):

        for step in model.named_steps.values():

            features = getattr(
                step,
                "feature_names_in_",
                None,
            )

            if features is not None:
                return list(features)

    transformer = find_column_transformer(model)

    if transformer is not None:

        features = getattr(
            transformer,
            "feature_names_in_",
            None,
        )

        if features is not None:
            return list(features)

    return DEFAULT_FEATURES.copy()


# ============================================================
# COLUMN TYPE INFORMATION
# ============================================================

def get_transformer_columns(
    model: Any,
):

    categorical = []
    numeric = []

    transformer = find_column_transformer(model)

    if transformer is None:
        return categorical, numeric

    try:

        for (
            name,
            fitted_transformer,
            columns,
        ) in transformer.transformers_:

            if name == "remainder":
                continue

            if fitted_transformer in (
                "drop",
                "passthrough",
            ):
                continue

            if not isinstance(
                columns,
                (list, tuple, np.ndarray),
            ):
                continue

            transformer_name = (
                fitted_transformer
                .__class__
                .__name__
                .lower()
            )

            columns = list(columns)

            is_categorical = any(
                keyword in transformer_name
                for keyword in [
                    "targetencoder",
                    "onehotencoder",
                    "ordinalencoder",
                    "encoder",
                    "categorical",
                ]
            )

            if is_categorical:
                categorical.extend(columns)
            else:
                numeric.extend(columns)

    except Exception as exc:

        logger.warning(
            "Could not inspect transformer columns: %s",
            exc,
        )

    categorical = list(
        dict.fromkeys(categorical)
    )

    numeric = list(
        dict.fromkeys(numeric)
    )

    return categorical, numeric


def get_categorical_columns(model):

    categorical, _ = get_transformer_columns(model)

    # TargetEncoder is authoritative
    for encoder in find_target_encoders(model):

        features = getattr(
            encoder,
            "feature_names_in_",
            None,
        )

        if features is not None:
            categorical.extend(
                list(features)
            )

    return list(
        dict.fromkeys(categorical)
    )


def get_numeric_columns(model):

    _, numeric = get_transformer_columns(model)

    return numeric


# ============================================================
# TARGET ENCODER DETAILS
# ============================================================

def get_encoder_for_column(
    model,
    column,
):

    encoders = find_target_encoders(model)

    for encoder in encoders:

        features = getattr(
            encoder,
            "feature_names_in_",
            None,
        )

        if features is None:
            continue

        if column in list(features):
            return encoder

    return None


def get_encoder_categories(
    model,
    column,
):

    encoder = get_encoder_for_column(
        model,
        column,
    )

    if encoder is None:
        return None

    features = getattr(
        encoder,
        "feature_names_in_",
        None,
    )

    categories = getattr(
        encoder,
        "categories_",
        None,
    )

    if (
        features is None
        or categories is None
    ):
        return None

    try:

        index = list(features).index(
            column
        )

        return list(
            categories[index]
        )

    except Exception:

        return None


# ============================================================
# IMPORTANT:
# NORMALIZE TARGET ENCODER CATEGORIES
# ============================================================

def normalize_target_encoder_for_prediction(
    model,
):
    """
    Creates a deep copy of the model and makes TargetEncoder
    category types consistent.

    This fixes errors such as:

        TypeError:
        '<' not supported between instances of
        'float' and 'str'

    caused by sklearn trying to sort mixed-type categories.
    """

    try:

        prediction_model = deepcopy(model)

    except Exception as exc:

        logger.warning(
            "Could not deepcopy model. "
            "Using original model: %s",
            exc,
        )

        prediction_model = model

    encoders = find_target_encoders(
        prediction_model
    )

    for encoder in encoders:

        categories = getattr(
            encoder,
            "categories_",
            None,
        )

        if categories is None:
            continue

        new_categories = []

        for category_values in categories:

            values = list(
                category_values
            )

            # ------------------------------------------------
            # Detect mixed Python types
            # ------------------------------------------------

            type_names = {
                type(value).__name__
                for value in values
                if not pd.isna(value)
            }

            mixed_types = len(type_names) > 1

            # ------------------------------------------------
            # If mixed categories exist, use strings
            # ------------------------------------------------

            if mixed_types:

                logger.warning(
                    "Mixed TargetEncoder categories detected: %s",
                    type_names,
                )

                normalized = []

                for value in values:

                    if pd.isna(value):
                        normalized.append(
                            "__RETAINIQ_NAN__"
                        )
                    else:
                        normalized.append(
                            str(value).strip()
                        )

                new_categories.append(
                    np.asarray(
                        normalized,
                        dtype=object,
                    )
                )

            else:

                new_categories.append(
                    np.asarray(
                        values,
                        dtype=object,
                    )
                )

        try:

            encoder.categories_ = new_categories

        except Exception as exc:

            logger.warning(
                "Could not update TargetEncoder categories: %s",
                exc,
            )

    return prediction_model


# ============================================================
# NORMALIZE INPUT FOR TARGET ENCODER
# ============================================================

def normalize_target_encoder_input(
    df: pd.DataFrame,
    model,
):

    result = df.copy()

    encoders = find_target_encoders(
        model
    )

    for encoder in encoders:

        features = getattr(
            encoder,
            "feature_names_in_",
            None,
        )

        categories = getattr(
            encoder,
            "categories_",
            None,
        )

        if (
            features is None
            or categories is None
        ):
            continue

        for index, column in enumerate(
            features
        ):

            if column not in result.columns:
                continue

            if index >= len(categories):
                continue

            learned_categories = list(
                categories[index]
            )

            non_null_types = {
                type(value).__name__
                for value in learned_categories
                if not pd.isna(value)
            }

            mixed_types = (
                len(non_null_types) > 1
            )

            # ------------------------------------------------
            # Mixed categories
            # ------------------------------------------------

            if mixed_types:

                result[column] = (
                    result[column]
                    .apply(
                        lambda value:
                        "__RETAINIQ_NAN__"
                        if pd.isna(value)
                        else str(value).strip()
                    )
                    .astype(object)
                )

                continue

            # ------------------------------------------------
            # String categories
            # ------------------------------------------------

            if (
                non_null_types
                and non_null_types.issubset(
                    {"str"}
                )
            ):

                result[column] = (
                    result[column]
                    .apply(
                        lambda value:
                        np.nan
                        if pd.isna(value)
                        else str(value).strip()
                    )
                    .astype(object)
                )

                continue

            # ------------------------------------------------
            # Numeric categories
            # ------------------------------------------------

            if non_null_types.issubset(
                {
                    "int",
                    "float",
                    "int64",
                    "float64",
                }
            ):

                result[column] = pd.to_numeric(
                    result[column],
                    errors="coerce",
                )

    return result


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input_dataframe(
    data: pd.DataFrame,
    required_columns: list[str],
    model: Any,
):

    df = data.copy()

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing model features:\n\n"
            + "\n".join(
                f"• {frontend_label(c)}"
                for c in missing
            )
        )

    # --------------------------------------------------------
    # Keep model features only
    # --------------------------------------------------------

    df = df[
        required_columns
    ].copy()

    categorical_columns = (
        get_categorical_columns(
            model
        )
    )

    numeric_columns = (
        get_numeric_columns(
            model
        )
    )

    # --------------------------------------------------------
    # Categorical columns
    # --------------------------------------------------------

    for column in categorical_columns:

        if column not in df.columns:
            continue

        df[column] = (
            df[column]
            .apply(
                lambda value:
                np.nan
                if pd.isna(value)
                else (
                    value.strip()
                    if isinstance(
                        value,
                        str,
                    )
                    else value
                )
            )
            .astype(object)
        )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    for column in numeric_columns:

        if column not in df.columns:
            continue

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Remaining columns
    # --------------------------------------------------------

    known_columns = set(
        categorical_columns
        + numeric_columns
    )

    for column in required_columns:

        if column in known_columns:
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        else:

            df[column] = (
                df[column]
                .astype(object)
            )

    # --------------------------------------------------------
    # TargetEncoder-specific normalization
    # --------------------------------------------------------

    df = normalize_target_encoder_input(
        df,
        model,
    )

    return df[
        required_columns
    ]


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_input(
    df: pd.DataFrame,
    required_columns: list[str],
):

    if df is None:
        raise ValueError(
            "No input data was supplied."
        )

    if len(df) == 0:
        raise ValueError(
            "No customer records were supplied."
        )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing model features:\n\n"
            + "\n".join(
                f"• {frontend_label(c)}"
                for c in missing
            )
        )


# ============================================================
# SAFE PREDICTION
# ============================================================

def run_prediction(
    model,
    df,
):

    # --------------------------------------------------------
    # First attempt with original model
    # --------------------------------------------------------

    try:

        prediction = model.predict(df)

        probabilities = model.predict_proba(df)

        return (
            prediction,
            probabilities,
            None,
            model,
        )

    except Exception as original_error:

        logger.warning(
            "Original model prediction failed."
        )

        logger.warning(
            "%s",
            original_error,
        )

        # ----------------------------------------------------
        # Second attempt:
        # normalize TargetEncoder
        # ----------------------------------------------------

        try:

            logger.info(
                "Retrying with normalized TargetEncoder model."
            )

            safe_model = (
                normalize_target_encoder_for_prediction(
                    model
                )
            )

            safe_df = (
                normalize_target_encoder_input(
                    df,
                    safe_model,
                )
            )

            prediction = (
                safe_model.predict(
                    safe_df
                )
            )

            probabilities = (
                safe_model.predict_proba(
                    safe_df
                )
            )

            logger.info(
                "Prediction succeeded after "
                "TargetEncoder normalization."
            )

            return (
                prediction,
                probabilities,
                None,
                safe_model,
            )

        except Exception as normalized_error:

            logger.error(
                "NORMALIZED MODEL PREDICTION FAILED"
            )

            logger.error(
                traceback.format_exc()
            )

            return (
                None,
                None,
                normalized_error,
                model,
            )


# ============================================================
# PROBABILITIES
# ============================================================

def calculate_probabilities(
    model,
    probabilities,
):

    classes = getattr(
        model,
        "classes_",
        None,
    )

    if classes is None:
        classes = [0, 1]

    classes = list(classes)

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Your application assumes:
    #
    # 0 = Churn
    # 1 = Retain
    #
    # --------------------------------------------------------

    if 0 in classes:

        churn_index = classes.index(0)

    else:

        churn_index = 0

    if 1 in classes:

        retain_index = classes.index(1)

    else:

        retain_index = (
            1
            if probabilities.shape[1] > 1
            else 0
        )

    churn_probability = (
        probabilities[
            :,
            churn_index,
        ]
    )

    retain_probability = (
        probabilities[
            :,
            retain_index,
        ]
    )

    confidence = (
        probabilities.max(
            axis=1
        )
    )

    return (
        churn_probability,
        retain_probability,
        confidence,
    )


# ============================================================
# SINGLE PREDICTION
# ============================================================

def predict_single(
    model,
    customer,
    required_columns,
):

    raw_df = pd.DataFrame(
        [customer]
    )

    validate_input(
        raw_df,
        required_columns,
    )

    model_df = (
        prepare_input_dataframe(
            raw_df,
            required_columns,
            model,
        )
    )

    (
        prediction,
        probabilities,
        error,
        used_model,
    ) = run_prediction(
        model,
        model_df,
    )

    if error is not None:
        raise error

    (
        churn_probability,
        retain_probability,
        confidence,
    ) = calculate_probabilities(
        used_model,
        probabilities,
    )

    return {

        "prediction":
            int(prediction[0]),

        "churn_probability":
            float(
                churn_probability[0]
            ),

        "retain_probability":
            float(
                retain_probability[0]
            ),

        "confidence":
            float(
                confidence[0]
            ),

        "model_input":
            model_df,

    }


# ============================================================
# BULK PREDICTION
# ============================================================

def predict_bulk(
    model,
    uploaded_df,
    required_columns,
):

    validate_input(
        uploaded_df,
        required_columns,
    )

    model_df = (
        prepare_input_dataframe(
            uploaded_df,
            required_columns,
            model,
        )
    )

    (
        prediction,
        probabilities,
        error,
        used_model,
    ) = run_prediction(
        model,
        model_df,
    )

    if error is not None:
        raise error

    (
        churn_probability,
        retain_probability,
        confidence,
    ) = calculate_probabilities(
        used_model,
        probabilities,
    )

    results = uploaded_df.copy()

    results[
        "Prediction"
    ] = [
        "Churn"
        if int(value) == 0
        else "Retain"
        for value in prediction
    ]

    results[
        "Churn Probability"
    ] = (
        churn_probability * 100
    ).round(2)

    results[
        "Retention Probability"
    ] = (
        retain_probability * 100
    ).round(2)

    results[
        "Confidence Score"
    ] = (
        confidence * 100
    ).round(2)

    results[
        "Risk Band"
    ] = [
        (
            "High"
            if value >= 0.70
            else
            "Moderate"
            if value >= 0.40
            else
            "Low"
        )
        for value in churn_probability
    ]

    return results


# ============================================================
# EXCEL DEFAULTS
# ============================================================

@st.cache_data
def load_excel_defaults():

    defaults = {}

    if not EXCEL_DEFAULT_PATH.exists():
        return defaults

    try:

        excel = pd.ExcelFile(
            EXCEL_DEFAULT_PATH
        )

        for sheet in excel.sheet_names:

            temp = pd.read_excel(
                EXCEL_DEFAULT_PATH,
                sheet_name=sheet,
            )

            if temp.empty:
                continue

            column_candidates = [
                "Column",
                "column",
                "Feature",
                "feature",
                "Field",
                "field",
            ]

            value_candidates = [
                "Default",
                "default",
                "Value",
                "value",
                "Default Value",
            ]

            column_name = next(
                (
                    c
                    for c in column_candidates
                    if c in temp.columns
                ),
                None,
            )

            value_name = next(
                (
                    c
                    for c in value_candidates
                    if c in temp.columns
                ),
                None,
            )

            if (
                column_name is not None
                and value_name is not None
            ):

                for _, row in temp.iterrows():

                    key = str(
                        row[column_name]
                    ).strip()

                    if key:

                        defaults[key] = (
                            row[value_name]
                        )

            else:

                row = temp.iloc[0]

                for column in temp.columns:

                    defaults[
                        str(column).strip()
                    ] = row[column]

        return defaults

    except Exception as exc:

        logger.warning(
            "Unable to load Excel defaults: %s",
            exc,
        )

        return defaults


EXCEL_DEFAULTS = load_excel_defaults()


# ============================================================
# DEFAULT VALUE
# ============================================================

def get_default(
    column,
    fallback,
):

    value = EXCEL_DEFAULTS.get(
        column
    )

    if value is None:
        return fallback

    try:

        if pd.isna(value):
            return fallback

    except Exception:
        pass

    return value


# ============================================================
# SAFE OPTION MATCHING
# ============================================================

def find_matching_option(
    options,
    default,
):

    if not options:
        return 0

    default_string = str(
        default
    ).strip().lower()

    for index, option in enumerate(
        options
    ):

        if (
            str(option)
            .strip()
            .lower()
            == default_string
        ):
            return index

    return 0


# ============================================================
# UI FIELD
# ============================================================

def render_field(
    column,
    model,
    key_prefix,
):

    label = frontend_label(
        column
    )

    key = (
        f"{key_prefix}_{column}"
    )

    # ========================================================
    # MODEL-LEARNED CATEGORIES FIRST
    # ========================================================

    categories = (
        get_encoder_categories(
            model,
            column,
        )
    )

    if categories:

        options = []

        for value in categories:

            if pd.isna(value):
                continue

            if value not in options:

                options.append(value)

        if options:

            default = get_default(
                column,
                options[0],
            )

            selected_index = (
                find_matching_option(
                    options,
                    default,
                )
            )

            return st.selectbox(
                label,
                options,
                index=selected_index,
                key=key,
            )

    # ========================================================
    # CITY
    # ========================================================

    if column == "City":

        return st.text_input(
            label,
            value=str(
                get_default(
                    column,
                    "Pune",
                )
            ),
            key=key,
        )

    # ========================================================
    # SENIOR CITIZEN
    # ========================================================

    if column == "Senior_Citizen":

        value = get_default(
            column,
            0,
        )

        try:
            value = int(
                float(value)
            )
        except Exception:
            value = 0

        selected = st.selectbox(
            label,
            [
                "No",
                "Yes",
            ],
            index=(
                1
                if value == 1
                else 0
            ),
            key=key,
        )

        return (
            1
            if selected == "Yes"
            else 0
        )

    # ========================================================
    # YES / NO
    # ========================================================

    yes_no_columns = {
        "Partner",
        "Dependents",
        "Phone_Service",
        "Paperless_Billing",
    }

    if column in yes_no_columns:

        default = str(
            get_default(
                column,
                "No",
            )
        )

        index = (
            1
            if default.lower() == "yes"
            else 0
        )

        return st.selectbox(
            label,
            [
                "No",
                "Yes",
            ],
            index=index,
            key=key,
        )

    # ========================================================
    # MULTIPLE LINES
    # ========================================================

    if column == "Multiple_Lines":

        options = [
            "No",
            "Yes",
            "No phone service",
        ]

        default = str(
            get_default(
                column,
                options[0],
            )
        )

        index = find_matching_option(
            options,
            default,
        )

        return st.selectbox(
            label,
            options,
            index=index,
            key=key,
        )

    # ========================================================
    # INTERNET SERVICE
    # ========================================================

    if column == "Internet_Service":

        options = [
            "DSL",
            "Fiber optic",
            "No",
        ]

        default = str(
            get_default(
                column,
                options[0],
            )
        )

        index = find_matching_option(
            options,
            default,
        )

        return st.selectbox(
            label,
            options,
            index=index,
            key=key,
        )

    # ========================================================
    # INTERNET DEPENDENT SERVICES
    # ========================================================

    internet_columns = {
        "Online_Security",
        "Online_Backup",
        "Device_Protection",
        "Tech_Support",
        "Streaming_TV",
    }

    if column in internet_columns:

        options = [
            "No",
            "Yes",
            "No internet service",
        ]

        default = str(
            get_default(
                column,
                options[0],
            )
        )

        index = find_matching_option(
            options,
            default,
        )

        return st.selectbox(
            label,
            options,
            index=index,
            key=key,
        )

    # ========================================================
    # PAYMENT
    # ========================================================

    if column == "Payment_Method":

        options = [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ]

        default = str(
            get_default(
                column,
                options[0],
            )
        )

        index = find_matching_option(
            options,
            default,
        )

        return st.selectbox(
            label,
            options,
            index=index,
            key=key,
        )

    # ========================================================
    # CONTRACT
    # ========================================================

    if column == "Contract":

        options = [
            "Month-to-month",
            "One year",
            "Two year",
        ]

        default = str(
            get_default(
                column,
                options[0],
            )
        )

        index = find_matching_option(
            options,
            default,
        )

        return st.selectbox(
            label,
            options,
            index=index,
            key=key,
        )

    # ========================================================
    # GENDER
    # ========================================================

    if column == "Gender":

        options = [
            "Female",
            "Male",
        ]

        default = str(
            get_default(
                column,
                options[0],
            )
        )

        index = find_matching_option(
            options,
            default,
        )

        return st.selectbox(
            label,
            options,
            index=index,
            key=key,
        )

    # ========================================================
    # TENURE
    # ========================================================

    if column == "Tenure_Months":

        default = get_default(
            column,
            12,
        )

        try:
            default = int(
                float(default)
            )
        except Exception:
            default = 12

        default = max(
            0,
            min(
                120,
                default,
            ),
        )

        return st.number_input(
            label,
            min_value=0,
            max_value=120,
            value=default,
            step=1,
            key=key,
        )

    # ========================================================
    # MONTHLY CHARGES
    # ========================================================

    if column == "Monthly_Charges":

        default = get_default(
            column,
            70.0,
        )

        try:
            default = float(
                default
            )
        except Exception:
            default = 70.0

        default = max(
            0.0,
            min(
                10000.0,
                default,
            ),
        )

        return st.number_input(
            label,
            min_value=0.0,
            max_value=10000.0,
            value=default,
            step=1.0,
            key=key,
        )

    # ========================================================
    # CHURN SCORE
    # ========================================================

    if column == "Churn_Score":

        default = get_default(
            column,
            50,
        )

        try:
            default = int(
                float(default)
            )
        except Exception:
            default = 50

        default = max(
            0,
            min(
                100,
                default,
            ),
        )

        return st.number_input(
            label,
            min_value=0,
            max_value=100,
            value=default,
            step=1,
            key=key,
        )

    # ========================================================
    # GENERIC NUMERIC
    # ========================================================

    numeric_columns = (
        get_numeric_columns(
            model
        )
    )

    if column in numeric_columns:

        default = get_default(
            column,
            0.0,
        )

        try:
            default = float(
                default
            )
        except Exception:
            default = 0.0

        return st.number_input(
            label,
            value=default,
            key=key,
        )

    # ========================================================
    # GENERIC TEXT
    # ========================================================

    return st.text_input(
        label,
        value=str(
            get_default(
                column,
                "",
            )
        ),
        key=key,
    )


# ============================================================
# RISK GAUGE
# ============================================================

def create_gauge(
    probability,
):

    percentage = (
        probability * 100
    )

    if percentage >= 70:
        color = "#e11d48"
    elif percentage >= 40:
        color = "#d97706"
    else:
        color = "#059669"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=percentage,
            number={
                "suffix": "%",
                "font": {
                    "size": 42
                },
            },
            title={
                "text": "Churn Probability"
            },
            gauge={
                "axis": {
                    "range": [
                        0,
                        100,
                    ]
                },
                "bar": {
                    "color": color
                },
                "steps": [
                    {
                        "range": [
                            0,
                            40,
                        ],
                        "color": "#d1fae5",
                    },
                    {
                        "range": [
                            40,
                            70,
                        ],
                        "color": "#fef3c7",
                    },
                    {
                        "range": [
                            70,
                            100,
                        ],
                        "color": "#ffe4e6",
                    },
                ],
            },
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=10,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# ============================================================
# MODEL ERROR
# ============================================================

def show_model_error(
    error,
    model,
):

    st.error(
        "❌ Prediction could not be completed."
    )

    st.markdown(
        """
### Model preprocessing rejected the supplied data

The application attempted to use the saved
machine-learning pipeline without manually encoding
the customer data.

A compatibility-safe TargetEncoder retry was also
performed.
"""
    )

    encoder = find_target_encoder(
        model
    )

    if encoder is not None:

        st.warning(
            """
A **TargetEncoder** was detected inside the
saved model.

The error indicates that sklearn encountered
different Python data types inside a categorical
feature, typically a mixture such as:

`string + float`

The rewritten prediction engine automatically
normalizes these categories where possible.
"""
        )

    with st.expander(
        "🔎 Prediction Exception",
        expanded=True,
    ):

        st.exception(
            error
        )

    with st.expander(
        "🧪 Runtime Information"
    ):

        st.code(
            "\n".join(
                [
                    f"Python: {sys.version}",
                    (
                        "Model: "
                        f"{model.__class__.__name__}"
                    ),
                ]
            )
        )

        try:

            import sklearn

            st.write(
                "scikit-learn:",
                sklearn.__version__,
            )

        except Exception:
            pass

        try:

            import imblearn

            st.write(
                "imbalanced-learn:",
                imblearn.__version__,
            )

        except Exception:
            pass

    if encoder is not None:

        with st.expander(
            "TargetEncoder Diagnostics"
        ):

            features = getattr(
                encoder,
                "feature_names_in_",
                None,
            )

            categories = getattr(
                encoder,
                "categories_",
                None,
            )

            if features is not None:

                for i, values in enumerate(
                    categories or []
                ):

                    if i >= len(features):
                        break

                    feature = features[i]

                    type_names = sorted(
                        {
                            type(value).__name__
                            for value in values
                            if not pd.isna(value)
                        }
                    )

                    st.write(
                        f"**{frontend_label(feature)}**"
                    )

                    st.write(
                        "Learned types:",
                        type_names,
                    )

                    if len(type_names) > 1:

                        st.error(
                            "⚠ Mixed category types detected."
                        )

                    st.write(
                        [
                            str(value)
                            for value in values
                        ]
                    )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_churn_model(
        str(MODEL_PATH)
    )

except Exception as exc:

    st.error(
        "Unable to load the churn model."
    )

    st.exception(
        exc
    )

    st.stop()


# ============================================================
# MODEL SCHEMA
# ============================================================

MODEL_FEATURES = (
    get_model_features(
        model
    )
)

CATEGORICAL_COLUMNS = (
    get_categorical_columns(
        model
    )
)

NUMERIC_COLUMNS = (
    get_numeric_columns(
        model
    )
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🚀 RetainIQ"
    )

    st.caption(
        "Customer Churn Intelligence Platform"
    )

    st.divider()

    st.markdown(
        """
### Prediction Workflow

**01** Customer information

**02** Data validation

**03** Model preprocessing

**04** Churn probability

**05** Retention action
"""
    )

    st.divider()

    st.success(
        "Prediction Engine Ready"
    )

    st.caption(
        f"{len(MODEL_FEATURES)} model features"
    )

    st.caption(
        f"{len(CATEGORICAL_COLUMNS)} categorical"
    )

    st.caption(
        f"{len(NUMERIC_COLUMNS)} numeric"
    )

    st.divider()

    with st.expander(
        "Model Features"
    ):

        for column in MODEL_FEATURES:

            st.write(
                f"• {frontend_label(column)}"
            )

    with st.expander(
        "Environment"
    ):

        try:

            import sklearn

            st.write(
                "scikit-learn",
                sklearn.__version__,
            )

        except Exception:
            pass

        try:

            import imblearn

            st.write(
                "imbalanced-learn",
                imblearn.__version__,
            )

        except Exception:
            pass


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">

<div class="eyebrow">
Customer Intelligence Platform
</div>

<h1>
Predict Churn. Protect Customers.
</h1>

<p>
Convert customer information into actionable
retention intelligence using your trained
machine-learning model.
</p>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# KPI
# ============================================================

a, b, c, d = st.columns(4)

a.metric(
    "Model Status",
    "Ready",
)

b.metric(
    "Model Features",
    len(MODEL_FEATURES),
)

c.metric(
    "Categorical Features",
    len(CATEGORICAL_COLUMNS),
)

d.metric(
    "Prediction",
    "Real Time",
)


# ============================================================
# TABS
# ============================================================

single_tab, bulk_tab, schema_tab = st.tabs(
    [
        "👤 Single Customer",
        "📂 Bulk Upload",
        "📊 Model Schema",
    ]
)


# ============================================================
# SINGLE CUSTOMER
# ============================================================

with single_tab:

    st.markdown(
        "### Customer Churn Assessment"
    )

    st.caption(
        "Enter customer information and run the trained prediction pipeline."
    )

    with st.form(
        "customer_prediction_form"
    ):

        customer = {}

        left_column, right_column = st.columns(
            2,
            gap="large",
        )

        midpoint = (
            len(MODEL_FEATURES) + 1
        ) // 2

        with left_column:

            st.markdown(
                "#### 👤 Customer Profile"
            )

            for column in MODEL_FEATURES[
                :midpoint
            ]:

                customer[column] = (
                    render_field(
                        column,
                        model,
                        "left",
                    )
                )

        with right_column:

            st.markdown(
                "#### ⚙ Services & Account"
            )

            for column in MODEL_FEATURES[
                midpoint:
            ]:

                customer[column] = (
                    render_field(
                        column,
                        model,
                        "right",
                    )
                )

        submitted = (
            st.form_submit_button(
                "🚀 Run Churn Assessment",
                use_container_width=True,
            )
        )

    # ========================================================
    # RESULT
    # ========================================================

    if submitted:

        try:

            with st.spinner(
                "Analyzing customer risk..."
            ):

                result = predict_single(
                    model,
                    customer,
                    MODEL_FEATURES,
                )

            prediction = (
                result["prediction"]
            )

            churn_probability = (
                result[
                    "churn_probability"
                ]
            )

            retain_probability = (
                result[
                    "retain_probability"
                ]
            )

            confidence = (
                result[
                    "confidence"
                ]
            )

            churn_pct = (
                churn_probability * 100
            )

            retain_pct = (
                retain_probability * 100
            )

            confidence_pct = (
                confidence * 100
            )

            is_churn = (
                prediction == 0
            )

            st.divider()

            st.markdown(
                "### Assessment Result"
            )

            c1, c2, c3 = st.columns(
                3
            )

            with c1:

                css_class = (
                    "risk-card"
                    if is_churn
                    else "safe-card"
                )

                prediction_text = (
                    "CHURN"
                    if is_churn
                    else "RETAIN"
                )

                st.markdown(
                    f"""
<div class="{css_class}">

<div class="eyebrow">
Prediction
</div>

<div class="big-value">
{prediction_text}
</div>

<div class="small-text">
Model predicted customer outcome
</div>

</div>
""",
                    unsafe_allow_html=True,
                )

            with c2:

                st.markdown(
                    f"""
<div class="card">

<div class="eyebrow">
Churn Probability
</div>

<div class="big-value">
{churn_pct:.1f}%
</div>

<div class="small-text">
Probability of customer churn
</div>

</div>
""",
                    unsafe_allow_html=True,
                )

            with c3:

                st.markdown(
                    f"""
<div class="card">

<div class="eyebrow">
Confidence
</div>

<div class="big-value">
{confidence_pct:.1f}%
</div>

<div class="small-text">
Model confidence
</div>

</div>
""",
                    unsafe_allow_html=True,
                )

            gauge_col, probability_col = (
                st.columns(2)
            )

            with gauge_col:

                st.plotly_chart(
                    create_gauge(
                        churn_probability
                    ),
                    use_container_width=True,
                    config={
                        "displayModeBar": False
                    },
                )

            with probability_col:

                st.markdown(
                    "### Probability Breakdown"
                )

                st.metric(
                    "Churn Probability",
                    f"{churn_pct:.1f}%",
                )

                st.progress(
                    float(
                        churn_probability
                    )
                )

                st.metric(
                    "Retention Probability",
                    f"{retain_pct:.1f}%",
                )

                st.progress(
                    float(
                        retain_probability
                    )
                )

                if churn_probability >= 0.70:

                    st.error(
                        "🔴 High churn risk — priority retention action recommended."
                    )

                elif churn_probability >= 0.40:

                    st.warning(
                        "🟠 Moderate churn risk — proactive engagement recommended."
                    )

                else:

                    st.success(
                        "🟢 Low churn risk — customer currently appears stable."
                    )

        except Exception as exc:

            show_model_error(
                exc,
                model,
            )


# ============================================================
# BULK UPLOAD
# ============================================================

with bulk_tab:

    st.markdown(
        "### Bulk Customer Prediction"
    )

    st.caption(
        "Upload a CSV containing the model's original feature names."
    )

    # --------------------------------------------------------
    # SAMPLE DATA
    # --------------------------------------------------------

    sample_row = {}

    for column in MODEL_FEATURES:

        categories = (
            get_encoder_categories(
                model,
                column,
            )
        )

        if categories:

            usable = [
                value
                for value in categories
                if not pd.isna(value)
            ]

            if usable:

                sample_row[column] = (
                    usable[0]
                )

                continue

        # ----------------------------------------------------
        # Fallback defaults
        # ----------------------------------------------------

        if column == "City":
            value = "Pune"

        elif column == "Gender":
            value = "Female"

        elif column == "Senior_Citizen":
            value = 0

        elif column in [
            "Partner",
            "Phone_Service",
        ]:
            value = "Yes"

        elif column == "Dependents":
            value = "No"

        elif column == "Multiple_Lines":
            value = "No"

        elif column == "Internet_Service":
            value = "Fiber optic"

        elif column in [
            "Online_Security",
            "Online_Backup",
            "Device_Protection",
            "Tech_Support",
            "Streaming_TV",
        ]:
            value = "No"

        elif column == "Paperless_Billing":
            value = "Yes"

        elif column == "Payment_Method":
            value = "Electronic check"

        elif column == "Contract":
            value = "Month-to-month"

        elif column == "Tenure_Months":
            value = 12

        elif column == "Monthly_Charges":
            value = 70.0

        elif column == "Churn_Score":
            value = 50

        else:
            value = ""

        sample_row[column] = value

    sample_df = pd.DataFrame(
        [sample_row]
    )

    st.download_button(
        "⬇ Download Sample CSV",
        data=sample_df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            "customer_churn_sample.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"],
    )

    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(
                uploaded_file
            )

            uploaded_df.columns = [
                str(c).strip()
                for c in uploaded_df.columns
            ]

            if uploaded_df.empty:

                st.error(
                    "The uploaded CSV contains no customer records."
                )

            else:

                st.success(
                    f"{len(uploaded_df):,} customer records loaded."
                )

                missing = [
                    column
                    for column in MODEL_FEATURES
                    if column not in uploaded_df.columns
                ]

                if missing:

                    st.error(
                        "The uploaded file is missing required model features."
                    )

                    for column in missing:

                        st.write(
                            f"• {frontend_label(column)}"
                        )

                else:

                    st.markdown(
                        "### Data Preview"
                    )

                    preview = (
                        uploaded_df
                        .head(10)
                        .copy()
                    )

                    preview.columns = [
                        frontend_label(
                            column
                        )
                        for column in preview.columns
                    ]

                    st.dataframe(
                        preview,
                        use_container_width=True,
                        hide_index=True,
                    )

                    if st.button(
                        "🚀 Run Bulk Prediction",
                        use_container_width=True,
                    ):

                        try:

                            with st.spinner(
                                "Running bulk churn assessment..."
                            ):

                                results = (
                                    predict_bulk(
                                        model,
                                        uploaded_df,
                                        MODEL_FEATURES,
                                    )
                                )

                            st.session_state[
                                "prediction_results"
                            ] = results

                            st.success(
                                "Bulk prediction completed successfully."
                            )

                        except Exception as exc:

                            show_model_error(
                                exc,
                                model,
                            )

        except Exception as exc:

            st.error(
                "Unable to read the uploaded CSV."
            )

            st.exception(
                exc
            )

    # ========================================================
    # RESULTS
    # ========================================================

    if (
        "prediction_results"
        in st.session_state
    ):

        results = st.session_state[
            "prediction_results"
        ]

        st.divider()

        total = len(
            results
        )

        churn_count = int(
            (
                results[
                    "Prediction"
                ]
                == "Churn"
            ).sum()
        )

        retain_count = (
            total
            - churn_count
        )

        churn_percentage = (
            churn_count
            / total
            * 100
            if total > 0
            else 0
        )

        average_confidence = (
            results[
                "Confidence Score"
            ].mean()
        )

        st.markdown(
            "### Portfolio Overview"
        )

        k1, k2, k3, k4, k5 = (
            st.columns(5)
        )

        k1.metric(
            "Customers",
            f"{total:,}",
        )

        k2.metric(
            "Predicted Churn",
            f"{churn_count:,}",
        )

        k3.metric(
            "Predicted Retain",
            f"{retain_count:,}",
        )

        k4.metric(
            "Churn %",
            f"{churn_percentage:.1f}%",
        )

        k5.metric(
            "Avg Confidence",
            f"{average_confidence:.1f}%",
        )

        # ----------------------------------------------------
        # CHARTS
        # ----------------------------------------------------

        chart1, chart2 = (
            st.columns(2)
        )

        with chart1:

            data = (
                results[
                    "Risk Band"
                ]
                .value_counts()
                .reset_index()
            )

            data.columns = [
                "Risk Band",
                "Customers",
            ]

            fig = px.bar(
                data,
                x="Risk Band",
                y="Customers",
                color="Risk Band",
                title=(
                    "Customer Risk Distribution"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with chart2:

            data = (
                results[
                    "Prediction"
                ]
                .value_counts()
                .reset_index()
            )

            data.columns = [
                "Prediction",
                "Customers",
            ]

            fig = px.pie(
                data,
                names="Prediction",
                values="Customers",
                hole=0.55,
                title="Churn vs Retain",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        # ----------------------------------------------------
        # RESULT TABLE
        # ----------------------------------------------------

        st.markdown(
            "### Customer-Level Results"
        )

        display_df = results.copy()

        display_df.columns = [
            frontend_label(
                column
            )
            for column in display_df.columns
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇ Download Prediction Results",
            data=results.to_csv(
                index=False
            ).encode("utf-8"),
            file_name=(
                "customer_churn_predictions.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# MODEL SCHEMA
# ============================================================

with schema_tab:

    st.markdown(
        "### Model Feature Schema"
    )

    schema = []

    for column in MODEL_FEATURES:

        if column in CATEGORICAL_COLUMNS:
            data_type = "Categorical"

        elif column in NUMERIC_COLUMNS:
            data_type = "Numeric"

        else:
            data_type = "Detected"

        schema.append(
            {
                "Model Column": column,
                "Frontend Label":
                    frontend_label(column),
                "Data Type": data_type,
            }
        )

    schema_df = pd.DataFrame(
        schema
    )

    st.dataframe(
        schema_df,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.markdown(
        "### TargetEncoder Inspection"
    )

    encoders = find_target_encoders(
        model
    )

    if not encoders:

        st.info(
            "No TargetEncoder was detected."
        )

    else:

        st.success(
            f"{len(encoders)} TargetEncoder component(s) detected."
        )

        for encoder_index, encoder in enumerate(
            encoders,
            start=1,
        ):

            st.markdown(
                f"#### TargetEncoder #{encoder_index}"
            )

            features = getattr(
                encoder,
                "feature_names_in_",
                None,
            )

            categories = getattr(
                encoder,
                "categories_",
                None,
            )

            if (
                features is not None
                and categories is not None
            ):

                for i, values in enumerate(
                    categories
                ):

                    if i >= len(features):
                        break

                    name = features[i]

                    type_names = sorted(
                        {
                            type(value).__name__
                            for value in values
                            if not pd.isna(value)
                        }
                    )

                    st.markdown(
                        f"**{frontend_label(name)}**"
                    )

                    st.write(
                        "Types:",
                        type_names,
                    )

                    st.write(
                        [
                            str(value)
                            for value in values
                        ]
                    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div style="
    text-align:center;
    padding:2rem 0 1rem;
    color:#64748b;
    font-size:.85rem;
">
RetainIQ • Customer Churn Intelligence
</div>
""",
    unsafe_allow_html=True,
)
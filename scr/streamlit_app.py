import sys
from pathlib import Path
import io

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. PATH RESOLUTION & CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "model"

EXCEL_PATH = DATA_DIR / "Telco_customer_churn.xlsx"
MODEL_PATH = MODELS_DIR / "Telco_customer_churn.pkl"

COLUMNS_TO_DROP = ['Lat_Long','Zip_Code','Churn_Reason','Count','Latitude','Longitude',
                   'Total_Charges','Churn_Value','CLTV','CustomerID','Country',
                   'State','Streaming_Movies']

TARGET_COL = 'Churn Label'

# Page Config with Corporate Styling
st.set_page_config(
    page_title="Customer Churn Analytics Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Vibrant CSS
st.markdown("""
    <style>
    .main { background-color: #FAFAFA; }
    .stMetric {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #6C5CE7;
    }
    .stButton>button {
        background: linear-gradient(135deg, #6C5CE7 0%, #a29bfe 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. CACHED RESOURCE LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_trained_model(model_file_path: Path):
    """Loads model pipeline artifact safely."""
    if not model_file_path.exists():
        raise FileNotFoundError(f"Model artifact not found at path: {model_file_path}")
    return joblib.load(model_file_path)


@st.cache_data
def load_reference_dataset(excel_file_path: Path) -> pd.DataFrame:
    """Loads reference Excel dataset and strips unnecessary/dropped columns."""
    if not excel_file_path.exists():
        raise FileNotFoundError(f"Reference file not found at path: {excel_file_path}")
    df = pd.read_excel(excel_file_path)
    
    # Clean target and dropped columns
    drop_list = [col for col in COLUMNS_TO_DROP + [TARGET_COL] if col in df.columns]
    if drop_list:
        df = df.drop(columns=drop_list)
        
    return df


# -----------------------------------------------------------------------------
# 3. UTILITY & VALIDATION FUNCTIONS
# -----------------------------------------------------------------------------
def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Removes COLUMNS_TO_DROP and TARGET_COL if present in input data."""
    drop_list = [c for c in COLUMNS_TO_DROP + [TARGET_COL] if c in df.columns]
    return df.drop(columns=drop_list) if drop_list else df


def validate_schema(uploaded_df: pd.DataFrame, expected_cols: list) -> tuple[bool, list]:
    """Validates if sanitized uploaded dataset contains all necessary feature columns."""
    df_clean = sanitize_dataframe(uploaded_df)
    missing_cols = [col for col in expected_cols if col not in df_clean.columns]
    return (len(missing_cols) == 0), missing_cols


def generate_sample_excel(ref_df: pd.DataFrame) -> bytes:
    """Generates sample template Excel bytes (excluding COLUMNS_TO_DROP)."""
    sample_df = ref_df.head(5).copy()
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        sample_df.to_excel(writer, index=False, sheet_name='Sample_Data')
    return output.getvalue()


# -----------------------------------------------------------------------------
# 4. MAIN APPLICATION
# -----------------------------------------------------------------------------
def main():
    st.title("⚡ Enterprise Churn Prediction Portal")
    st.caption("Real-time Machine Learning Inference Engine for Customer Retention")

    # Load Resources with Error Handling
    try:
        model = load_trained_model(MODEL_PATH)
        ref_df = load_reference_dataset(EXCEL_PATH)
        expected_features = ref_df.columns.tolist()
    except Exception as e:
        st.error(f"🚨 Initialization Error: Failure loading critical assets. Details: {str(e)}")
        st.stop()

    # Sidebar Options
    st.sidebar.image("https://img.icons8.com/fluency/96/analytics.png", width=70)
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Select Prediction Mode",
        ["Single Customer Prediction", "Bulk Customer Upload"]
    )

    # -------------------------------------------------------------------------
    # MODE 1: SINGLE CUSTOMER PREDICTION
    # -------------------------------------------------------------------------
    if app_mode == "Single Customer Prediction":
        st.subheader("👤 Single Customer Intelligence")
        st.info("Input customer properties below. Form is pre-populated with enterprise reference defaults.")

        input_data = {}
        
        with st.form("single_prediction_form"):
            cols = st.columns(3)
            for idx, feature in enumerate(expected_features):
                col = cols[idx % 3]
                series = ref_df[feature]

                # Coerce to numeric for type safety
                series_numeric = pd.to_numeric(series, errors='coerce')
                
                is_numeric_type = pd.api.types.is_numeric_dtype(series_numeric) and series_numeric.notna().any()
                is_explicit_categorical = isinstance(series.dtype, (pd.CategoricalDtype, object)) or str(series.dtype).startswith('string')

                if is_numeric_type and not is_explicit_categorical:
                    clean_series = series_numeric.dropna()
                    
                    default_val = float(clean_series.median())
                    min_val = float(clean_series.min())
                    max_val = float(clean_series.max())
                    
                    if min_val == max_val:
                        max_val += 1.0

                    input_data[feature] = col.number_input(
                        label=f"{feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val
                    )
                else:
                    unique_vals = [str(val) for val in series.dropna().unique().tolist()]
                    input_data[feature] = col.selectbox(
                        label=f"{feature}",
                        options=unique_vals if unique_vals else ["N/A"],
                        index=0
                    )

            submit_btn = st.form_submit_button("Run Risk Analysis")

        if submit_btn:
            try:
                single_df = pd.DataFrame([input_data])
                
                # Execute Model Pipeline
                pred_class = model.predict(single_df)[0]
                pred_proba = model.predict_proba(single_df)[0]
                churn_prob = float(pred_proba[1])
                confidence = float(np.max(pred_proba))

                st.markdown("---")
                st.subheader("📊 Diagnostic Summary")
                
                m1, m2, m3 = st.columns(3)
                status_text = "High Churn Risk" if pred_class == 1 else "Low Risk / Retained"
                
                m1.metric("Predicted Status", status_text)
                m2.metric("Churn Probability", f"{churn_prob * 100:.2f}%")
                m3.metric("Model Confidence Ratio", f"{confidence * 100:.2f}%")

                # Visual Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=churn_prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Churn Probability (%)", 'font': {'size': 18}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#FF7675" if churn_prob > 0.5 else "#55E6C1"},
                        'steps': [
                            {'range': [0, 40], 'color': "#55E6C1"},
                            {'range': [40, 70], 'color': "#FFEAA7"},
                            {'range': [70, 100], 'color': "#FF7675"}
                        ],
                    }
                ))
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Execution Error during single prediction processing: {str(e)}")

    # -------------------------------------------------------------------------
    # MODE 2: BULK CUSTOMER UPLOAD
    # -------------------------------------------------------------------------
    elif app_mode == "Bulk Customer Upload":
        st.subheader("📁 Bulk Customer Processing & Portfolio Analytics")

        # Download Sample Template
        sample_bytes = generate_sample_excel(ref_df)
        st.download_button(
            label="📥 Download Clean Bulk Template (.xlsx)",
            data=sample_bytes,
            file_name="churn_prediction_bulk_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        uploaded_file = st.file_uploader(
            "Upload Batch Dataset (CSV or XLSX)",
            type=["csv", "xlsx"]
        )

        if uploaded_file is not None:
            try:
                # File Reader
                if uploaded_file.name.endswith('.csv'):
                    raw_bulk_df = pd.read_csv(uploaded_file)
                else:
                    raw_bulk_df = pd.read_excel(uploaded_file)

                st.write(f"**Loaded File:** `{uploaded_file.name}` ({len(raw_bulk_df)} records)")

                # Validate Schema
                is_valid, missing_columns = validate_schema(raw_bulk_df, expected_features)

                if not is_valid:
                    st.error("🚨 Data Schema Validation Failed!")
                    st.warning(
                        f"The uploaded file is missing the following required feature columns: "
                        f"`{', '.join(missing_columns)}`. "
                        "Please align your dataset with the downloadable sample template."
                    )
                    st.stop()

                # Drop COLUMNS_TO_DROP before passing into model pipeline
                clean_bulk_df = sanitize_dataframe(raw_bulk_df)[expected_features]

                if st.button("Execute Batch Prediction Engine"):
                    with st.spinner("Processing records through Machine Learning Pipeline..."):
                        preds = model.predict(clean_bulk_df)
                        probas = model.predict_proba(clean_bulk_df)

                        churn_probs = probas[:, 1]
                        confidences = np.max(probas, axis=1)

                        # Build output dataframe
                        results_df = raw_bulk_df.copy()
                        results_df['Predicted_Churn'] = preds
                        results_df['Churn_Probability'] = churn_probs
                        results_df['Confidence_Ratio'] = confidences

                    st.success("Batch Inference Completed Successfully!")

                    # Executive Summary Metrics
                    total_customers = len(results_df)
                    churn_count = int(np.sum(preds))
                    churn_percentage = (churn_count / total_customers) * 100
                    mean_confidence = float(np.mean(confidences)) * 100

                    st.markdown("### 📈 Portfolio Metrics")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Processed", f"{total_customers:,}")
                    c2.metric("Predicted Churn Volume", f"{churn_count:,}")
                    c3.metric("Portfolio Churn Rate", f"{churn_percentage:.2f}%")
                    c4.metric("Avg Confidence Ratio", f"{mean_confidence:.2f}%")

                    # Visual Charts
                    col_chart1, col_chart2 = st.columns(2)

                    with col_chart1:
                        fig_pie = px.pie(
                            names=['Retained', 'Churn Risk'],
                            values=[total_customers - churn_count, churn_count],
                            title="Customer Risk Distribution",
                            color_discrete_sequence=['#55E6C1', '#FF7675'],
                            hole=0.4
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col_chart2:
                        fig_hist = px.histogram(
                            results_df,
                            x='Churn_Probability',
                            nbins=20,
                            title="Churn Probability Distribution",
                            color_discrete_sequence=['#6C5CE7']
                        )
                        fig_hist.update_layout(xaxis_title="Churn Probability", yaxis_title="Count")
                        st.plotly_chart(fig_hist, use_container_width=True)

                    # Export Results
                    st.markdown("### 📋 Detailed Predictions Output")
                    st.dataframe(results_df.head(100), use_container_width=True)

                    csv_data = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Export Predictions CSV",
                        data=csv_data,
                        file_name="churn_predictions_export.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"🚨 Processing Error: An exception occurred while evaluating file. Details: {str(e)}")


if __name__ == "__main__":
    main()
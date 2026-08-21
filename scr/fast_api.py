# ============================================================
# RETAINIQ - CUSTOMER CHURN FASTAPI SERVICE
# ============================================================

from pathlib import Path
import logging
import traceback

import joblib
import uvicorn
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# APPLICATION PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR.parent
    / "model"
    / "Telco_customer_churn.pkl"
)

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "retainiq_api.log"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_FILE,
            encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("retainiq_api")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "RetainIQ API for predicting customer churn "
        "and calculating churn probability and "
        "prediction confidence."
    ),
    version="1.0.0",
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    logger.info(
        "Churn model loaded successfully from %s",
        MODEL_PATH
    )

except Exception as exc:

    logger.exception(
        "Unable to load churn model."
    )

    model = None


# ============================================================
# INPUT MODEL
# ============================================================

class CustomerChurn(BaseModel):

    City: str = Field(..., example="Pune")

    Gender: str = Field(
        ...,
        example="Female"
    )

    Senior_Citizen: str = Field(
        ...,
        example="No"
    )

    Partner: str = Field(
        ...,
        example="Yes"
    )

    Dependents: str = Field(
        ...,
        example="No"
    )

    Phone_Service: str = Field(
        ...,
        example="Yes"
    )

    Multiple_Lines: str = Field(
        ...,
        example="No"
    )

    Internet_Service: str = Field(
        ...,
        example="Fiber optic"
    )

    Online_Security: str = Field(
        ...,
        example="No"
    )

    Online_Backup: str = Field(
        ...,
        example="No"
    )

    Device_Protection: str = Field(
        ...,
        example="No"
    )

    Tech_Support: str = Field(
        ...,
        example="No"
    )

    Streaming_TV: str = Field(
        ...,
        example="No"
    )

    Contract: str = Field(
        ...,
        example="Month-to-month"
    )

    Paperless_Billing: str = Field(
        ...,
        example="Yes"
    )

    Payment_Method: str = Field(
        ...,
        example="Electronic check"
    )

    Tenure_Months: float = Field(
        ...,
        ge=0,
        example=12
    )

    Monthly_Charges: float = Field(
        ...,
        ge=0,
        example=70.50
    )

    Churn_Score: float = Field(
        ...,
        ge=0,
        le=100,
        example=50
    )


# ============================================================
# NORMALIZE INPUT
# ============================================================

def normalize_customer(data: CustomerChurn) -> pd.DataFrame:

    values = data.model_dump()

    # --------------------------------------------------------
    # String fields
    # --------------------------------------------------------

    categorical_columns = [
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

    for column in categorical_columns:

        value = values.get(column)

        if value is None:
            continue

        values[column] = str(value).strip()

    # --------------------------------------------------------
    # Senior Citizen normalization
    #
    # The trained model expects categorical values.
    # Accept:
    #
    # No / Yes
    # 0 / 1
    # True / False
    # --------------------------------------------------------

    senior_value = str(
        values["Senior_Citizen"]
    ).strip().lower()

    senior_mapping = {
        "0": "No",
        "0.0": "No",
        "no": "No",
        "false": "No",
        "n": "No",

        "1": "Yes",
        "1.0": "Yes",
        "yes": "Yes",
        "true": "Yes",
        "y": "Yes",
    }

    if senior_value in senior_mapping:

        values["Senior_Citizen"] = (
            senior_mapping[senior_value]
        )

    else:

        raise ValueError(
            "Senior_Citizen must be "
            "'Yes' or 'No'."
        )

    # --------------------------------------------------------
    # Numeric fields
    # --------------------------------------------------------

    values["Tenure_Months"] = float(
        values["Tenure_Months"]
    )

    values["Monthly_Charges"] = float(
        values["Monthly_Charges"]
    )

    values["Churn_Score"] = float(
        values["Churn_Score"]
    )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame([values])

    # Explicit model feature order
    model_features = [
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

    df = df[model_features]

    return df


# ============================================================
# PREDICTION
# ============================================================

def perform_prediction(
    input_df: pd.DataFrame
):

    if model is None:

        raise RuntimeError(
            "The churn prediction model "
            "could not be loaded."
        )

    logger.info(
        "Starting customer churn prediction."
    )

    try:

        prediction = model.predict(
            input_df
        )

        prediction_value = prediction[0]

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probability = None

        if hasattr(
            model,
            "predict_proba"
        ):

            probability = model.predict_proba(
                input_df
            )[0]

        # ----------------------------------------------------
        # Determine model classes
        # ----------------------------------------------------

        classes = getattr(
            model,
            "classes_",
            None
        )

        if classes is None:

            classes = [0, 1]

        classes = list(classes)

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Standard Telco churn models commonly use:
        #
        # 0 = No churn
        # 1 = Churn
        #
        # We therefore interpret 1 as churn.
        # ----------------------------------------------------

        if str(prediction_value).lower() in {
            "1",
            "1.0",
            "yes",
            "churn",
            "true",
        }:

            churn_prediction = True

        else:

            churn_prediction = False

        # ----------------------------------------------------
        # Calculate probabilities
        # ----------------------------------------------------

        if probability is not None:

            churn_probability = 0.0
            retain_probability = 0.0

            for index, class_value in enumerate(
                classes
            ):

                class_text = str(
                    class_value
                ).lower()

                if class_text in {
                    "1",
                    "1.0",
                    "yes",
                    "churn",
                    "true",
                }:

                    churn_probability = float(
                        probability[index]
                    )

                else:

                    retain_probability = float(
                        probability[index]
                    )

            confidence = max(
                churn_probability,
                retain_probability
            )

        else:

            if churn_prediction:

                churn_probability = 1.0
                retain_probability = 0.0

            else:

                churn_probability = 0.0
                retain_probability = 1.0

            confidence = 1.0

        # ----------------------------------------------------
        # Risk classification
        # ----------------------------------------------------

        if churn_probability >= 0.70:

            risk_band = "High"

            recommendation = (
                "Immediate retention action is recommended. "
                "The customer shows a high likelihood of churn. "
                "Consider personalized offers, service recovery, "
                "priority outreach, or an account review."
            )

        elif churn_probability >= 0.40:

            risk_band = "Moderate"

            recommendation = (
                "Proactive customer engagement is recommended. "
                "The customer shows a moderate level of churn risk. "
                "Consider targeted communication, service review, "
                "and relationship-building activities."
            )

        else:

            risk_band = "Low"

            recommendation = (
                "The customer currently shows relatively low "
                "churn risk. Continue normal engagement and "
                "monitor future changes in customer behaviour."
            )

        # ----------------------------------------------------
        # Customer message
        # ----------------------------------------------------

        if churn_prediction:

            message = (
                "Our assessment indicates that this customer "
                "has an elevated likelihood of leaving the service. "
                "The churn probability is "
                f"{churn_probability * 100:.1f}%, "
                "so proactive retention activity should be "
                "considered."
            )

        else:

            message = (
                "Our assessment indicates that this customer "
                "is currently more likely to remain with the service. "
                "The predicted retention probability is "
                f"{retain_probability * 100:.1f}%. "
                "Continue regular engagement and monitor "
                "future risk indicators."
            )

        logger.info(
            "Prediction completed successfully | "
            "prediction=%s | churn_probability=%.4f | "
            "confidence=%.4f",
            prediction_value,
            churn_probability,
            confidence,
        )

        return {
            "prediction": (
                "Churn"
                if churn_prediction
                else "Retain"
            ),

            "churn_probability": round(
                churn_probability,
                4
            ),

            "churn_probability_percent": round(
                churn_probability * 100,
                2
            ),

            "retention_probability": round(
                retain_probability,
                4
            ),

            "retention_probability_percent": round(
                retain_probability * 100,
                2
            ),

            "confidence_score": round(
                confidence,
                4
            ),

            "confidence_percent": round(
                confidence * 100,
                2
            ),

            "risk_band": risk_band,

            "message": message,

            "recommended_action": recommendation,
        }

    except Exception as exc:

        logger.exception(
            "Prediction failure: %s",
            exc
        )

        raise


# ============================================================
# HOME
# ============================================================

@app.get("/")
def first_pgm():

    return {
        "application": "RetainIQ",
        "service": "Customer Churn Prediction API",
        "status": (
            "ready"
            if model is not None
            else "model unavailable"
        ),
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    if model is None:

        return {
            "status": "unhealthy",
            "model_loaded": False,
        }

    return {
        "status": "healthy",
        "model_loaded": True,
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(data: CustomerChurn):

    try:

        # ----------------------------------------------------
        # Convert request to model DataFrame
        # ----------------------------------------------------

        input_data = normalize_customer(
            data
        )

        logger.info(
            "Received customer prediction request."
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        result = perform_prediction(
            input_data
        )

        return {
            "success": True,
            "result": result,
        }

    except ValueError as exc:

        logger.exception(
            "Invalid customer input: %s",
            exc
        )

        raise HTTPException(
            status_code=422,
            detail={
                "success": False,
                "message": (
                    "The customer information could "
                    "not be processed."
                ),
                "reason": str(exc),
            },
        )

    except Exception as exc:

        logger.exception(
            "Unexpected prediction error."
        )

        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": (
                    "The customer assessment could "
                    "not be completed."
                ),
                "reason": (
                    "An unexpected prediction error "
                    "occurred. The technical details "
                    "have been recorded in the application log."
                ),
            },
        )


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import io


# Define the input data model
class PredictionInput(BaseModel):
    air_temperature: float = Field(..., description="Air temperature in Kelvin")
    process_temperature: float = Field(..., description="Process temperature in Kelvin")
    rotational_speed: float = Field(..., description="Rotational speed in rpm")
    torque: float = Field(..., description="Torque in Nm")
    tool_wear: float = Field(..., description="Tool wear in minutes")
    product_quality: str = Field(..., description="Product quality (L/M/H)")

    class Config:
        json_schema_extra = {
            "example": {
                "air_temperature": 300.0,
                "process_temperature": 310.0,
                "rotational_speed": 1500,
                "torque": 40.0,
                "tool_wear": 100,
                "product_quality": "M",
            }
        }


app = FastAPI(title="Manufacturing Predictor API")

# Global variables
model = None
failure_type_model = None

# Expected column names
EXPECTED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Target",
    "Failure Type",
]


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Validate required columns
        missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing_cols)}. Available columns: {df.columns.tolist()}",
            )

        df.to_csv("manufacturing_data.csv", index=False)
        return {
            "message": "Data uploaded successfully",
            "rows": len(df),
            "columns": df.columns.tolist(),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/train")
async def train_model():
    try:
        # Load data
        df = pd.read_csv("manufacturing_data.csv")

        # Select feature columns
        feature_columns = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]

        # Verify all features exist
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required feature columns: {missing_features}. Available columns: {df.columns.tolist()}",
            )

        # Prepare features
        X = df[feature_columns].copy()

        # Add product quality as one-hot encoded features
        product_quality = df["Product ID"].str[0]
        quality_dummies = pd.get_dummies(product_quality, prefix="quality")
        X = pd.concat([X, quality_dummies], axis=1)

        # Prepare targets
        y_failure = df["Target"]
        y_type = df["Failure Type"]

        # Split data
        X_train, X_test, y_failure_train, y_failure_test, y_type_train, y_type_test = (
            train_test_split(X, y_failure, y_type, test_size=0.25, random_state=42)
        )

        # Train models
        global model, failure_type_model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_failure_train)

        failure_type_model = RandomForestClassifier(n_estimators=100, random_state=42)
        failure_type_model.fit(X_train, y_type_train)

        # Calculate metrics
        failure_pred = model.predict(X_test)
        type_pred = failure_type_model.predict(X_test)

        metrics = {
            "failure_prediction": {
                "accuracy": round(accuracy_score(y_failure_test, failure_pred), 3),
                "f1_score": round(f1_score(y_failure_test, failure_pred), 3),
            },
            "failure_type_prediction": {
                "accuracy": round(accuracy_score(y_type_test, type_pred), 3),
                "report": classification_report(
                    y_type_test, type_pred, output_dict=True
                ),
            },
        }

        # Save models
        joblib.dump(model, "failure_model.joblib")
        joblib.dump(failure_type_model, "failure_type_model.joblib")

        return metrics

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Load models if not in memory
        global model, failure_type_model
        if model is None or failure_type_model is None:
            try:
                model = joblib.load("failure_model.joblib")
                failure_type_model = joblib.load("failure_type_model.joblib")
            except FileNotFoundError:
                raise HTTPException(
                    status_code=400,
                    detail="Models not trained. Please train the models first.",
                )

        # Prepare input data with correct column names
        input_df = pd.DataFrame(
            [
                {
                    "Air temperature [K]": input_data.air_temperature,
                    "Process temperature [K]": input_data.process_temperature,
                    "Rotational speed [rpm]": input_data.rotational_speed,
                    "Torque [Nm]": input_data.torque,
                    "Tool wear [min]": input_data.tool_wear,
                }
            ]
        )

        # Add one-hot encoded product quality
        quality_cols = ["quality_H", "quality_L", "quality_M"]
        for col in quality_cols:
            input_df[col] = 0
        input_df[f"quality_{input_data.product_quality}"] = 1

        # Make predictions
        failure_pred = model.predict(input_df)[0]
        failure_prob = model.predict_proba(input_df)[0].max()

        if failure_pred:
            failure_type = failure_type_model.predict(input_df)[0]
            type_prob = failure_type_model.predict_proba(input_df)[0].max()
        else:
            failure_type = "NONE"
            type_prob = 1.0

        return {
            "machine_failure": bool(failure_pred),
            "failure_probability": round(float(failure_prob), 3),
            "failure_type": failure_type,
            "type_probability": round(float(type_prob), 3),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

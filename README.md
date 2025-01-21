# **Manufacturing Predictive Analysis API**

A FastAPI-based REST API for predicting machine failures in manufacturing operations using machine learning. The API provides endpoints for uploading manufacturing data, training a predictive model, and making failure predictions.

---

## **Table of Contents**

1. [Installation](#installation)  
2. [Running the API](#running-the-api)  
3. [API Endpoints](#api-endpoints)  
   - [1. Upload Data](#1-upload-data)  
   - [2. Train Model](#2-train-model)  
   - [3. Predict](#3-predict)  
4. [Sample Dataset](#sample-dataset)  
5. [Error Handling](#error-handling)  
6. [Testing](#testing)  
   - [Using Postman Collection](#using-postman-collection)  
7. [Model Details](#model-details)  
8. [Project Structure](#project-structure)  
9. [Contributing](#contributing)  
10. [License](#license)  

---

## **1. Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/govind516/Machine-Predictive-Analysis/
   cd Machine-Predictive-Analysis
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **Unix/MacOS:**
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **2. Running the API**

1. Start the server:
   ```bash
   python app.py
   ```
   or
   ```bash
   uvicorn app:app --reload
   ```

2. The API will be available at `http://localhost:8000`
3. Access the interactive API documentation at `http://localhost:8000/docs`

![Screenshot 2025-01-21 154002](https://github.com/user-attachments/assets/e960dd60-dc43-426e-a906-8a5c7526c674)

---

## **3. API Endpoints**

### **1. Upload Data**
- **Endpoint**: `POST /upload`
- **Purpose**: Upload manufacturing data CSV file
- **Required Columns**:
  - UDI
  - Product ID
  - Air temperature
  - Process temperature
  - Rotational speed
  - Torque
  - Tool wear
  - Machine failure

Example using curl:
```bash
curl -X POST -F "file=@data.csv" http://localhost:8000/upload
```

Example Response:
```json
{
    "message": "Data uploaded successfully",
    "rows": 10000,
    "columns": ["UDI", "Product ID", "Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear", "Machine failure"]
}
```

### **2. Train Model**
- **Endpoint**: `POST /train`
- **Purpose**: Train the machine learning model using uploaded data

Example using curl:
```bash
curl -X POST http://localhost:8000/train
```

Example Response:
```json
{
    "failure_prediction": {
        "accuracy": 0.985,
        "f1_score": 0.876
    },
    "failure_type_prediction": {
        "accuracy": 0.923,
        "report": "Classification report..."
    }
}
```

### **3. Predict**
- **Endpoint**: `POST /predict`
- **Purpose**: Make failure predictions for new data

Example Request:
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{
           "air_temperature": 300.0,
           "process_temperature": 310.0,
           "rotational_speed": 1500,
           "torque": 40.0,
           "tool_wear": 100,
           "product_quality": "M"
         }' \
     http://localhost:8000/predict
```

Example Response:
```json
{
    "machine_failure": false,
    "failure_probability": 0.123,
    "failure_type": "NONE",
    "type_probability": 1.0
}
```

---

## **4. Sample Dataset**

The API expects a CSV file with the following format:

```csv
UDI,Product ID,Air temperature,Process temperature,Rotational speed,Torque,Tool wear,Machine failure
1,M14860,298.1,308.6,1551,42.8,0,0
2,L47181,298.2,308.7,1408,46.3,3,0
...
```

You can find sample datasets at:
1. [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
2. [Kaggle - Predictive Maintenance Dataset](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)

---

## **5. Error Handling**

The API includes comprehensive error handling for:
- Missing or invalid columns in uploaded data
- Invalid input data formats
- Model not trained before prediction
- Server-side errors

Example Error Response:
```json
{
    "detail": "Missing required columns: Air temperature, Process temperature"
}
```

---

## **6. Testing**

### **Using Postman Collection**
1. Import the `Machine Predictive Analysis.postman_collection.json` file into Postman.
2. In the upload endpoint request, **update the CSV** file in body -> form-data to use the **sample CSV** from the data folder.
3. Run the collection endpoints in sequence:
   - Upload Data
   - Train Model
   - Predict

![image](https://github.com/user-attachments/assets/ce934b24-1229-438e-bafa-930ebbc0e72e)

---

## **7. Model Details**

The implementation uses:
- `RandomForestClassifier` for both failure prediction and failure type classification
- Feature engineering for product quality
- Specific failure mode detection based on domain rules
- Model persistence using `joblib`

---

## **8. Project Structure**

```
manufacturing-predictor/
├── app.py                                         # Main FastAPI application
├── requirements.txt                               # Python dependencies
├── README.md                                      # This file
├── Machine Predictive Analysis.postman_collection.json  # Postman collection for API testing
└── data/
    └── sample.csv                                 # Sample dataset
```

---

## **9. Contributing**

1. Fork the repository.
2. Create a feature branch.
3. Commit changes.
4. Push to the branch.
5. Create a Pull Request.

---

## **10. License**

This project is licensed under the MIT License - see the LICENSE file for details. 

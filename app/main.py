from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import joblib
from io import BytesIO
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from pydantic import BaseModel

class PredictionResponse(BaseModel):
    prediction: int
    probability_of_risk: float


app = FastAPI(title="Heart Attack Risk Predictor")

# Загружаем модель
model = joblib.load("models/heart_attack_model.pkl")

# Pydantic модель для JSON ввода
class PatientData(BaseModel):
    Age: float
    Cholesterol: float
    Heart_rate: float
    Diabetes: int
    Family_History: int
    Smoking: int
    Obesity: int
    Alcohol_Consumption: int
    Exercise_Hours_Per_Week: float
    Diet: float
    Previous_Heart_Problems: int
    Medication_Use: int
    Stress_Level: float
    Sedentary_Hours_Per_Day: float
    Income: float
    BMI: float
    Triglycerides: float
    Physical_Activity_Days_Per_Week: float
    Sleep_Hours_Per_Day: float
    Blood_sugar: float
    CK_MB: float
    Troponin: float
    Systolic_blood_pressure: float
    Diastolic_blood_pressure: float
    Gender: int

@app.post("/predict_json", response_model=PredictionResponse)
def predict_json(data: PatientData):
    input_df = pd.DataFrame([data.dict()])

    # Приведение названий колонок к виду, который ждёт модель
    column_name_mapping = {
        "Age": "Age",
        "Cholesterol": "Cholesterol",
        "Heart_rate": "Heart rate",
        "Diabetes": "Diabetes",
        "Family_History": "Family History",
        "Smoking": "Smoking",
        "Obesity": "Obesity",
        "Alcohol_Consumption": "Alcohol Consumption",
        "Exercise_Hours_Per_Week": "Exercise Hours Per Week",
        "Diet": "Diet",
        "Previous_Heart_Problems": "Previous Heart Problems",
        "Medication_Use": "Medication Use",
        "Stress_Level": "Stress Level",
        "Sedentary_Hours_Per_Day": "Sedentary Hours Per Day",
        "Income": "Income",
        "BMI": "BMI",
        "Triglycerides": "Triglycerides",
        "Physical_Activity_Days_Per_Week": "Physical Activity Days Per Week",
        "Sleep_Hours_Per_Day": "Sleep Hours Per Day",
        "Blood_sugar": "Blood sugar",
        "CK_MB": "CK-MB",
        "Troponin": "Troponin",
        "Systolic_blood_pressure": "Systolic blood pressure",
        "Diastolic_blood_pressure": "Diastolic blood pressure",
        "Gender": "Gender"
}

    input_df.rename(columns=column_name_mapping, inplace=True)

    prediction = int(model.predict(input_df)[0])
    probability = round(model.predict_proba(input_df)[0][1], 4)
    return {"prediction": prediction, "probability_of_risk": probability}


@app.post("/predict_csv")
def predict_csv(file: UploadFile = File(...)):
    content = file.file.read()
    df = pd.read_csv(BytesIO(content))

    ids = df['id']

    # Кодируем Gender
    if df['Gender'].dtype == object:
        le = LabelEncoder()
        df['Gender'] = le.fit_transform(df['Gender'])

    # Импутация пропусков
    imputer = SimpleImputer(strategy='mean')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    X = df.drop(columns=['id'])
    predictions = model.predict(X).astype(int)
    probabilities = model.predict_proba(X)[:, 1]

    result = pd.DataFrame({
        "id": ids.astype(int),
        "prediction": predictions,
        "probability_of_risk": probabilities
    })

    return result.to_dict(orient="records")

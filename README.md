# Heart Attack Risk Predictor

FastAPI-приложение с моделью CatBoost для предсказания риска сердечного приступа по данным пациента.

---

## 📌 Возможности:

- Приём **JSON-запросов** с данными пациента
- Приём **CSV файлов** для пакетного предсказания
- Возврат **предсказания и вероятности риска**
- Удобный **Swagger UI для тестирования**
- Профессиональная структура и документация

---

## 📂 Структура репозитория

```
heart-attack-risk-predictor/
├── app/
│   └── main.py                               # FastAPI приложение
├── models/
│   └── heart_attack_model.pkl                # Обученная модель CatBoost
├── notebooks/
│   ├── notebooke_тестирование_модели.ipynb   # Ноутбук с тестированием модели
│   ├── notebook_обучение_модели.ipynb        # Ноутбук с обучением модели
│   └──notebook_предобработка.ipynb           # Ноутбук с предобработкой данных
├── requirements.txt                          # Зависимости проекта
└── README.md                                 # Документация проекта

```

---

## 🚀 Установка и запуск

1️⃣ Клонируй репозиторий:

```bash
git clone https://github.com/Sergey1916/heart-attack-risk-predictor.git
cd heart-attack-risk-predictor
```

2️⃣ Установи зависимости:

```bash
pip install -r requirements.txt
```

3️⃣ Запусти FastAPI:

```bash
python -m uvicorn app.main:app --reload
```

4️⃣ Открой в браузере:

```
http://127.0.0.1:8000/docs
```

для удобного тестирования через **Swagger UI**.

---

## 🩺 Использование

### Предсказание через JSON

- Открой `/predict_json` в Swagger.
- Нажми **Try it out**.
- Введи данные пациента в JSON, например:

```json
{
  "Age": 0.5,
  "Cholesterol": 0.6,
  "Heart_rate": 0.4,
  "Diabetes": 0,
  "Family_History": 1,
  "Smoking": 0,
  "Obesity": 0,
  "Alcohol_Consumption": 1,
  "Exercise_Hours_Per_Week": 0.5,
  "Diet": 0.4,
  "Previous_Heart_Problems": 0,
  "Medication_Use": 0,
  "Stress_Level": 0.3,
  "Sedentary_Hours_Per_Day": 0.6,
  "Income": 0.5,
  "BMI": 0.4,
  "Triglycerides": 0.5,
  "Physical_Activity_Days_Per_Week": 0.5,
  "Sleep_Hours_Per_Day": 0.5,
  "Blood_sugar": 0.4,
  "CK_MB": 0.3,
  "Troponin": 0.2,
  "Systolic_blood_pressure": 0.5,
  "Diastolic_blood_pressure": 0.4,
  "Gender": 1
}
```

- Нажми **Execute** и получи предсказание:

```json
{
  "prediction": 0,
  "probability_of_risk": 0.4479
}
```

---

### Предсказание через CSV

- Открой `/predict_csv` в Swagger.
- Нажми **Try it out**.
- Загрузите CSV файл с колонками, соответствующими признакам модели.
- Получи JSON с предсказаниями по каждой строке.

---

## Используемые технологии

- **Python 3.11**
- **FastAPI**
- **Uvicorn**
- **Pandas**
- **CatBoost**



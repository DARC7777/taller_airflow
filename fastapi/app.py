from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
import os

app = FastAPI()

# Lista de nombres de modelos que esperamos encontrar
model_names = ["random_forest", "decision_tree", "svm", "logistic_regression"]

# Cargamos los modelos desde el directorio compartido (asegúrate de que la ruta coincide con el mapeo en Docker)
models = {}
for name in model_names:
    file_path = f"/opt/airflow/models/{name}.pkl"
    if os.path.exists(file_path):
        models[name] = joblib.load(file_path)
    else:
        print(f"Warning: El modelo {name} no existe en {file_path}")

# Modelo seleccionado por defecto
selected_model = "random_forest"

# Definición del esquema de entrada para la predicción
class PenguinFeatures(BaseModel):
    culmen_length_mm: float
    culmen_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    island: int

@app.post("/predict/")
def predict(features: PenguinFeatures):
    global selected_model
    if selected_model not in models:
        raise HTTPException(status_code=400, detail=f"Modelo {selected_model} no encontrado.")
    
    model_data = models[selected_model]
    model = model_data["model"]
    scaler = model_data["scaler"]

    # Preparar los datos de entrada
    input_data = np.array([[features.culmen_length_mm,
                             features.culmen_depth_mm,
                             features.flipper_length_mm,
                             features.body_mass_g,
                             features.island]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    sex = "MALE" if prediction[0] == 1 else "FEMALE"
    
    return {"selected_model": selected_model, "prediction": sex}

@app.put("/select_model/{model_name}")
def select_model(model_name: str):
    global selected_model
    if model_name not in models:
        raise HTTPException(status_code=400, detail="Modelo no encontrado.")
    selected_model = model_name
    return {"message": f"Modelo cambiado a {model_name}"}

@app.get("/")
def home():
    return {"message": "API de Predicción de Pingüinos con FastAPI"}

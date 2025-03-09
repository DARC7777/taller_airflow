from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.operators.mysql import MySqlOperator
from datetime import datetime, timedelta
import pandas as pd
import sqlalchemy
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Parámetros por defecto del DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 3, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'penguins_workflow',
    default_args=default_args,
    description='Workflow para cargar, preprocesar y entrenar modelos de datos de pingüinos',
    schedule_interval=timedelta(days=1),
    catchup=False
)

# Tarea 1: Borrar las tablas existentes en MySQL (si existen)
clear_tables = MySqlOperator(
    task_id='clear_penguins_tables',
    mysql_conn_id='mysql_default',  # Asegúrate de haber configurado esta conexión en Airflow
    sql="""
        DROP TABLE IF EXISTS penguins_raw;
        DROP TABLE IF EXISTS penguins_preprocessed;
    """,
    dag=dag,
)

# Tarea 2: Cargar datos CSV en MySQL sin preprocesamiento
def load_data():
    # Ruta donde se encuentran los archivos CSV. Ajusta según como hayas montado el volumen.
    data_dir = '/data'
    file1 = os.path.join(data_dir, 'penguins_lter.csv')
    file2 = os.path.join(data_dir, 'penguins_size.csv')
    
    # Lectura de los archivos CSV
    df1 = pd.read_csv(file1, sep=",")
    df2 = pd.read_csv(file2, sep=",")
    
    # Procesamiento básico para el primer archivo
    df1["Sex"] = df1["Sex"].replace({".": None})
    df1_clean = df1.dropna(subset=["Sex"])
    df1_clean = df1_clean.rename(columns={
        "Species": "species",
        "Island": "island",
        "Culmen Length (mm)": "culmen_length_mm",
        "Culmen Depth (mm)": "culmen_depth_mm",
        "Flipper Length (mm)": "flipper_length_mm",
        "Body Mass (g)": "body_mass_g",
        "Sex": "sex"
    })
    df1_clean = df1_clean[["species", "island", "culmen_length_mm", "culmen_depth_mm",
                           "flipper_length_mm", "body_mass_g", "sex"]]
    
    # Procesamiento básico para el segundo archivo
    df2["sex"] = df2["sex"].replace({".": None})
    df2_clean = df2.dropna(subset=["sex"])
    
    # Concatenar ambos DataFrames
    df_combined = pd.concat([df1_clean, df2_clean], ignore_index=True)
    
    # Convertir la columna 'sex' a valores numéricos: 1 para MALE y 0 para FEMALE
    df_combined["sex"] = df_combined["sex"].map({"MALE": 1, "FEMALE": 0})
    
    # Conectar a la base de datos MySQL
    engine = sqlalchemy.create_engine('mysql+pymysql://model_user:model_password@mysql/model_db')
    # Guardar los datos en la tabla 'penguins_raw'
    df_combined.to_sql('penguins_raw', con=engine, if_exists='replace', index=False)
    print("Datos crudos cargados en MySQL.")

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

# Tarea 3: Preprocesar los datos y guardarlos en MySQL
def preprocess_data():
    engine = sqlalchemy.create_engine('mysql+pymysql://model_user:model_password@mysql/model_db')
    # Leer los datos cargados
    df = pd.read_sql('SELECT * FROM penguins_raw', con=engine)
    
    # Codificar la columna 'island'
    label_encoder = LabelEncoder()
    df["island"] = label_encoder.fit_transform(df["island"])
    
    # Guardar los datos preprocesados en la tabla 'penguins_preprocessed'
    df.to_sql('penguins_preprocessed', con=engine, if_exists='replace', index=False)
    print("Datos preprocesados almacenados en MySQL.")

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

# Tarea 4: Entrenar modelos con los datos preprocesados y guardar los artefactos
def train_models():
    engine = sqlalchemy.create_engine('mysql+pymysql://model_user:model_password@mysql/model_db')
    df = pd.read_sql('SELECT * FROM penguins_preprocessed', con=engine)
    
    # Seleccionar características (features) y objetivo (target)
    X = df[["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g", "island"]]
    y = df["sex"]
    
    # Normalización de características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # División de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Definición de modelos a entrenar
    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "decision_tree": DecisionTreeClassifier(),
        "svm": SVC(kernel="linear", probability=True),
        "logistic_regression": LogisticRegression()
    }
    
    # Asegurar que la carpeta para guardar modelos exista.
    # La ruta debe ser la misma que se comparte con FastAPI (por ejemplo, usando un volumen)
    models_dir = "/home/app/models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Entrenar cada modelo y guardarlo
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_path = os.path.join(models_dir, f"{name}.pkl")
        joblib.dump({"model": model, "scaler": scaler}, model_path)
        print(f"Modelo {name} entrenado y guardado en {model_path}")
    
    print("Todos los modelos han sido entrenados.")

train_models_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    dag=dag,
)

# Definir la secuencia de ejecución:
clear_tables >> load_data_task >> preprocess_data_task >> train_models_task

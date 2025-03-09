FROM apache/airflow:2.6.0

# Cambiamos a usuario root para poder instalar dependencias del sistema
USER root

# Actualizamos e instalamos dependencias del sistema necesarias para compilar paquetes de Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libssl-dev \
    libffi-dev \
    build-essential \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copiamos el archivo requirements.txt al contenedor
COPY requirements.txt /tmp/requirements.txt

# Instalamos las librerías adicionales necesarias para el entrenamiento
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Volvemos al usuario airflow para la ejecución de Airflow
USER airflow

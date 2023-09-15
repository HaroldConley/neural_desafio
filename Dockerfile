# Usa una imagen de Python como base
FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia el archivo Python al directorio de trabajo
COPY api_rest.py .
COPY reg_logist_best.pkl .
COPY new_train.py .
COPY demo_df.pkl .

# Instala Flask, pandas y scikit
RUN pip install pandas flask scikit-learn

# Expone el puerto 5000 para la aplicación
EXPOSE 5000

# Ejecuta la aplicación cuando se inicie el contenedor
CMD ["python", "api_rest.py"]

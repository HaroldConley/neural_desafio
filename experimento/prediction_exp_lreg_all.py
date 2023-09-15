# Código para realizar experimentos con el modelo de predicción de Regresión Logística usando toda la data (sin balancear)
import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # Configurando MLFlow
    mlflow.set_tracking_uri("venv/scripts/mlruns")  # Reemplaza con la URI de tu servidor de MLflow
    mlflow.set_experiment("LR_all_experiment")  # Define el nombre y la descripción del experimento en MLflow. Un experimento puede tener varias corridas y se comparan los valores de distintas corridas.

    # Importando la base de datos lista para el experimento
    atrasos_df = pd.read_csv('atrasos_df.csv')

    # Dividir los datos en conjuntos de entrenamiento y prueba
    x = atrasos_df.loc[:, ~atrasos_df.columns.isin(['label'])]
    y = atrasos_df['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Iniciar una corrida de MLflow
    with mlflow.start_run(run_name='test_class_weight'):  # se puede cambiar el nombre entre cada corrida.
        # Crear y entrenar el modelo
        #class_weight = 'balanced'
        logReg = LogisticRegression(random_state=13, class_weight='balanced')
        model = logReg.fit(x_train, y_train)

        # Realizar predicciones en el conjunto de prueba
        y_pred = model.predict(x_test)

        # Calcular métricas del modelo
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Extraer los valores de la matriz de confusión
        tn, fp, fn, tp = conf_matrix.ravel()

        # Calcular la Especificidad
        specificity = tn / (tn + fp)

        # Registrar los parámetros, métricas y modelo en MLflow
        #mlflow.log_param("Proporcion_0", proporcion_label_0)
        #mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("specificity", specificity)
        #mlflow.sklearn.log_model(model, "model")

    # Finalizar la corrida
    mlflow.end_run()

# Código para mejorar y hacer experimentos con el modelo de predicción XGBoost.
import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # Configurando MLFlow
    mlflow.set_tracking_uri("venv/scripts/mlruns")  # Reemplaza con la URI de tu servidor de MLflow
    mlflow.set_experiment("XGBoost_mejora")  # Define el nombre del experimento en MLflow. Un experimento puede tener varias corridas y se comparan los valores de distintas corridas.

    # Importando la base de datos lista para el experimento
    atrasos_df = pd.read_csv('atrasos_df_weather_airport.csv')


    for learning_rate in np.arange(0.005, 0.021, 0.001):
        #learning_rate = 0.01
        # Dividir los datos en conjuntos de entrenamiento y prueba
        x = atrasos_df.loc[:, ~atrasos_df.columns.isin(['label'])]
        y = atrasos_df['label']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        learning_rate = round(learning_rate, 3)
        # Iniciar una corrida de MLflow
        with mlflow.start_run(run_name=f'test_{learning_rate}'):  # se puede cambiar el nombre entre cada corrida.
            # Crear y entrenar el modelo
            clase_atraso = len(atrasos_df[atrasos_df['label'] == 1])
            clase_no_atraso = len(atrasos_df[atrasos_df['label'] == 0])
            proporcion_clases_atraso = clase_no_atraso/clase_atraso

            modelxgb = xgb.XGBClassifier(random_state=1, learning_rate=learning_rate, scale_pos_weight=proporcion_clases_atraso)
            modelxgb = modelxgb.fit(x_train, y_train)

            # Realizar predicciones en el conjunto de prueba
            y_pred = modelxgb.predict(x_test)

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
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("specificity", specificity)
            #mlflow.sklearn.log_model(model, "model")

        # Finalizar la corrida
        mlflow.end_run()

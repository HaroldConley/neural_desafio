# Código para realizar experimentos con el modelo de predicción XGBoost.
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
    mlflow.set_experiment("XGBoost_weather_airport")  # Define el nombre del experimento en MLflow. Un experimento puede tener varias corridas y se comparan los valores de distintas corridas.

    # Importando la base de datos lista para el experimento
    atrasos_df = pd.read_csv('atrasos_df_weather_airport.csv')


    # Función para balancear la base de datos en distintas proporciones de label=0 y label=1
    def balance(proporcion_label_0):
        #proporcion_label_0 = 1
        # Divide el DataFrame en dos partes: una con label 0 y otra con label 1
        df_label_0 = atrasos_df[atrasos_df['label'] == 0]
        df_label_1 = atrasos_df[atrasos_df['label'] == 1]

        # Contar cuántas muestras hay de la clase sub-representada
        count_label_1 = len(df_label_1)

        # Calcular el número de muestras a tomar de la clase sobre-representada
        min_count_label_0 = int(count_label_1 * proporcion_label_0)

        # Muestrear aleatoriamente 'min_count_label_0' muestras de label 0
        df_label_0_sampled = df_label_0.sample(n=min_count_label_0, random_state=42)

        # Concatenar los dos DataFrames resultantes para obtener el conjunto de datos balanceado
        balanced_data = pd.concat([df_label_0_sampled, df_label_1], axis=0)

        return balanced_data

    for prop in np.arange(0.5, 1.6, 0.1):
        for learning_rate in np.arange(0.005, 0.021, 0.001):
            # Balanceo
            proporcion_label_0 = round(prop, 1)
            atrasos_df_balance = balance(proporcion_label_0)

            # Dividir los datos en conjuntos de entrenamiento y prueba
            x = atrasos_df_balance.loc[:, ~atrasos_df_balance.columns.isin(['label'])]
            y = atrasos_df_balance['label']
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            learning_rate = round(learning_rate, 3)
            # Iniciar una corrida de MLflow
            with mlflow.start_run(run_name=f'test_{proporcion_label_0}_{learning_rate}'):  # se puede cambiar el nombre entre cada corrida.
                # Crear y entrenar el modelo
                #learning_rate = 0.01
                modelxgb = xgb.XGBClassifier(random_state=1, learning_rate=learning_rate)
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
                mlflow.log_param("Proporcion_0", proporcion_label_0)
                mlflow.log_param("learning_rate", learning_rate)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("specificity", specificity)
                #mlflow.sklearn.log_model(model, "model")

            # Finalizar la corrida
            mlflow.end_run()

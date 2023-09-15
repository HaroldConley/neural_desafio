# Código para Serializar la LR con nuevos datos.
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

import warnings

warnings.filterwarnings('ignore')


# Función para balancear la base de datos en distintas proporciones de label=0 y label=1
def balance(proporcion_label_0, data_df):
    # proporcion_label_0 = 1
    # Divide el DataFrame en dos partes: una con label 0 y otra con label 1
    df_label_0 = data_df[data_df['label'] == 0]
    df_label_1 = data_df[data_df['label'] == 1]

    # Contar cuántas muestras hay de la clase sub-representada
    count_label_1 = len(df_label_1)

    # Calcular el número de muestras a tomar de la clase sobre-representada
    min_count_label_0 = int(count_label_1 * proporcion_label_0)

    # Muestrear aleatoriamente 'min_count_label_0' muestras de label 0
    df_label_0_sampled = df_label_0.sample(n=min_count_label_0, random_state=42)

    # Concatenar los dos DataFrames resultantes para obtener el conjunto de datos balanceado
    balanced_data = pd.concat([df_label_0_sampled, df_label_1], axis=0)

    return balanced_data


# Función para entrenar el modelo dada una base de datos en formato DF
def entrenamiento(data_df):
    # Balanceo
    proporcion_label_0 = 1  # Es la proporción que EN ESTE CASO generó mejores resultados
    atrasos_df_balance = balance(proporcion_label_0, data_df)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    x = atrasos_df_balance.loc[:, ~atrasos_df_balance.columns.isin(['label'])]
    y = atrasos_df_balance['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo
    logReg = LogisticRegression(random_state=13)
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

    print(accuracy, recall, specificity)

    # Serializar (guardar) el modelo
    with open('/mi_volumen/reg_logist_new.pkl', 'wb') as file:
        pickle.dump(model, file)

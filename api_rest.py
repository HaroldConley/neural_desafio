# Código para implementar la API REST del modelo de predicción

import pickle
from flask import Flask, request, jsonify
import pandas as pd
import os
from new_train import entrenamiento

app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
    # Chequea si existe algún modelo NUEVO entrenado. Si no existe, usa el original
    if os.path.exists('/mi_volumen/reg_logist_new.pkl'):  # OJO que para que funcione en el contenedor DOCKER la dirección debe ser '/mi_volumen/hello.txt' (absoluta), pero fuera del docker es 'mi_volumen/hello.txt' (relativa)
        with open('/mi_volumen/reg_logist_new.pkl', 'rb') as file:
            model = pickle.load(file)
    else:
        # Carga el modelo predeterminado (inicial)
        with open('reg_logist_best.pkl', 'rb') as file:
            model = pickle.load(file)

    # Obtén los datos de entrada desde la solicitud en formato JSON
    data_json = request.get_json()

    # Convierte los datos JSON en un DataFrame
    input_df = pd.DataFrame(data_json)

    # Realiza la predicción utilizando el modelo cargado
    prediction = model.predict(input_df)

    # Devuelve la predicción como respuesta
    return jsonify({'prediction': prediction.tolist()})


# Ruta para re-entrenar el modelo con nuevos datos
@app.route('/retrain/', methods=['POST'])
def retrain():
    # Obtén los datos de entrada desde la solicitud en formato JSON
    data_json = request.get_json()

    # Convierte los datos JSON en un DataFrame
    input_df = pd.DataFrame(data_json)

    # Re-entrena el modelo
    entrenamiento(input_df)

    return 'Modelo re-entrenado.'


# Ruta para re-entrenar el modelo con nuevos datos
@app.route('/demo/')
def demo():
    # Carga el DF de demo
    with open('demo_df.pkl', 'rb') as file:
        data = pickle.load(file)

    # Carga el modelo predeterminado (inicial)
    with open('reg_logist_best.pkl', 'rb') as file:
        model = pickle.load(file)

    # Realiza la predicción utilizando el modelo cargado
    prediction = model.predict(data)

    # Devuelve la predicción como respuesta
    return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

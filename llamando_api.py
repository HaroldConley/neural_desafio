import requests
import pandas as pd
import json


# Importando la base de datos para tener el modelo de Input para los llamados a la API
atrasos_df = pd.read_csv('atrasos_df.csv')


###################################################################
######               Pedir predicción a la API              #######
###################################################################

# URL de la API
url = 'http://64.176.6.184:5000'

# Dato en el formato necesario (como el DF con el que fue entrenado el modelo).
data = atrasos_df.iloc[9:10]  # Debe ser tipo DF. Por ejemplo x_test.iloc[0:1]

# Convertir el DataFrame a formato JSON
data_json = data.to_json(orient='records')

# Realizar la solicitud POST a la API
response = requests.post(url, json=json.loads(data_json))

# Verificar la respuesta
if response.status_code == 200:
    result = response.json()
    prediction = result['prediction']
    print(f'Predicción del modelo: {prediction}')
else:
    print('Error en la solicitud a la API.')


################################################################################
######               Pedir a la API generar un nuevo modelo              #######
################################################################################

# URL de la API
url = 'http://64.176.6.184:5000/retrain/'

# Dato en el formato necesario (como el DF con el que fue entrenado el modelo).
data = atrasos_df.iloc[0:35000]

# Convertir el DataFrame a formato JSON
data_json = data.to_json(orient='records')

# Realizar la solicitud POST a la API
response = requests.post(url, json=json.loads(data_json))

# Verificar la respuesta
if response.status_code == 200:
    print(response.text)
else:
    print('Error en la solicitud a la API.')


################################################################################
######                 DEMO con una solicitud predefinida                #######
################################################################################

# URL de la API
url = 'http://64.176.6.184:5000/demo/'

# Realizar la solicitud a la API
response = requests.get(url)

# Verificar la respuesta
if response.status_code == 200:
    result = response.json()
    prediction = result['prediction']
    print(f'Predicción del modelo: {prediction}')
else:
    print('Error en la solicitud a la API.')

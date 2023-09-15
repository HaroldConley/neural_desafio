#Se importan las librerías necesarias para el problema
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

import missingno as msng
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (15, 10)


# Importando DB
df = pd.read_csv('dataset_SCL.csv')

##########################################
####    Generando datos adicionales   ####
##########################################
# Temporada alta
def temporada_alta(fecha):
    fecha_año = int(fecha.split('-')[0])
    fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year=fecha_año)
    range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year=fecha_año)
    range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year=fecha_año)
    range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year=fecha_año)
    range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year=fecha_año)
    range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year=fecha_año)
    range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year=fecha_año)
    range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year=fecha_año)

    if ((fecha >= range1_min and fecha <= range1_max) or
            (fecha >= range2_min and fecha <= range2_max) or
            (fecha >= range3_min and fecha <= range3_max) or
            (fecha >= range4_min and fecha <= range4_max)):
        return 1
    else:
        return 0

df['temporada_alta'] = df['Fecha-I'].apply(temporada_alta)

# Diferencia en minutos entre Viaje programado y Viaje realizado
def dif_min(data):
    fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    dif_min = ((fecha_o - fecha_i).total_seconds())/60
    return dif_min

df['dif_min'] = df.apply(dif_min, axis = 1)

# Atraso: si demora más de 15 minutos = atrasado (1), si sale bien = 0.
df['atraso_15'] = np.where(df['dif_min'] > 15, 1, 0)

# Periodo del día: mañana (entre 5:00 y 11:59), tarde (entre 12:00 y 18:59) y noche (entre 19:00 y 4:59), en base a  Fecha-I (programada)
def get_periodo_dia(fecha):
    fecha_time = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S').time()
    mañana_min = datetime.strptime("05:00", '%H:%M').time()
    mañana_max = datetime.strptime("11:59", '%H:%M').time()
    tarde_min = datetime.strptime("12:00", '%H:%M').time()
    tarde_max = datetime.strptime("18:59", '%H:%M').time()
    noche_min1 = datetime.strptime("19:00", '%H:%M').time()
    noche_max1 = datetime.strptime("23:59", '%H:%M').time()
    noche_min2 = datetime.strptime("00:00", '%H:%M').time()
    noche_max2 = datetime.strptime("4:59", '%H:%M').time()

    if (fecha_time > mañana_min and fecha_time < mañana_max):
        return 'mañana'
    elif (fecha_time > tarde_min and fecha_time < tarde_max):
        return 'tarde'
    elif ((fecha_time > noche_min1 and fecha_time < noche_max1) or
          (fecha_time > noche_min2 and fecha_time < noche_max2)):
        return 'noche'

df['periodo_dia'] = df['Fecha-I'].apply(get_periodo_dia)


############################################################################
#############            Preparando Distintos DBs           ################
############################################################################

# Copiando lo hecho con la DB en el trabajo original
data = shuffle(df[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'atraso_15']], random_state = 111)

features = pd.concat([pd.get_dummies(data['OPERA'], prefix='OPERA'),
                      pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
                      pd.get_dummies(data['MES'], prefix='MES')],
                     axis=1)
label = data['atraso_15']

# Juntado en un solo DF para guardarlo:
atrasos_df = features
atrasos_df['label'] = label

atrasos_df.to_csv('atrasos_df.csv', index=False)  # Guarda el DF.


# Agregando el temporada alto y sacando tipo de vuelo (pq se considerará el destino y sería redundante)
data = shuffle(df[['OPERA', 'MES', 'SIGLADES', 'DIANOM', 'temporada_alta', 'atraso_15']], random_state = 111)

features = pd.concat([pd.get_dummies(data['OPERA'], prefix='OPERA'),
                      pd.get_dummies(data['MES'], prefix='MES'),
                      pd.get_dummies(data['SIGLADES'], prefix='SIGLADES'),
                      pd.get_dummies(data['DIANOM'], prefix='DIANOM')],
                     axis=1)
label = data['atraso_15']

# Juntado en un solo DF para guardarlo:
atrasos_df = features
atrasos_df['temporada_alta'] = data['temporada_alta']
atrasos_df['label'] = label

atrasos_df.to_csv('atrasos_df_weather_airport.csv', index=False)  # Guarda el DF.


# Agregando el destino
data = shuffle(df[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'atraso_15']], random_state = 111)
features = pd.concat([pd.get_dummies(data['OPERA'], prefix='OPERA'),
                      pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
                      pd.get_dummies(data['MES'], prefix='MES'),
                      pd.get_dummies(data['SIGLADES'], prefix='DES')],
                     axis=1)

label = data['atraso_15']

# Juntado en un solo DF para guardarlo:
atrasos_df = features
atrasos_df['label'] = label

atrasos_df.to_csv('atrasos_df_destino.csv', index=False)  # Guarda el DF.


# Dejando solo destino y opera
data = shuffle(df[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'atraso_15']], random_state = 111)
features = pd.concat([pd.get_dummies(data['OPERA'], prefix='OPERA'),
                      pd.get_dummies(data['SIGLADES'], prefix='DES')],
                     axis=1)

label = data['atraso_15']

# Juntado en un solo DF para guardarlo:
atrasos_df = features
atrasos_df['label'] = label

atrasos_df.to_csv('atrasos_df_dest_opera.csv', index=False)  # Guarda el DF.





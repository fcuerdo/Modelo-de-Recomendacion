import streamlit as st
import pandas as pd
import joblib
from google.cloud import storage

# Función para evaluar la viabilidad en función del score
def evaluar_viabilidad(score):
    if score > 18000:
        return "Altamente viable"
    elif score > 13513:
        return "Viable"
    elif score > 7000:
        return "Poco viable"
    else:
        return "No viable"

# Función para descargar el modelo desde Cloud Storage y cargarlo
def cargar_modelo_desde_cloud(bucket_name, source_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    with blob.open("rb") as f:
        modelo = joblib.load(f)
    return modelo

# Carga el modelo directamente desde Cloud Storage
bucket_name = 'cloud-ai-platform-8d772b74-2a6f-4a50-8e4b-fac7fb0c8586'
source_blob_name = 'Modelo_RandomForest.pkl'
modelo = cargar_modelo_desde_cloud(bucket_name, source_blob_name)

# Interfaz de Streamlit
st.title('Recomendación de Inversión')

latitude = st.number_input('Latitud', format="%.6f")
longitude = st.number_input('Longitud', format="%.6f")

if st.button('Obtener Recomendación'):
    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
        st.error('Las coordenadas proporcionadas no son válidas.')
    else:
        coordenadas = pd.DataFrame([[latitude, longitude]], columns=['latitude', 'longitude'])
        prediccion = modelo.predict(coordenadas)
        recomendacion = evaluar_viabilidad(prediccion[0])
        st.success(f'La inversion en esta Zona es: {recomendacion}')

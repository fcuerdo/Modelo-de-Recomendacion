import streamlit as st
import pandas as pd
import joblib
from google.cloud import storage
from googlemaps import Client as GoogleMapsClient
import os


google_maps_client = GoogleMapsClient(key=os.getenv('GOOGLE_MAPS_API_KEY'))

def obtener_lugares_cercanos(latitude, longitude, tipo_lugar='restaurant'):
    # Encuentra lugares cercanos del tipo especificado
    lugares = google_maps_client.places_nearby(location=(latitude, longitude), type=tipo_lugar, radius=1000)
    return lugares['results']


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
st.title('Análisis Predictivo de Ubicaciones para Inversiones Hoteleras')

st.markdown("""
Esta aplicación proporciona una recomendación de inversión basada en un modelo predictivo que analiza la viabilidad de ubicaciones hoteleras en zonas próximas a estadios seleccionados en los Estados Unidos. Utilizando datos enriquecidos de Google Maps, el modelo compara las puntuaciones y evalúa el potencial de éxito para el crecimiento sostenido de negocios en el contexto del Mundial de Fútbol 2026 y otros eventos significativos.
""")

latitude = st.number_input('Latitud', format="%.6f")
longitude = st.number_input('Longitud', format="%.6f")

if st.button('Obtén una Recomendación'):
    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
        st.error('Las coordenadas proporcionadas no son válidas.')
    else:
        coordenadas = pd.DataFrame([[latitude, longitude]], columns=['latitude', 'longitude'])
        prediccion = modelo.predict(coordenadas)
        recomendacion = evaluar_viabilidad(prediccion[0])
        st.success(f'La inversión en esta Zona es: {recomendacion}')
        
        # Obtener lugares cercanos que podrían mejorar la inversión
        lugares_cercanos = obtener_lugares_cercanos(latitude, longitude)
        st.write('Lugares cercanos que podrían mejorar la inversión:')
        for lugar in lugares_cercanos:
            st.write(f"- {lugar['name']} ({lugar['vicinity']})")


if st.button('Mostrar Mapa'):
    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
        st.error('Las coordenadas proporcionadas no son válidas.')
    else:
        mapa_data = pd.DataFrame([[latitude, longitude]], columns=['lat', 'lon'])
        st.map(mapa_data)

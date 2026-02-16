import streamlit as st
import librosa
import numpy as np
import tempfile
import os

# Título y diseño
st.title("🎹 Detector de Acordes de Edmuz")
st.write("Sube tu archivo MP3 y la IA intentará decirte qué acordes suenan.")

# Subida de archivo
archivo_audio = st.file_uploader("Sube tu canción aquí", type=["mp3", "wav"])

if archivo_audio is not None:
    # 1. Mostrar reproductor
    st.audio(archivo_audio)
    
    if st.button("Analizar ahora"):
        st.info("Procesando... Esto puede tardar unos 30 segundos. ¡Paciencia!")
        
        # 2. Guardar temporalmente el archivo para que librosa lo lea
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(archivo_audio.getvalue())
            ruta_temporal = tmp.name

        try:
            # 3. Cargar audio (Solo los primeros 60 segundos para no saturar la memoria gratis)
            y, sr = librosa.load(ruta_temporal, duration=60)
            
            # 4. Magia matemática: Cromagrama (detectar notas)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # Definir notas musicales
            notas = ['Do', 'Do#', 'Re', 'Re#', 'Mi', 'Fa', 'Fa#', 'Sol', 'Sol#', 'La', 'La#', 'Si']
            
            # Analizar cada segundo
            resultados = []
            tiempo_total = librosa.get_duration(y=y, sr=sr)
            
            # Vamos a revisar cada 2 segundos
            for t in range(0, int(tiempo_total), 2):
                # Tomar un pedazo del audio
                frame_index = librosa.time_to_frames(t, sr=sr)
                if frame_index < chroma.shape[1]:
                    # Ver qué nota suena más fuerte en ese momento
                    columna = chroma[:, frame_index]
                    indice_nota_mas_fuerte = np.argmax(columna)
                    nota_detectada = notas[indice_nota_mas_fuerte]
                    
                    resultados.append(f"Segundos {t}-{t+2}: Probablemente {nota_detectada}")

            # 5. Mostrar resultados
            st.success("¡Análisis terminado!")
            st.text_area("Resultados:", value="\n".join(resultados), height=300)

        except Exception as e:
            st.error(f"Error al procesar: {e}")
            
        finally:
            # Limpiar archivo temporal
            os.remove(ruta_temporal)
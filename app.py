import streamlit as st
import librosa
import numpy as np
import whisper
import tempfile
import os

# --- Configuración Visual ---
st.set_page_config(page_title="Acordes y Letra IA", page_icon="🎵")

# Estilo CSS para que se vea como en tu imagen
st.markdown("""
    <style>
    .bloque-musical {
        font-family: monospace;
        margin-bottom: 20px;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .acorde {
        color: #0068c9; /* Azul fuerte */
        font-weight: bold;
        font-size: 18px;
    }
    .letra {
        color: #31333F;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎵 Transcriptor de Canciones con Acordes")
st.info("Sube tu canción. La IA escuchará la letra y calculará los acordes por frase.")

# --- 1. Función de Detección de Acordes (Croma) ---
def detectar_acorde_en_segmento(y, sr):
    # Notas musicales (Cifrado Americano)
    notas = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Extraer energía de notas (Chroma)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    # Promediar la energía de cada nota en este segmento
    promedio_notas = np.mean(chroma, axis=1)
    
    # Encontrar la nota más fuerte (Raíz)
    idx_raiz = np.argmax(promedio_notas)
    nota_raiz = notas[idx_raiz]
    
    # Determinar si es Mayor o Menor (heurística simple basada en la tercera)
    # Tercera Mayor esta a +4 semitonos, Menor a +3
    idx_tercera_mayor = (idx_raiz + 4) % 12
    idx_tercera_menor = (idx_raiz + 3) % 12
    
    val_mayor = promedio_notas[idx_tercera_mayor]
    val_menor = promedio_notas[idx_tercera_menor]
    
    calidad = "m" if val_menor > val_mayor else "" # Si gana la menor, ponemos 'm'
    
    return f"{nota_raiz}{calidad}"

# --- 2. Cargar Modelo Whisper (Letras) ---
@st.cache_resource
def cargar_whisper():
    # Usamos el modelo "tiny" porque Streamlit Cloud tiene poca memoria RAM.
    # Si fuera en tu PC potente, usaríamos "base" o "small".
    return whisper.load_model("tiny")

# --- Interfaz Principal ---
archivo = st.file_uploader("Sube tu archivo MP3", type=["mp3", "wav"])

if archivo is not None:
    st.audio(archivo)
    
    if st.button("Analizar Ahora (Letra + Acordes)"):
        with st.spinner("⏳ Cargando IA y escuchando la canción... (Esto puede tardar 1-2 minutos)"):
            
            # Guardar temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(archivo.getvalue())
                ruta_tmp = tmp.name
            
            try:
                # A. Transcribir Letra con Whisper
                modelo = cargar_whisper()
                resultado = modelo.transcribe(ruta_tmp)
                segmentos = resultado['segments'] # Lista de frases con tiempos
                
                # B. Cargar Audio para música con Librosa
                y_completo, sr = librosa.load(ruta_tmp)
                
                st.success("¡Análisis completado! Aquí tienes tu canción:")
                st.divider()
                
                # C. Procesar cada frase
                for seg in segmentos:
                    inicio = seg['start']
                    fin = seg['end']
                    texto = seg['text']
                    
                    # Cortar el audio justo en esa frase
                    inicio_sample = int(inicio * sr)
                    fin_sample = int(fin * sr)
                    
                    if fin_sample > len(y_completo):
                        fin_sample = len(y_completo)
                        
                    y_segmento = y_completo[inicio_sample:fin_sample]
                    
                    # Detectar acorde de ese pedacito
                    if len(y_segmento) > 0:
                        acorde = detectar_acorde_en_segmento(y_segmento, sr)
                    else:
                        acorde = ""
                    
                    # D. Mostrar Estilo "Imagen Adjunta" (Azul arriba, Texto abajo)
                    html_bloque = f"""
                    <div class="bloque-musical">
                        <div class="acorde">{acorde}</div>
                        <div class="letra">{texto}</div>
                    </div>
                    """
                    st.markdown(html_bloque, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")
                st.warning("Nota: A veces la memoria gratis de Streamlit se llena. Si falla, intenta con una canción más corta.")
            
            finally:
                os.remove(ruta_tmp)

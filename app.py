import streamlit as st
import librosa
import numpy as np
import whisper
import tempfile
import os

# --- 1. Configuración de la Página ---
st.set_page_config(page_title="Cancionero IA", page_icon="🎸", layout="wide")

# CSS: El diseño visual (Acorde azul arriba, letra negra abajo)
st.markdown("""
    <style>
    .cancionero-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        line-height: 1.6;
        font-family: Arial, sans-serif;
        padding: 20px;
        background-color: #ffffff;
    }
    .bloque-palabra {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-right: 4px;
        margin-bottom: 15px;
    }
    .acorde-style {
        color: #007bff; /* Azul */
        font-weight: bold;
        font-size: 15px;
        height: 20px;
        margin-bottom: 2px;
    }
    .letra-style {
        color: #000;
        font-size: 18px;
    }
    .instrumental-box {
        border: 1px dashed #aaa;
        padding: 5px 10px;
        border-radius: 5px;
        background-color: #f9f9f9;
        margin-bottom: 15px;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .inst-label {
        font-size: 12px;
        color: #666;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎸 Tu Cancionero Automático")

# --- 2. Funciones Musicales ---
def obtener_nombre_acorde(chroma_mean):
    notas = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    idx = np.argmax(chroma_mean)
    nota = notas[idx]
    
    tercera_mayor = (idx + 4) % 12
    tercera_menor = (idx + 3) % 12
    
    if chroma_mean[tercera_menor] > chroma_mean[tercera_mayor] * 1.1:
        return f"{nota}m"
    return nota

def analizar_segmento(y, sr):
    if len(y) == 0: return ""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    promedio = np.mean(chroma, axis=1)
    return obtener_nombre_acorde(promedio)

# --- 3. Carga IA ---
@st.cache_resource
def cargar_whisper():
    return whisper.load_model("tiny")

# --- 4. App Principal ---
archivo = st.file_uploader("Sube tu MP3/WAV", type=["mp3", "wav"])

if archivo is not None:
    st.audio(archivo)
    
    if st.button("Generar Cancionero"):
        with st.spinner("🎧 Procesando... esto puede tardar 1-2 minutos."):
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(archivo.getvalue())
                ruta_tmp = tmp.name
            
            try:
                # Cargar audio y transcribir
                y, sr = librosa.load(ruta_tmp)
                modelo = cargar_whisper()
                resultado = modelo.transcribe(ruta_tmp, language="es")
                segmentos = resultado['segments']
                
                st.success("¡Listo! Aquí tienes la canción:")
                st.divider()
                
                # --- CONSTRUCCIÓN DEL HTML (Sin espacios extra para evitar el error) ---
                html_final = '<div class="cancionero-container">'
                cursor_tiempo = 0.0
                
                for seg in segmentos:
                    inicio = seg['start']
                    fin = seg['end']
                    texto_frase = seg['text'].strip()
                    
                    # 1. Detectar Instrumental (Huecos de silencio vocal)
                    if inicio - cursor_tiempo > 2.0:
                        idx_ini_gap = int(cursor_tiempo * sr)
                        idx_fin_gap = int(inicio * sr)
                        if idx_fin_gap > idx_ini_gap:
                            acorde_gap = analizar_segmento(y[idx_ini_gap:idx_fin_gap], sr)
                            # HTML Compacto en una sola linea
                            html_final += f'<div class="instrumental-box"><div class="acorde-style">{acorde_gap}</div><div class="inst-label">Música</div></div>'

                    # 2. Detectar Acorde de la Voz
                    idx_ini = int(inicio * sr)
                    idx_fin = int(fin * sr)
                    acorde_voz = analizar_segmento(y[idx_ini:idx_fin], sr)
                    
                    palabras = texto_frase.split(" ")
                    
                    # 3. Poner acorde solo en la primera palabra
                    for i, palabra in enumerate(palabras):
                        acorde_mostrar = acorde_voz if i == 0 else "&nbsp;"
                        # HTML Compacto
                        html_final += f'<div class="bloque-palabra"><div class="acorde-style">{acorde_mostrar}</div><div class="letra-style">{palabra}</div></div>'
                    
                    cursor_tiempo = fin
                
                html_final += '</div>'
                
                # Renderizar
                st.markdown(html_final, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                if os.path.exists(ruta_tmp):
                    os.remove(ruta_tmp)

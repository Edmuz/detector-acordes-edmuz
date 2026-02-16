import streamlit as st
import librosa
import numpy as np
import whisper
import tempfile
import os

# --- 1. Configuración de la Página y Estilo Visual (CSS) ---
st.set_page_config(page_title="Cancionero IA", page_icon="🎸", layout="wide")

# CSS CORREGIDO: Usamos "Flexbox" para asegurar que el acorde quede arriba de la letra
st.markdown("""
    <style>
    /* Contenedor principal: hace que las palabras fluyan como un texto normal */
    .cancionero-container {
        display: flex;
        flex-wrap: wrap; /* Permite que baje a la siguiente línea si no cabe */
        gap: 8px; /* Espacio entre palabras */
        line-height: 1.5;
        font-family: sans-serif;
    }

    /* Cada bloque es una pareja: Acorde + Palabra */
    .bloque-palabra {
        display: flex;
        flex-direction: column; /* Apila verticalmente (Acorde arriba, letra abajo) */
        align-items: center;    /* Centra el acorde con la palabra */
        margin-bottom: 15px;    /* Espacio entre líneas */
    }

    /* Estilo del Acorde (Arriba, Azul) */
    .acorde-style {
        color: #007bff; /* Azul como pediste */
        font-weight: bold;
        font-size: 14px;
        height: 20px; /* Altura fija para que si no hay acorde, reserve el espacio igual */
        margin-bottom: 2px;
    }

    /* Estilo de la Letra (Abajo, Negro) */
    .letra-style {
        color: #000000;
        font-size: 18px;
        white-space: pre; /* Respeta espacios */
    }
    
    /* Estilo para bloques instrumentales (Intro, Solos) */
    .instrumental-style {
        color: #888;
        font-style: italic;
        font-size: 14px;
        border: 1px dashed #ccc;
        padding: 5px;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎸 Transcriptor de Canciones (Estilo Cancionero)")

# --- 2. Funciones de Lógica Musical ---
def obtener_nombre_acorde(chroma_mean):
    # Diccionario de notas
    notas = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    idx = np.argmax(chroma_mean)
    nota = notas[idx]
    
    # Heurística simple para Mayor/Menor
    tercera_mayor = (idx + 4) % 12
    tercera_menor = (idx + 3) % 12
    
    # Si la tercera menor suena casi tan fuerte como la mayor, es menor
    if chroma_mean[tercera_menor] > chroma_mean[tercera_mayor] * 1.1:
        return f"{nota}m"
    return nota

def analizar_segmento(y, sr):
    if len(y) == 0: return ""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    promedio = np.mean(chroma, axis=1)
    return obtener_nombre_acorde(promedio)

# --- 3. Carga de IA (Whisper) ---
@st.cache_resource
def cargar_whisper():
    return whisper.load_model("tiny")

# --- 4. Aplicación Principal ---
archivo = st.file_uploader("Sube tu archivo (MP3/WAV)", type=["mp3", "wav"])

if archivo is not None:
    st.audio(archivo)
    
    if st.button("Analizar Canción"):
        with st.spinner("🎧 Escuchando letra y detectando acordes... (Paciencia)"):
            
            # Guardar archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(archivo.getvalue())
                ruta_tmp = tmp.name
            
            try:
                # A. Cargar Audio
                y, sr = librosa.load(ruta_tmp)
                
                # B. Transcribir (Forzando Español)
                modelo = cargar_whisper()
                resultado = modelo.transcribe(ruta_tmp, language="es")
                segmentos = resultado['segments']
                
                st.success("¡Transcripción Completa! Aquí tienes tu guía:")
                st.divider()
                
                # Iniciamos el contenedor HTML principal
                html_final = '<div class="cancionero-container">'
                
                cursor_tiempo = 0.0
                
                for seg in segmentos:
                    inicio = seg['start']
                    fin = seg['end']
                    texto_frase = seg['text'].strip()
                    
                    # --- Detectar Huecos (Instrumental) ---
                    if inicio - cursor_tiempo > 2.5:
                        # Analizamos el acorde del hueco
                        idx_ini_gap = int(cursor_tiempo * sr)
                        idx_fin_gap = int(inicio * sr)
                        
                        if idx_fin_gap > idx_ini_gap:
                            acorde_gap = analizar_segmento(y[idx_ini_gap:idx_fin_gap], sr)
                            # Añadimos bloque instrumental
                            html_final += f"""
                            <div class="bloque-palabra">
                                <div class="acorde-style">{acorde_gap}</div>
                                <div class="letra-style instrumental-style">Intermedio</div>
                            </div>
                            """

                    # --- Detectar Frase Cantada ---
                    # Dividimos la frase en palabras para intentar distribuir acordes (aprox)
                    # Nota: Whisper nos da la frase entera. Asignaremos el acorde principal a la primera palabra.
                    idx_ini = int(inicio * sr)
                    idx_fin = int(fin * sr)
                    acorde_voz = analizar_segmento(y[idx_ini:idx_fin], sr)
                    
                    palabras = texto_frase.split(" ")
                    
                    for i, palabra in enumerate(palabras):
                        # Solo ponemos el acorde en la primera palabra de la frase (aproximación)
                        acorde_a_mostrar = acorde_voz if i == 0 else "&nbsp;" 
                        
                        html_final += f"""
                        <div class="bloque-palabra">
                            <div class="acorde-style">{acorde_a_mostrar}</div>
                            <div class="letra-style">{palabra}</div>
                        </div>
                        """
                    
                    cursor_tiempo = fin

                html_final += '</div>' # Cerrar contenedor
                
                # --- PUNTO CLAVE: Aquí es donde fallaba antes ---
                # Usamos unsafe_allow_html=True para que NO salga el código, sino el diseño.
                st.markdown(html_final, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error técnico: {e}")
            finally:
                if os.path.exists(ruta_tmp):
                    os.remove(ruta_tmp)

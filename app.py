import streamlit as st
import librosa
import numpy as np
import whisper
import tempfile
import os

# --- 1. CONFIGURACIÓN VISUAL (ESTILO CANCIONERO PRO) ---
st.set_page_config(page_title="Cancionero Banda Pro", page_icon="🎸", layout="wide")

st.markdown("""
<style>
/* Estilo general limpio */
.main-container {
    background-color: #ffffff;
    padding: 30px;
    border-radius: 10px;
    font-family: 'Arial', sans-serif;
    line-height: 2.8; /* Mucho espacio para los acordes */
}

/* Contenedor de palabras */
.word-box {
    display: inline-flex;
    flex-direction: column;
    align-items: center; /* Centra el acorde sobre la palabra */
    margin-right: 6px; /* Espacio entre palabras */
    vertical-align: bottom;
    position: relative;
    height: 60px; /* Altura fija para alinear todo */
    justify-content: flex-end; /* Texto abajo */
}

/* El Acorde (Arriba y destacado) */
.chord-label {
    font-size: 14px;
    font-weight: 900;
    color: #0044cc; /* Azul profesional */
    margin-bottom: 2px;
    position: absolute;
    top: 5px; /* Fijado arriba */
    background-color: rgba(255,255,255,0.9);
    padding: 0 4px;
    border-radius: 4px;
    min-width: 20px;
    text-align: center;
}

/* La Palabra (Abajo y legible) */
.lyrics-label {
    font-size: 18px;
    color: #111;
    white-space: nowrap;
}

/* Secciones Instrumentales */
.instrumental-break {
    display: block;
    margin: 20px 0;
    padding: 10px;
    background-color: #f1f3f5;
    border-left: 4px solid #0044cc;
    color: #555;
    font-family: monospace;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎸 Transcriptor para Bandas (Precisión por Palabra)")
st.info("💡 Tip: Usando separación armónica para ignorar la batería y detectar mejores acordes.")

# --- 2. LÓGICA MUSICAL AVANZADA ---

def generar_templates_acordes():
    """Genera moldes matemáticos para acordes Mayores y Menores."""
    templates = {}
    notas = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    for i, nota in enumerate(notas):
        # Mayor (Raíz, +4, +7)
        vec_maj = np.roll(np.array([1,0,0,0,1,0,0,1,0,0,0,0]), i)
        templates[f"{nota}"] = vec_maj 
        
        # Menor (Raíz, +3, +7)
        vec_min = np.roll(np.array([1,0,0,1,0,0,0,1,0,0,0,0]), i)
        templates[f"{nota}m"] = vec_min
        
    return templates, notas

def detectar_acorde_preciso(chroma_segmento, templates):
    """Compara el audio con los moldes y devuelve el mejor candidato."""
    if chroma_segmento.shape[1] == 0: return ""
    
    # Promediamos el segmento de audio
    vector_audio = np.mean(chroma_segmento, axis=1)
    
    mejor_acorde = ""
    max_score = -float('inf')
    
    for nombre, template in templates.items():
        # Correlación matemática (similitud)
        score = np.dot(vector_audio, template)
        if score > max_score:
            max_score = score
            mejor_acorde = nombre
            
    return mejor_acorde

# --- 3. CARGA DE IA ---
@st.cache_resource
def cargar_whisper():
    # Usamos 'small' para balancear precisión de letra y memoria
    return whisper.load_model("small")

# --- 4. APP PRINCIPAL ---
archivo = st.file_uploader("Sube el audio de la banda (MP3/WAV)", type=["mp3", "wav"])

if archivo is not None:
    st.audio(archivo)
    
    if st.button("Transcribir Partitura"):
        with st.spinner("🥁 Separando instrumentos, analizando armonía y sincronizando letra... (Esto toma 2-3 min)"):
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(archivo.getvalue())
                ruta_tmp = tmp.name
            
            try:
                # A. PROCESAMIENTO DE AUDIO
                y, sr = librosa.load(ruta_tmp)
                
                # Separar armónicos (ignorar batería)
                y_harmonica, _ = librosa.effects.hpss(y)
                
                # Calcular CENS (Chroma Energy Normalized)
                chroma = librosa.feature.chroma_cens(y=y_harmonica, sr=sr)
                templates, _ = generar_templates_acordes()
                
                # B. TRANSCRIPCIÓN IA
                modelo = cargar_whisper()
                resultado = modelo.transcribe(
                    ruta_tmp, 
                    language="es", 
                    word_timestamps=True, # ¡CRUCIAL PARA UBICACIÓN EXACTA!
                    condition_on_previous_text=False
                )
                
                st.success("¡Análisis completado! Aquí está tu guía:")
                
                # --- CONSTRUCCIÓN DEL HTML (SIN ESPACIOS EXTRA) ---
                # Importante: Todo el HTML se concatena en una sola línea larga para evitar errores de renderizado.
                html_out = '<div class="main-container">'
                
                last_time = 0.0
                
                # Iteramos por segmentos
                for segmento in resultado['segments']:
                    words = segmento.get('words', [])
                    if not words: 
                        words = [{'word': segmento['text'], 'start': segmento['start'], 'end': segmento['end']}]
                    
                    for word_obj in words:
                        palabra = word_obj['word'].strip()
                        start = word_obj['start']
                        end = word_obj['end']
                        
                        # 1. DETECTAR SILENCIOS (Solo / Intro)
                        if start - last_time > 3.0: 
                            mid_start = int(last_time * sr / 512)
                            mid_end = int(start * sr / 512)
                            if mid_end > mid_start:
                                acorde_inst = detectar_acorde_preciso(chroma[:, mid_start:mid_end], templates)
                                # HTML COMPACTO (Sin saltos de linea dentro del string)
                                html_out += f'<div class="instrumental-break">🎵 Solo / Intro ({int(last_time)}s - {int(start)}s): <strong>{acorde_inst}</strong></div>'
                        
                        # 2. DETECTAR ACORDE DE LA PALABRA
                        idx_start = int(start * sr / 512)
                        idx_end = int(end * sr / 512)
                        if idx_end <= idx_start: idx_end = idx_start + 1
                            
                        acorde = detectar_acorde_preciso(chroma[:, idx_start:idx_end], templates)
                        
                        # 3. DIBUJAR CAJA (HTML COMPACTO)
                        html_out += f'<div class="word-box"><div class="chord-label">{acorde}</div><div class="lyrics-label">{palabra}</div></div>'
                        
                        last_time = end
                    
                    # Salto de línea al final de la frase
                    html_out += "<br>" 
                
                html_out += '</div>'
                
                # Renderizar diciéndole a Streamlit que es HTML seguro
                st.markdown(html_out, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Ocurrió un error técnico: {e}")
                st.warning("Nota: Asegúrate de que 'requirements.txt' tenga: openai-whisper>=20231117")
            finally:
                if os.path.exists(ruta_tmp):
                    os.remove(ruta_tmp)

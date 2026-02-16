import streamlit as st
import librosa
import numpy as np
import whisper
import tempfile
import os

# --- 1. CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="Cancionero Banda Pro", page_icon="🎸", layout="wide")

st.markdown("""
<style>
.main-container {
    background-color: #ffffff;
    padding: 30px;
    border-radius: 10px;
    font-family: 'Arial', sans-serif;
    line-height: 3.5; /* Altura de línea amplia para que quepan los acordes */
}
.word-box {
    display: inline-block; /* Comportamiento natural de texto */
    position: relative;
    margin-right: 5px;
    height: 50px;
    vertical-align: bottom;
}
.chord-label {
    font-size: 14px;
    font-weight: 900;
    color: #0044cc;
    position: absolute;
    top: -20px; /* Acorde flotando arriba */
    left: 0;
    white-space: nowrap;
}
.lyrics-label {
    font-size: 18px;
    color: #000;
}
.instrumental-break {
    display: block;
    margin: 30px 0;
    padding: 10px;
    background-color: #f1f3f5;
    border-left: 5px solid #0044cc;
    color: #555;
    font-family: monospace;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("🎸 Transcriptor Limpio (Solo cambios de Acorde)")

# --- 2. LÓGICA MUSICAL ---
def generar_templates_acordes():
    templates = {}
    notas = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    for i, nota in enumerate(notas):
        # Mayor
        vec_maj = np.roll(np.array([1,0,0,0,1,0,0,1,0,0,0,0]), i)
        templates[nota] = vec_maj 
        # Menor
        vec_min = np.roll(np.array([1,0,0,1,0,0,0,1,0,0,0,0]), i)
        templates[f"{nota}m"] = vec_min
    return templates

def detectar_acorde_preciso(chroma_segmento, templates):
    if chroma_segmento.shape[1] == 0: return ""
    vector_audio = np.mean(chroma_segmento, axis=1)
    mejor_acorde = ""
    max_score = -float('inf')
    for nombre, template in templates.items():
        score = np.dot(vector_audio, template)
        if score > max_score:
            max_score = score
            mejor_acorde = nombre
    return mejor_acorde

# --- 3. CARGA IA ---
@st.cache_resource
def cargar_whisper():
    return whisper.load_model("small")

# --- 4. APP PRINCIPAL ---
archivo = st.file_uploader("Sube el audio (MP3/WAV)", type=["mp3", "wav"])

if archivo is not None:
    st.audio(archivo)
    
    if st.button("Transcribir Cancionero"):
        with st.spinner("Procesando audio y letra..."):
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(archivo.getvalue())
                ruta_tmp = tmp.name
            
            try:
                # A. Audio
                y, sr = librosa.load(ruta_tmp)
                y_harmonica, _ = librosa.effects.hpss(y)
                chroma = librosa.feature.chroma_cens(y=y_harmonica, sr=sr)
                templates = generar_templates_acordes()
                
                # B. Letra
                modelo = cargar_whisper()
                resultado = modelo.transcribe(
                    ruta_tmp, 
                    language="es", 
                    word_timestamps=True, 
                    condition_on_previous_text=False
                )
                
                st.success("¡Transcripción lista!")
                
                # --- CONSTRUCCIÓN DEL HTML ---
                # Usamos una lista para evitar problemas de indentación
                html_parts = []
                html_parts.append('<div class="main-container">')
                
                last_time = 0.0
                acorde_anterior = None # Para recordar el último acorde tocado
                
                for segmento in resultado['segments']:
                    words = segmento.get('words', [])
                    if not words: 
                        words = [{'word': segmento['text'], 'start': segmento['start'], 'end': segmento['end']}]
                    
                    # Salto de línea visual entre frases
                    if html_parts[-1] != '<div class="main-container">':
                         html_parts.append("<br><br>")

                    for word_obj in words:
                        palabra = word_obj['word'].strip()
                        start = word_obj['start']
                        end = word_obj['end']
                        
                        # 1. Instrumental (Si hay hueco > 3s)
                        if start - last_time > 3.0: 
                            mid_start = int(last_time * sr / 512)
                            mid_end = int(start * sr / 512)
                            if mid_end > mid_start:
                                acorde_inst = detectar_acorde_preciso(chroma[:, mid_start:mid_end], templates)
                                # Forzamos mostrar el acorde instrumental siempre
                                html_parts.append(f'<div class="instrumental-break">🎵 Intro/Solo ({int(last_time)}s): {acorde_inst}</div>')
                                acorde_anterior = acorde_inst # Actualizamos contexto

                        # 2. Detectar acorde actual
                        idx_start = int(start * sr / 512)
                        idx_end = int(end * sr / 512)
                        if idx_end <= idx_start: idx_end = idx_start + 1
                        
                        acorde_actual = detectar_acorde_preciso(chroma[:, idx_start:idx_end], templates)
                        
                        # 3. LÓGICA DE LIMPIEZA (Solo mostrar si cambia)
                        if acorde_actual != acorde_anterior:
                            acorde_visual = acorde_actual
                            acorde_anterior = acorde_actual
                        else:
                            # Si es igual al anterior, ponemos espacio vacío invisible
                            acorde_visual = "&nbsp;" 

                        # 4. Construir bloque palabra
                        # NOTA: Todo en una sola linea f-string para evitar que Streamlit lo detecte como código
                        bloque = f'<div class="word-box"><div class="chord-label">{acorde_visual}</div><div class="lyrics-label">{palabra}</div></div>'
                        html_parts.append(bloque)
                        
                        last_time = end
                
                html_parts.append('</div>')
                
                # Renderizar uniendo todo sin saltos de línea extraños
                st.markdown("".join(html_parts), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                if os.path.exists(ruta_tmp):
                    os.remove(ruta_tmp)

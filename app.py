import streamlit as st
import librosa
import numpy as np
import whisper
import tempfile
import os

# --- CONFIGURACIÓN VISUAL (ESTILO CANCIONERO PRO) ---
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
    background-color: rgba(255,255,255,0.8);
    padding: 0 2px;
    border-radius: 4px;
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
    background-color: #f8f9fa;
    border-left: 4px solid #0044cc;
    color: #555;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)

st.title("🎸 Transcriptor para Bandas (Precisión por Palabra)")
st.info("💡 Tip: Usando separación armónica para ignorar la batería y detectar mejores acordes.")

# --- LÓGICA MUSICAL AVANZADA ---

def generar_templates_acordes():
    """Genera moldes matemáticos para acordes Mayores y Menores."""
    templates = {}
    notas = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    for i, nota in enumerate(notas):
        # Mayor (Raíz, +4, +7)
        vec_maj = np.roll(np.array([1,0,0,0,1,0,0,1,0,0,0,0]), i)
        templates[f"{nota}"] = vec_maj # Cifrado simple (C, D) para mayor
        
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

# --- CARGA DE IA ---
@st.cache_resource
def cargar_whisper():
    # Usamos 'small' para balancear precisión de letra y memoria
    return whisper.load_model("small")

# --- APP PRINCIPAL ---
archivo = st.file_uploader("Sube el audio de la banda (MP3/WAV)", type=["mp3", "wav"])

if archivo is not None:
    st.audio(archivo)
    
    if st.button("Transcribir Partitura"):
        with st.spinner("🥁 Separando instrumentos, analizando armonía y sincronizando letra... (Esto toma 2-3 min)"):
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(archivo.getvalue())
                ruta_tmp = tmp.name
            
            try:
                # 1. PROCESAMIENTO DE AUDIO (Heavy Lifting)
                # Cargamos audio
                y, sr = librosa.load(ruta_tmp)
                
                # TRUCO PRO: Separar la parte armónica (piano/guitarra) de la percusiva (batería)
                # Esto evita que el golpe de caja confunda al detector de acordes.
                y_harmonica, _ = librosa.effects.hpss(y)
                
                # Calculamos CENS (Chroma Energy Normalized) - Mejor para acordes estables
                chroma = librosa.feature.chroma_cens(y=y_harmonica, sr=sr)
                templates, _ = generar_templates_acordes()
                
                # 2. TRANSCRIPCIÓN CON TIMESTAMPS POR PALABRA
                modelo = cargar_whisper()
                # word_timestamps=True es la clave para la ubicación exacta
                resultado = modelo.transcribe(
                    ruta_tmp, 
                    language="es", 
                    word_timestamps=True, # ¡CRUCIAL!
                    condition_on_previous_text=False # Evita bucles
                )
                
                st.success("¡Análisis completado! Aquí está tu guía:")
                
                # 3. RENDERIZADO VISUAL
                html_out = '<div class="main-container">'
                
                last_time = 0.0
                
                # Iteramos por segmentos y LUEGO por palabras
                for segmento in resultado['segments']:
                    words = segmento.get('words', [])
                    
                    # Si Whisper no devuelve palabras (versiones viejas), fallback a segmento
                    if not words: 
                        words = [{'word': segmento['text'], 'start': segmento['start'], 'end': segmento['end']}]
                    
                    for word_obj in words:
                        palabra = word_obj['word'].strip()
                        start = word_obj['start']
                        end = word_obj['end']
                        
                        # A. DETECTAR SILENCIOS LARGOS (Instrumental)
                        if start - last_time > 3.0: # Más de 3 seg de silencio
                            # Analizamos el centro de ese silencio
                            mid_start = int(last_time * sr / 512)
                            mid_end = int(start * sr / 512)
                            if mid_end > mid_start:
                                # Tomamos una muestra del chroma en ese hueco
                                acorde_inst = detectar_acorde_preciso(chroma[:, mid_start:mid_end], templates)
                                html_out += f"""
                                <div class="instrumental-break">
                                    🎵 Solo / Intro ({int(last_time)}s - {int(start)}s): <strong>{acorde_inst}</strong>
                                </div>
                                """
                        
                        # B. DETECTAR ACORDE DE LA PALABRA
                        # Convertimos tiempo a "frames" del chroma
                        # librosa CENS suele tener hop_length=512 por defecto
                        idx_start = int(start * sr / 512)
                        idx_end = int(end * sr / 512)
                        
                        # Si la palabra es muy corta, tomamos un margen mínimo
                        if idx_end <= idx_start:
                            idx_end = idx_start + 1
                            
                        acorde = detectar_acorde_preciso(chroma[:, idx_start:idx_end], templates)
                        
                        # Verificar si el acorde cambió respecto al anterior para no repetirlo tanto?
                        # Para bandas, mejor mostrarlo siempre si hay dudas.
                        
                        # C. DIBUJAR CAJA
                        html_out += f"""
                        <div class="word-box">
                            <div class="chord-label">{acorde}</div>
                            <div class="lyrics-label">{palabra}</div>
                        </div>
                        """
                        
                        last_time = end
                    
                    # Salto de línea visual al terminar una frase completa
                    html_out += "<br>" 
                
                html_out += '</div>'
                st.markdown(html_out, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Ocurrió un error técnico: {e}")
                st.warning("Asegúrate de haber actualizado 'requirements.txt' con la versión de whisper indicada.")
            finally:
                if os.path.exists(ruta_tmp):
                    os.remove(ruta_tmp)

import streamlit as st
import librosa
import numpy as np
import whisper
import tempfile
import os

# --- 1. Configuración Visual (CSS Robusto) ---
st.set_page_config(page_title="Cancionero Pro 2.0", page_icon="🎵", layout="wide")

# CSS: Definimos el estilo para que se vea profesional y forzamos la alineación
st.markdown("""
<style>
.cancionero-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    line-height: 2.2;
    font-family: 'Segoe UI', sans-serif;
    padding: 30px;
    background-color: #ffffff;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}
.bloque-palabra {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-right: 4px;
    margin-bottom: 25px;
}
.acorde-style {
    color: #007bff;
    font-weight: 900;
    font-size: 15px;
    height: 20px;
    margin-bottom: 4px;
    min-width: 20px;
    text-align: center;
}
.letra-style {
    color: #111;
    font-size: 19px;
    letter-spacing: 0.5px;
}
.bloque-instrumental {
    display: flex;
    flex-direction: column;
    align-items: center;
    background-color: #f1f3f5;
    padding: 4px 12px;
    border-radius: 6px;
    border: 1px solid #dee2e6;
    margin-right: 10px;
    margin-bottom: 25px;
}
.texto-instrumental {
    font-size: 11px;
    color: #868e96;
    font-weight: bold;
    text-transform: uppercase;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎹 Transcriptor Pro (Modelo Small + Anti-Alucinaciones)")
st.info("Usando modelo 'SMALL'. Puede tardar un poco más, pero la calidad es superior.")

# --- 2. Funciones Musicales ---
def obtener_nombre_acorde(chroma_mean):
    notas = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    idx = np.argmax(chroma_mean)
    nota = notas[idx]
    
    tercera_mayor = (idx + 4) % 12
    tercera_menor = (idx + 3) % 12
    
    # Ajuste de sensibilidad para acordes menores
    if chroma_mean[tercera_menor] > chroma_mean[tercera_mayor] * 1.05:
        return f"{nota}m"
    return nota

def analizar_segmento(y, sr):
    if len(y) == 0: return ""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    promedio = np.mean(chroma, axis=1)
    return obtener_nombre_acorde(promedio)

# --- 3. Carga IA (MODELO SMALL) ---
@st.cache_resource
def cargar_whisper():
    # 'small' es el equilibrio perfecto. 'medium' suele colapsar la memoria gratis.
    return whisper.load_model("small") 

# --- 4. Aplicación Principal ---
archivo = st.file_uploader("Sube tu audio (MP3/WAV)", type=["mp3", "wav"])

if archivo is not None:
    st.audio(archivo)
    
    if st.button("Analizar con Precisión"):
        with st.spinner("🧠 Cerebro 'Small' pensando... (Esto toma unos 2 minutos, paciencia)..."):
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(archivo.getvalue())
                ruta_tmp = tmp.name
            
            try:
                # A. Cargar Audio Musical
                y, sr = librosa.load(ruta_tmp)
                
                # B. Transcribir con PARAMETROS ANTI-ALUCINACIÓN
                modelo = cargar_whisper()
                
                # Estos parámetros evitan que la IA invente texto cuando hay solo música
                opciones = {
                    "language": "es", 
                    "temperature": 0.0, # Creatividad 0 para ser fiel
                    "condition_on_previous_text": False, # Evita bucles de repetición
                    "no_speech_threshold": 0.6, # Ignora ruidos bajos
                    "initial_prompt": "Esta es la letra de una canción en español." # Contexto forzado
                }
                
                resultado = modelo.transcribe(ruta_tmp, **opciones)
                segmentos = resultado['segments']
                
                st.success("¡Análisis Completado!")
                st.divider()
                
                # --- C. CONSTRUCCIÓN HTML COMPACTA (Solución al error visual) ---
                # Importante: No dejar espacios ni saltos de línea dentro del string HTML
                html_code = '<div class="cancionero-container">'
                cursor_tiempo = 0.0
                
                for seg in segmentos:
                    inicio = seg['start']
                    fin = seg['end']
                    texto_frase = seg['text'].strip()
                    
                    # 1. DETECTAR PARTES INSTRUMENTALES (GAPS > 2 seg)
                    if inicio - cursor_tiempo > 2.0:
                        idx_ini_gap = int(cursor_tiempo * sr)
                        idx_fin_gap = int(inicio * sr)
                        
                        if idx_fin_gap > idx_ini_gap:
                            acorde_gap = analizar_segmento(y[idx_ini_gap:idx_fin_gap], sr)
                            # HTML en una sola línea para evitar errores
                            html_code += f'<div class="bloque-instrumental"><div class="acorde-style">{acorde_gap}</div><div class="texto-instrumental">Música</div></div>'
                    
                    # 2. PROCESAR FRASE CANTADA
                    idx_ini = int(inicio * sr)
                    idx_fin = int(fin * sr)
                    acorde_voz = analizar_segmento(y[idx_ini:idx_fin], sr)
                    
                    palabras = texto_frase.split(" ")
                    
                    for i, palabra in enumerate(palabras):
                        # Acorde solo en la primera palabra
                        acorde_vis = acorde_voz if i == 0 else "&nbsp;"
                        if palabra: # Solo si la palabra no está vacía
                            # HTML en una sola línea
                            html_code += f'<div class="bloque-palabra"><div class="acorde-style">{acorde_vis}</div><div class="letra-style">{palabra}</div></div>'
                    
                    cursor_tiempo = fin
                
                # Rellenar final si hay outro musical
                dur_total = librosa.get_duration(y=y, sr=sr)
                if dur_total - cursor_tiempo > 2.0:
                     idx_ini_end = int(cursor_tiempo * sr)
                     acorde_end = analizar_segmento(y[idx_ini_end:], sr)
                     html_code += f'<div class="bloque-instrumental"><div class="acorde-style">{acorde_end}</div><div class="texto-instrumental">Final</div></div>'

                html_code += '</div>'
                
                # RENDERIZADO FINAL
                st.markdown(html_code, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error técnico: {e}")
                st.warning("Si falla por memoria, intenta con una canción de menos de 4 minutos.")
            finally:
                if os.path.exists(ruta_tmp):
                    os.remove(ruta_tmp)

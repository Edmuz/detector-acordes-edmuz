import streamlit as st
import librosa
import numpy as np
import whisper
import tempfile
import os

# --- Configuración Visual ---
st.set_page_config(page_title="Cancionero IA", page_icon="🎸", layout="centered")

# CSS para que se vea como un cancionero real (Acordes azules sobre texto)
st.markdown("""
    <style>
    .cancionero {
        font-family: 'Courier New', monospace; /* Fuente tipo máquina de escribir para alinear */
        white-space: pre-wrap;
        line-height: 2.5; /* Espacio para que entre el acorde arriba */
        font-size: 16px;
        color: #333;
    }
    .bloque {
        display: inline-block;
        position: relative;
        margin-right: 5px;
        margin-bottom: 10px;
    }
    .acorde {
        position: absolute;
        top: -20px; /* Sube el acorde */
        left: 0;
        color: #007bff; /* Azul intenso */
        font-weight: bold;
        font-size: 14px;
    }
    .letra {
        display: inline;
    }
    .instrumental {
        color: #888;
        font-style: italic;
        border-bottom: 1px dashed #ccc;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎸 Transcriptor de Canciones (Español)")

# --- Lógica Musical ---
def obtener_nombre_acorde(chroma_mean):
    """Convierte vectores matemáticos a nombres de acordes (C, Dm, etc.)"""
    notas = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    idx = np.argmax(chroma_mean)
    nota = notas[idx]
    
    # Detección simple Mayor/Menor
    tercera_mayor = (idx + 4) % 12
    tercera_menor = (idx + 3) % 12
    if chroma_mean[tercera_menor] > chroma_mean[tercera_mayor] * 1.1: # Umbral ligero
        return f"{nota}m"
    return nota

def analizar_segmento(y, sr):
    if len(y) == 0: return ""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    promedio = np.mean(chroma, axis=1)
    return obtener_nombre_acorde(promedio)

# --- Cargar IA ---
@st.cache_resource
def cargar_whisper():
    return whisper.load_model("tiny") # Usa 'base' si quieres más precisión (pero es más lento)

# --- App Principal ---
archivo = st.file_uploader("Sube tu MP3/WAV", type=["mp3", "wav"])

if archivo is not None:
    st.audio(archivo)
    
    if st.button("Generar Cancionero"):
        with st.spinner("🎧 Escuchando (Español) y sacando acordes..."):
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(archivo.getvalue())
                ruta_tmp = tmp.name
            
            try:
                # 1. Cargar Audio
                y, sr = librosa.load(ruta_tmp)
                duracion_total = librosa.get_duration(y=y, sr=sr)
                
                # 2. Transcribir Letra (FORZANDO ESPAÑOL)
                modelo = cargar_whisper()
                # Aquí forzamos el idioma español 'es'
                resultado = modelo.transcribe(ruta_tmp, language="es") 
                segmentos = resultado['segments']
                
                # 3. Rellenar Huecos (Intro, Intermedios, Outro)
                linea_tiempo = [] # Lista final de eventos
                cursor_tiempo = 0.0
                
                for seg in segmentos:
                    inicio = seg['start']
                    fin = seg['end']
                    texto = seg['text'].strip()
                    
                    # A. ¿Hay un hueco grande antes de esta frase? (Intro o Intermedio)
                    if inicio - cursor_tiempo > 2.0: 
                        # Es música instrumental
                        duracion_gap = inicio - cursor_tiempo
                        # Si es muy largo, sacamos varios acordes (cada 2 seg)
                        pasos = int(duracion_gap // 2) or 1
                        for i in range(pasos):
                            t_sub_inicio = cursor_tiempo + (i*2)
                            t_sub_fin = min(t_sub_inicio + 2, inicio)
                            
                            idx_ini = int(t_sub_inicio * sr)
                            idx_fin = int(t_sub_fin * sr)
                            acorde_inst = analizar_segmento(y[idx_ini:idx_fin], sr)
                            
                            linea_tiempo.append({
                                'tipo': 'instr',
                                'acorde': acorde_inst,
                                'texto': '▬' # Símbolo de música
                            })
                    
                    # B. Analizar la frase cantada
                    idx_ini = int(inicio * sr)
                    idx_fin = int(fin * sr)
                    acorde_voz = analizar_segmento(y[idx_ini:idx_fin], sr)
                    
                    linea_tiempo.append({
                        'tipo': 'voz',
                        'acorde': acorde_voz,
                        'texto': texto
                    })
                    
                    cursor_tiempo = fin

                # 4. Chequear el Final (Outro)
                if duracion_total - cursor_tiempo > 2.0:
                    idx_ini = int(cursor_tiempo * sr)
                    acorde_final = analizar_segmento(y[idx_ini:], sr)
                    linea_tiempo.append({'tipo': 'instr', 'acorde': acorde_final, 'texto': '(Final)'})

                # 5. Renderizar HTML Bonito
                st.success("¡Transcripción Completa!")
                st.markdown("---")
                
                html_salida = '<div class="cancionero">'
                for evento in linea_tiempo:
                    acorde = evento['acorde']
                    texto = evento['texto']
                    clase_extra = " instrumental" if evento['tipo'] == 'instr' else ""
                    
                    # Creamos el bloque HTML: Acorde flotando sobre Texto
                    html_salida += f"""
                    <div class="bloque{clase_extra}">
                        <div class="acorde">{acorde}</div>
                        <div class="letra">{texto}</div>
                    </div>
                    """
                html_salida += '</div>'
                
                st.markdown(html_salida, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                os.remove(ruta_tmp)

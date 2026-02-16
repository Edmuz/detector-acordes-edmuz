import streamlit as st
import librosa
import numpy as np
import whisper
import tempfile
import os

# --- 1. Configuración Visual (CSS Mejorado) ---
st.set_page_config(page_title="Cancionero Pro", page_icon="🎹", layout="wide")

st.markdown("""
    <style>
    /* Contenedor principal: estilo hoja de papel */
    .cancionero-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px; /* Espacio entre sílabas/palabras */
        line-height: 2.0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding: 30px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Bloque normal: Acorde + Palabra */
    .bloque-palabra {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-right: 2px;
        margin-bottom: 20px; /* Espacio entre renglones */
    }

    /* Estilo del Acorde (Azul fuerte) */
    .acorde-style {
        color: #007bff; 
        font-weight: 800; /* Más negrita */
        font-size: 15px;
        height: 20px;
        margin-bottom: 2px;
        min-width: 20px; /* Para que siempre ocupe espacio */
        text-align: center;
    }

    /* Estilo de la Letra (Negro) */
    .letra-style {
        color: #222;
        font-size: 18px;
    }

    /* Bloque para partes SOLO MÚSICA */
    .bloque-instrumental {
        display: flex;
        flex-direction: column;
        align-items: center;
        background-color: #f0f0f0; /* Fondo grisacio para diferenciar */
        padding: 0 8px;
        border-radius: 4px;
        border: 1px dashed #bbb;
        margin-right: 8px;
        margin-bottom: 20px;
    }
    .texto-instrumental {
        font-size: 12px;
        color: #666;
        font-style: italic;
        margin-top: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎹 Transcriptor Fiel (Letra + Acordes)")
st.info("Ahora usando el modelo 'BASE' para mayor precisión en la letra.")

# --- 2. Funciones Musicales ---
def obtener_nombre_acorde(chroma_mean):
    notas = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    idx = np.argmax(chroma_mean)
    nota = notas[idx]
    
    # Lógica Mayor/Menor
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

# --- 3. Carga IA (CAMBIO IMPORTANTE AQUÍ) ---
@st.cache_resource
def cargar_whisper():
    # CAMBIO: Usamos "base" en lugar de "tiny". 
    # "base" es mucho más inteligente y no alucina tanto, aunque tarda un pelín más.
    return whisper.load_model("base")

# --- 4. Aplicación Principal ---
archivo = st.file_uploader("Sube tu audio (MP3/WAV)", type=["mp3", "wav"])

if archivo is not None:
    st.audio(archivo)
    
    if st.button("Analizar Canción"):
        with st.spinner("🎧 Escuchando con atención (Modelo Base)... Por favor espera..."):
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(archivo.getvalue())
                ruta_tmp = tmp.name
            
            try:
                # 1. Cargar Audio Musical
                y, sr = librosa.load(ruta_tmp)
                
                # 2. Transcribir Letra (IA Mejorada)
                modelo = cargar_whisper()
                # temperature=0 reduce la creatividad (evita invenciones)
                resultado = modelo.transcribe(ruta_tmp, language="es", temperature=0)
                segmentos = resultado['segments']
                
                st.success("¡Transcripción completada con éxito!")
                st.markdown("---")
                
                # --- CONSTRUCCIÓN VISUAL ---
                html_final = '<div class="cancionero-container">'
                cursor_tiempo = 0.0
                
                for seg in segmentos:
                    inicio = seg['start']
                    fin = seg['end']
                    texto_frase = seg['text'].strip()
                    
                    # A. DETECTAR MÚSICA/SILENCIO (Gaps de más de 2 segundos)
                    if inicio - cursor_tiempo > 2.0:
                        # Analizamos qué acorde suena en ese silencio
                        idx_ini_gap = int(cursor_tiempo * sr)
                        idx_fin_gap = int(inicio * sr)
                        
                        if idx_fin_gap > idx_ini_gap:
                            # Sacamos el acorde predominante de esa sección musical
                            acorde_gap = analizar_segmento(y[idx_ini_gap:idx_fin_gap], sr)
                            
                            # Lo añadimos como un bloque especial "Instrumental"
                            html_final += f"""
                            <div class="bloque-instrumental">
                                <div class="acorde-style">{acorde_gap}</div>
                                <div class="texto-instrumental">Música</div>
                            </div>
                            """
                    
                    # B. PROCESAR LA FRASE CANTADA
                    idx_ini = int(inicio * sr)
                    idx_fin = int(fin * sr)
                    acorde_voz = analizar_segmento(y[idx_ini:idx_fin], sr)
                    
                    palabras = texto_frase.split(" ")
                    
                    for i, palabra in enumerate(palabras):
                        # Ponemos el acorde en la primera palabra de la frase
                        acorde_mostrar = acorde_voz if i == 0 else "&nbsp;"
                        
                        # Bloque normal de letra
                        html_final += f"""
                        <div class="bloque-palabra">
                            <div class="acorde-style">{acorde_mostrar}</div>
                            <div class="letra-style">{palabra}</div>
                        </div>
                        """
                    
                    cursor_tiempo = fin
                
                html_final += '</div>'
                
                # Renderizar (Mostrar en pantalla)
                st.markdown(html_final, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Ocurrió un error: {e}")
                st.warning("Si la app se reinicia, es posible que el archivo sea muy pesado para la versión gratuita.")
            finally:
                if os.path.exists(ruta_tmp):
                    os.remove(ruta_tmp)

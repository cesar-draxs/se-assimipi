import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

# Configuración de la página
st.set_page_config(
    page_title="Señas Simipi T'ikraq",
    page_icon="🖐️",
    layout="wide"
)

# Estilos personalizados (Tema Oscuro y Accesibilidad)
st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        color: #1E88E5;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 20px;
        color: #555;
        text-align: center;
        margin-bottom: 25px;
    }
    .word-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: #1E88E5;
        border: 2px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🖐️ Señas Simipi T\'ikraq</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Traductor Inteligente de Lenguaje de Señas Bimanual</div>', unsafe_allow_html=True)

# Diccionario de 100 palabras cotidianas clasificadas
diccionario = {
    "Saludos y Cortesía": ["Hola", "Adiós", "Buenos días", "Buenas tardes", "Buenas noches", "Por favor", "Gracias", "De nada", "Lo siento", "Disculpe", "Bienvenido", "¿Cómo estás?"],
    "Familia y Personas": ["Mamá", "Papá", "Hijo", "Hija", "Hermano", "Hermana", "Abuelo", "Abuela", "Amigo", "Tío", "Tía", "Primo", "Yo", "Tú", "Nosotros", "Familia"],
    "Necesidades y Emergencias": ["Ayuda", "Hospital", "Médico", "Policía", "Peligro", "Baño", "Agua", "Comida", "Medicina", "Dolor", "Urgente", "Casa"],
    "Preguntas Frecuentes": ["Qué", "Quién", "Cuándo", "Dónde", "Por qué", "Cómo", "Cuánto", "Cuál"],
    "Alimentos y Bebidas": ["Pan", "Leche", "Carne", "Fruta", "Arroz", "Huevo", "Manzana", "Agua", "Jugo", "Desayuno", "Almuerzo", "Cena"],
    "Emociones y Sentimientos": ["Feliz", "Triste", "Enojado", "Cansado", "Enfermo", "Asustado", "Bien", "Mal", "Amor", "Preocupado", "Sorprendido"],
    "Tiempo y Calendario": ["Hoy", "Ayer", "Mañana", "Ahora", "Después", "Año", "Mes", "Día", "Hora", "Semana", "Lunes", "Domingo"],
    "Colores y Objetos": ["Rojo", "Azul", "Amarillo", "Verde", "Blanco", "Negro", "Libro", "Celular", "Llave", "Dinero", "Escuela", "Trabajo"],
    "Respuestas Básicas": ["Sí", "No", "Entiendo", "No entiendo", "Claro", "Tal vez", "Correcto", "Incorrecto"]
}

# Mostrar diccionario en la barra lateral
st.sidebar.title("📚 Vocabulario (100 Palabras)")
for categoria, palabras in diccionario.items():
    with st.sidebar.expander(categoria):
        st.write(", ".join(palabras))

# Inicializar estados de la frase e historial
if 'frase' not in st.session_state:
    st.session_state.frase = []

col1, col2 = st.columns()

with col1:
    st.write("### 🎥 Entrada de Video")
    run_cam = st.checkbox("Activar Cámara", value=False)
    FRAME_WINDOW = st.image([])

with col2:
    st.write("### 📝 Traducción en Tiempo Real")
    palabra_actual_placeholder = st.empty()
    palabra_actual_placeholder.markdown('<div class="word-box">Esperando seña...</div>', unsafe_allow_html=True)
    
    st.write("#### 💬 Frase Acumulada:")
    frase_placeholder = st.empty()
    frase_placeholder.info("Las palabras detectadas se irán sumando aquí.")
    
    if st.button("Limpiar Frase 🗑️"):
        st.session_state.frase = []
        frase_placeholder.info("Las palabras detectadas se irán sumando aquí.")

# Configuración de MediaPipe para rastreo de dos manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Aplanamos la lista de palabras para simular la detección
todas_las_palabras = [p for cat in diccionario.values() for p in cat]

if run_cam:
    cap = cv2.VideoCapture(0)
    last_word = ""
    stable_counter = 0
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:
        
        while run_cam:
            ret, frame = cap.read()
            if not ret:
                st.error("No se pudo acceder a la cámara web.")
                break
                
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            detected_word = None
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Algoritmo bimanual simulado basado en cantidad de manos en pantalla
                num_manos = len(results.multi_hand_landmarks)
                
                if num_manos == 2:
                    l_hand = results.multi_hand_landmarks[0].landmark
                    r_hand = results.multi_hand_landmarks[1].landmark
                    distancia = np.sqrt((l_hand[0].x - r_hand[0].x)**2 + (l_hand[0].y - r_hand[0].y)**2)
                    
                    if distancia < 0.2:
                        detected_word = "Gracias"
                    elif l_hand[12].y < 0.4 and r_hand[12].y < 0.4:
                        detected_word = "Ayuda"
                    else:
                        detected_word = "Hola"
                elif num_manos == 1:
                    thumb_tip = results.multi_hand_landmarks[0].landmark[4]
                    index_tip = results.multi_hand_landmarks[0].landmark[8]
                    if thumb_tip.y < index_tip.y:
                        detected_word = "Bien"
                    else:
                        detected_word = "No"
            
            # Filtro de estabilidad para evitar parpadeos en la detección
            if detected_word:
                if detected_word == last_word:
                    stable_counter += 1
                else:
                    stable_counter = 0
                    last_word = detected_word
                
                if stable_counter > 15:  # Debe mantenerse estable por 15 fotogramas
                    palabra_actual_placeholder.markdown(f'<div class="word-box">{detected_word}</div>', unsafe_allow_html=True)
                    if not st.session_state.frase or st.session_state.frase[-1] != detected_word:
                        st.session_state.frase.append(detected_word)
                        frase_placeholder.success(" ".join(st.session_state.frase))
                    stable_counter = 0
            else:
                stable_counter = 0
                palabra_actual_placeholder.markdown('<div class="word-box">Esperando seña...</div>', unsafe_allow_html=True)
            
            # Dibujar cantidad de manos detectadas en el feed de video
            cv2.putText(frame, f"Manos: {len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            FRAME_WINDOW.image(frame, channels="BGR")
            
        cap.release()
else:
    FRAME_WINDOW.info("Haz clic en 'Activar Cámara' para iniciar el traductor.")

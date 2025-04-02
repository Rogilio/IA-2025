import cv2
import mediapipe as mp
import numpy as np
import time

# Inicialización de MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,  # Se analiza una cara por vez
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Inicialización de MediaPipe Drawing (opcional para dibujar landmarks)
mp_drawing = mp.solutions.drawing_utils

# Captura de video
cap = cv2.VideoCapture(0)

# Lista de índices de landmarks seleccionados para análisis (ejemplo: ojos y boca)
selected_points = {
    'ojo_izq_interno': 33,
    'ojo_izq_externo': 133,
    'ojo_der_interno': 362,
    'ojo_der_externo': 263,
    'boca_superior': 13,    # Ejemplo, se puede ajustar según la documentación
    'boca_inferior': 14,    # Ejemplo, se puede ajustar según la documentación
    # Se pueden agregar más puntos de interés para el análisis de emociones y gestos
}

def distancia(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Variables para detección de movimiento y vivacidad
tiempo_inicial = time.time()
contador_blink = 0
umbral_blink = 5  # Número mínimo de parpadeos en un intervalo para considerar 'vivacidad'
historial_boca = []  # Para almacenar valores de apertura de boca

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Voltear la imagen para dar sensación de espejo
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    puntos = {}  # Diccionario para almacenar coordenadas de los puntos seleccionados

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extraer los puntos de interés
            for nombre, idx in selected_points.items():
                x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                puntos[nombre] = (x, y)
                # Dibujar el punto en la imagen para visualización
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Ejemplo: Medición de distancia entre ojos para analizar simetría y posible parpadeo
            if 'ojo_izq_interno' in puntos and 'ojo_der_interno' in puntos:
                d_ojos = distancia(puntos['ojo_izq_interno'], puntos['ojo_der_interno'])
                cv2.putText(frame, f"Dist Ojos: {int(d_ojos)}", (puntos['ojo_izq_interno'][0], puntos['ojo_izq_interno'][1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Ejemplo: Detección de movimiento de boca (apertura de boca)
            if 'boca_superior' in puntos and 'boca_inferior' in puntos:
                apertura_boca = distancia(puntos['boca_superior'], puntos['boca_inferior'])
                historial_boca.append(apertura_boca)
                cv2.putText(frame, f"Apertura Boca: {int(apertura_boca)}", (puntos['boca_superior'][0], puntos['boca_superior'][1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # Se puede establecer un umbral dinámico o fijo para determinar si la persona habla o mueve la boca
                if apertura_boca > 20:  # Valor de umbral experimental, se debe ajustar
                    cv2.putText(frame, "Moviendo boca", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Detección de parpadeo (como indicador de vivacidad)
            # Se puede analizar la variación temporal en la apertura de los ojos
            # NOTA: Esta parte requiere definir correctamente las medidas para el EAR (Eye Aspect Ratio) o similar.
            # Aquí se deja un comentario para indicar dónde integrarlo.
            # Por ejemplo, si la relación entre distancias verticales y horizontales de los ojos cae por debajo de un umbral, se cuenta como parpadeo.
            # if ear < umbral_ear:
            #     contador_blink += 1

            # Detección de emociones (conceptual)
            # Se pueden usar relaciones angulares y distancias entre puntos clave para inferir expresiones.
            # Por ejemplo: si la distancia entre las cejas disminuye y la boca se curva hacia abajo,
            # se puede inferir tristeza. Se recomienda usar modelos de clasificación entrenados para mayor precisión.
            # Aquí se muestra un ejemplo simplificado:
            emocion_detectada = "Neutral"
            # Valores de ejemplo basados en aperturas y posiciones (se deben calibrar):
            if 'boca_superior' in puntos and 'boca_inferior' in puntos:
                if apertura_boca > 30:
                    emocion_detectada = "Feliz o Hablando"
                elif apertura_boca < 10:
                    emocion_detectada = "Serio"
            cv2.putText(frame, f"Emocion: {emocion_detectada}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Se pueden agregar más análisis para otros gestos y expresiones utilizando otros landmarks

    # Mostrar la imagen resultante
    cv2.imshow('Deteccion Facial con MediaPipe', frame)
    
    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

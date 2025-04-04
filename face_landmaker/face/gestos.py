import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# 1. Configuración inicial de MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,  # Habilita landmarks precisos de ojos y labios
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. Definición de landmarks clave
EYE_LEFT = [33, 133, 145, 159, 263, 362, 386, 374]
EYE_RIGHT = [362, 263, 386, 374, 33, 133, 145, 159]
MOUTH = [61, 291, 13, 14, 78, 308, 87, 178]
EYEBROWS = [55, 65, 70, 105, 285, 295, 300, 336]
selected_points = EYE_LEFT + EYE_RIGHT + MOUTH + EYEBROWS  # Corregido EYBROWS -> EYEBROWS

# 3. Constantes de análisis
EAR_THRESHOLD = 0.25    # Umbral para detección de parpadeo
MAR_THRESHOLD = 0.5     # Umbral para detección de apertura bucal
HISTORY_SIZE = 10       # Tamaño del buffer para detección temporal

# 4. Variables de estado globales
eye_history = deque(maxlen=HISTORY_SIZE)
mouth_history = deque(maxlen=HISTORY_SIZE)

def distancia(p1, p2):
    """Calcula la distancia euclidiana entre dos puntos normalizados."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calcular_ear(landmarks, eye_points):
    """Calcula el Eye Aspect Ratio (EAR) para detección de parpadeos."""
    # Extracción de coordenadas normalizadas
    p1 = [landmarks[eye_points[0]].x, landmarks[eye_points[0]].y]
    p2 = [landmarks[eye_points[1]].x, landmarks[eye_points[1]].y]
    p3 = [landmarks[eye_points[2]].x, landmarks[eye_points[2]].y]
    p4 = [landmarks[eye_points[3]].x, landmarks[eye_points[3]].y]
    p5 = [landmarks[eye_points[4]].x, landmarks[eye_points[4]].y]
    p6 = [landmarks[eye_points[5]].x, landmarks[eye_points[5]].y]
    
    # Fórmula EAR estándar
    vert_dist = distancia(p2, p6) + distancia(p3, p5)
    horiz_dist = 2 * distancia(p1, p4)
    return vert_dist / horiz_dist if horiz_dist != 0 else 0.0

def calcular_mar(landmarks):
    """Calcula el Mouth Aspect Ratio (MAR) para apertura bucal."""
    vertical = distancia(
        [landmarks[13].x, landmarks[13].y],
        [landmarks[14].x, landmarks[14].y]
    )
    horizontal = distancia(
        [landmarks[61].x, landmarks[61].y],
        [landmarks[291].x, landmarks[291].y]
    )
    return vertical / horizontal if horizontal != 0 else 0.0

def analizar_emocion(landmarks):
    """Determina emoción basada en geometría facial usando múltiples parámetros."""
    # 1. Posición de las cejas
    ceja_izq = distancia([landmarks[65].x, landmarks[65].y], 
                        [landmarks[159].x, landmarks[159].y])
    ceja_der = distancia([landmarks[295].x, landmarks[295].y], 
                        [landmarks[386].x, landmarks[386].y])
    
    # 2. Curvatura de la boca
    boca_curva = landmarks[14].y - landmarks[78].y  # Diferencia vertical
    
    # 3. Apertura ocular
    apertura_ojo_izq = calcular_ear(landmarks, EYE_LEFT)
    apertura_ojo_der = calcular_ear(landmarks, EYE_RIGHT)
    
    # Lógica de detección combinada
    if boca_curva > 0.03 and apertura_ojo_izq > 0.2 and apertura_ojo_der > 0.2:
        return "😊 Feliz"
    elif ceja_izq > 0.07 and ceja_der > 0.07:
        return "😯 Sorpresa"
    elif ceja_izq < 0.03 and ceja_der < 0.03 and boca_curva < -0.01:
        return "😠 Enojo"
    elif boca_curva < -0.02 and apertura_ojo_izq < 0.2 and apertura_ojo_der < 0.2:
        return "😢 Triste"
    else:
        return "😐 Neutral"

# 5. Configuración de captura de video
cap = cv2.VideoCapture(0)
cv2.namedWindow('Sistema de Analisis Facial', cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
            landmarks = face_landmarks.landmark
            
            # A. Procesamiento de landmarks
            puntos = {}
            for idx in selected_points:
                x = int(landmarks[idx].x * frame.shape[1])
                y = int(landmarks[idx].y * frame.shape[0])
                puntos[idx] = (x, y)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            # B. Detección de vitalidad (parpadeo)
            ear_left = calcular_ear(landmarks, EYE_LEFT)
            ear_right = calcular_ear(landmarks, EYE_RIGHT)
            eye_history.append((ear_left + ear_right) / 2)
            
            # C. Detección de habla (movimiento bucal)
            mar = calcular_mar(landmarks)
            mouth_history.append(mar)
            
            # D. Análisis de emociones
            emocion = analizar_emocion(landmarks)
            
            # E. Determinar estados finales
            vivo = "SI ✅" if len(eye_history) >= 3 and min(eye_history) < EAR_THRESHOLD else "NO ❌"
            hablando = "SI 🗣️" if np.std(mouth_history) > 0.05 else "NO 🤐"
            
            # F. Mostrar resultados en pantalla
            y_offset = 40 + (face_id * 120)  # Espaciado para múltiples rostros
            cv2.putText(frame, f"Rostro #{face_id+1}", (10, y_offset-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.putText(frame, f"Vivo: {vivo}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Habla: {hablando}", (10, y_offset+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
            cv2.putText(frame, f"Emocion: {emocion}", (10, y_offset+60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # 6. Mostrar frame procesado
    cv2.imshow('Sistema de Analisis Facial', frame)
    
    # 7. Salir con Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 8. Liberar recursos
cap.release()
cv2.destroyAllWindows()
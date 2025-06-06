# Documentación de juego Phaser

---

## Informacion

**Alumno:** Rogelio Cristian Punzo Castro  
**Materia:** Inteligencia Artificial  
**No. Control:** 21120245
**Docente:** Jesus Eduardo Alcaraz Chavez  
**Institución:** Instituto Tecnologico de Morelia
**Fecha:** Junio 2025

---

## Algoritmo

**Tipo:** Juego en 2D con IA (Clasificación supervisada)

**Objetivo:** Implementar un entorno interactivo donde el jugador puede controlar manualmente un personaje o dejar que un modelo de aprendizaje automático tome decisiones de movimiento (saltar, moverse a izquierda o derecha) para evitar obstáculos.

**Librerías:**

* `pygame`: motor principal para gráficos y lógica del juego
* `pandas`, `numpy`: procesamiento de datos
* `sklearn`: modelos de clasificación (`MLPClassifier`, `DecisionTreeClassifier`, `KNeighborsClassifier`), escalado de características (`StandardScaler`)

---

## Descripción General

**Tipo de proyecto:** Juego 2D con integración de IA (clasificación supervisada)
**Objetivo:** Crear un entorno interactivo en el que el jugador pueda controlar manualmente un personaje o permitir que un modelo de aprendizaje automático (árbol de decisión, red neuronal o KNN) tome las decisiones de movimiento (saltar, moverse a la izquierda o derecha) para evitar dos balas enemigas (una horizontal y otra vertical).

**Tecnologías y librerías utilizadas:**

* `pygame`: framework para gráficos, manejo de eventos y lógica del juego.
* `pandas` y `numpy`: para almacenar, procesar y manipular datos de juego.
* `scikit-learn`: para los modelos de clasificación supervisada:

  * `DecisionTreeClassifier` (árbol de decisión)
  * `MLPClassifier` (red neuronal multicapa)
  * `KNeighborsClassifier` (KNN)
  * `StandardScaler` (escalado de características para perceptrón y KNN)

**Arquitectura y flujo:**

1. **Menú de selección**: Al iniciar, el juego pide elegir modo:

   * **Manual**: El jugador controla al personaje, se recolectan datos de estado para entrenar posteriormente.
   * **Automático**: Se elige un modelo de IA (árbol, perceptrón o KNN) y, con datos previos cargados, el modelo toma el control del personaje.
2. **Recolección de datos** (modo Manual): Cada 0.05 segundos se registra el estado (`vel_bala`, `dist_h`, `dist_v`, `acción`).
3. **Entrenamiento de modelos** (modo Automático): Se cargan registros del archivo `datos.csv` y se entrenan los modelos según la elección:

   * **Árbol de decisión**: profundida máxima = 5.
   * **Red neuronal**: MLP con una capa oculta de 30 neuronas, hasta 4000 iteraciones.
   * **KNN**: k = 3 o número de muestras disponibles, lo que sea menor.
4. **Inferencia en tiempo real**: Durante el bucle del juego, el modelo predice la acción (0 = nada, 1 = salto, 2 = izquierda, 3 = derecha) basándose en la velocidad de bala y distancias.
5. **Lógica de juego y físicas**: Se implementa gravedad, colisión con suelo, generación aleatoria de balas y detección de colisiones jugador–bala.
6. **Guardado de datos** (al finalizar en modo Manual): Se crea/actualiza `datos.csv` con la distribución de acciones.

---

## Parámetros Clave y Variables

| Parámetro / Variable                       | Descripción                                                                                     |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| `WIDTH`, `HEIGHT`                          | Resolución de la ventana del juego (800 × 400 píxeles).                                         |
| `FPS`                                      | Cuadros por segundo para el bucle del juego (60).                                               |
| `PLAYER_SPEED`                             | Velocidad en píxeles/segundo para movimiento horizontal del jugador (275).                      |
| `gravity`, `player_vel_y`                  | Aceleración vertical constante (900) y velocidad vertical actual del jugador.                   |
| `FRAME_WIDTH`, `FRAME_HEIGHT`              | Tamaño de cada frame en la spritesheet (32 × 48).                                               |
| `SPRITE_ROWS`, `SPRITE_COLS`               | Dimensiones de la spritesheet (1 fila × 4 columnas).                                            |
| `SCALE_FACTOR`                             | Factor de escalado aplicado a cada sprite (1 = sin escala).                                     |
| `RECORD_INTERVAL`                          | Intervalo en segundos para recolectar datos en modo Manual (0.05).                              |
| `last_record_time`                         | Marca de tiempo del último registro de datos.                                                   |
| `DATASET_FILE`                             | Nombre del archivo CSV donde se guardan/reposan los registros de entrenamiento (`"datos.csv"`). |
| `ia_model_perceptron`                      | Instancia de `MLPClassifier` entrenada (o `None` si no se ha entrenado).                        |
| `ia_model_tree`                            | Instancia de `DecisionTreeClassifier` entrenada (o `None`).                                     |
| `ia_model_knn`                             | Instancia de `KNeighborsClassifier` entrenada (o `None`).                                       |
| `scaler_perceptron`                        | Instancia de `StandardScaler` usada para normalizar datos de la red neuronal.                   |
| `scaler_knn`                               | Instancia de `StandardScaler` usada para normalizar datos de KNN.                               |
| `flag_perceptron`, `flag_tree`, `flag_knn` | Banderas booleanas que indican si cada modelo fue entrenado correctamente.                      |
| `bullet_rect`, `bullet2_rect`              | Rectángulos de `pygame.Rect` que representan la posición de las balas (horizontal y vertical).  |
| `bullet_speed`, `bullet2_speed`            | Velocidades en píxeles/segundo de las balas.                                                    |
| `bullet_active`, `bullet2_active`          | Banderas booleanas que indican si cada bala está activa dentro de la pantalla.                  |
| `session_data`                             | Lista interna donde se acumulan temporalmente los registros en una partida en modo Manual.      |
| `frames`                                   | Lista de superficies (sprites) para la animación del personaje.                                 |
| `anim_idx`, `anim_timer`                   | Variables para controlar el índice y el tiempo entre frames en la animación.                    |
| `bg_x`                                     | Coordenada X para el scroll (desplazamiento horizontal) del fondo de juego.                     |
| `score`                                    | Contador de puntos (número de balas esquivadas con éxito).                                      |
| `paused`                                   | Bandera que indica si el juego está en pausa tras colisión jugador–bala.                        |
| `just_jumped`                              | Bandera temporal que indica si el jugador acaba de saltar (para recolectar datos).              |
| `INIT_X`, `return_to_init`, `return_speed` | Variables que centran al jugador en X cuando no se mueve.                                       |
| `snd_jump`, `snd_gameover`                 | Objetos `pygame.mixer.Sound` para efectos de sonido (salto y fin de partida).                   |

---

## Ejemplo de Uso

1. **Instalar dependencias**:

   ```bash
   pip install pygame pandas numpy scikit-learn
   ```

2. **Ejecutar el juego**:

   ```bash
   python phaser_final.py
   ```

3. **Seleccionar modo de juego**:

   * **Manual**: Usa las flechas izquierda/derecha o A/D para mover y Barra espaciadora para saltar. Se recolectarán datos de estado.

     * Cuando termine (colisión), presiona `R` para reiniciar. Se guardarán los datos en `datos.csv`.

   * **Automático**: Requiere que exista `datos.csv` con registros previos.

     1. Elige modelo de IA: 1 (Árbol), 2 (Perceptrón), 3 (KNN).
     2. El modelo se entrena con los datos cargados.
     3. La IA toma el control automáticamente: presiona `R` para reiniciar cuando haya colisión.

4. **Revisar datos guardados**:

   * El archivo `datos.csv` tendrá columnas: `vel_bala`, `dist_h`, `dist_v`, `accion`.
   * Acción: 0 (Ninguno), 1 (Salto), 2 (Izquierda), 3 (Derecha).

---

## Notas y Consideraciones

* **Calidad de datos**: Es importante que el jugador genere suficientes registros en modo Manual (>2) para entrenar los modelos.
* **Normalización**: Los modelos Perceptrón y KNN aplican `StandardScaler` para normalizar las características (`vel_bala`, `dist_h`, `dist_v`). El árbol de decisión no requiere escalado.
* **Profundidad en el árbol**: Se fija en 5 para evitar sobreajuste. Ajustable en código si se desea mayor complejidad.
* **Número de vecinos en KNN**: Se elige 3 o menos, dependiendo del tamaño del dataset. Esto puede ajustarse en `entrenar_knn()`.
* **Robustez**: Todas las funciones de predicción (`predecir_accion`) manejan excepciones para evitar que el juego crashee si el modelo falla.
* **Animación**: Cada frame se actualiza cada 0.125 s (1/8). Se puede ajustar `ANIM_INTERVAL` para más velocidad.
* **Gravedad y física**: Se simula con aceleración de 900 píxeles/s². Ajustable en `gravity`.

---

## Código Fuente con Comentarios

A continuación se muestra el código completo de `phaser_final.py`, ya comentado en detalle para cada sección.

```python
import pygame
import random
import os
import sys
import time
import pandas as pd
import numpy as np
from collections import Counter

# ------------
#  LIBRERÍAS ML
# ------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ============================
#  CONSTANTES DEL JUEGO
# ============================
WIDTH, HEIGHT = 800, 400   # Tamaño de la ventana del juego
FPS = 60                   # Cuadros por segundo para el bucle del juego

# Variables para el sprite del jugador (animación)
FRAME_WIDTH = 32           # Ancho del frame en la spritesheet
FRAME_HEIGHT = 48          # Alto del frame en la spritesheet
SPRITE_ROWS = 1            # Filas en la spritesheet
SPRITE_COLS = 4            # Columnas en la spritesheet
SCALE_FACTOR = 1           # Factor de escala para el sprite

# ============================
#  VARIABLES GLOBALES IA & DATOS
# ============================
ia_model_perceptron = None  # Modelo de red neuronal (Perceptrón)
ia_model_tree = None        # Modelo de árbol de decisión
ia_model_knn = None         # Modelo KNN
scaler_perceptron = None    # Escalador para red neuronal
scaler_knn = None           # Escalador para KNN

# Banderas que indican si cada modelo fue entrenado exitosamente
flag_perceptron = False
flag_tree = False
flag_knn = False

# Lista para almacenar registros de juego manual (dataset)
game_records = []
RECORD_INTERVAL = 0.05      # Intervalo en segundos para muestrear datos durante el juego manual
last_record_time = 0.0      # Marca de tiempo del último registro
# Nombre del archivo CSV donde se guardan los datos
DATASET_FILE = "datos.csv"

# ============================
#  UTILIDADES DE CARGA DE IMÁGENES
# ============================
def load_img(path):
    """
    Carga una imagen desde la ruta dada. Si no existe, genera un placeholder fucsia.
    """
    if not os.path.isfile(path):
        # Si el archivo no existe, creamos una superficie de 32x32 fucsia con dos líneas
        tmp = pygame.Surface((32, 32))
        tmp.fill((255, 0, 255))
        pygame.draw.line(tmp, (0, 0, 0), (0, 0), (32, 32))
        pygame.draw.line(tmp, (0, 0, 0), (0, 32), (32, 0))
        print(f"Imagen no encontrada: {path}, usando placeholder")
        return tmp.convert_alpha()
    return pygame.image.load(path).convert_alpha()

# ============================
#  FUNCIONES PARA INICIALIZAR O REINICIAR MODELOS
# ============================
def limpiar_modelos():
    """
    Reinicia todos los modelos y escaladores a None y restablece las banderas a False.
    """
    global ia_model_perceptron, ia_model_tree, ia_model_knn, scaler_perceptron, scaler_knn
    global flag_perceptron, flag_tree, flag_knn

    ia_model_perceptron = None
    ia_model_tree = None
    ia_model_knn = None
    scaler_perceptron = None
    scaler_knn = None
    flag_perceptron = False
    flag_tree = False
    flag_knn = False

# ============================
#  CARGA DE PARTIDAS PASADAS
# ============================
def cargar_registros_csv(ruta: str = DATASET_FILE) -> bool:
    """
    Intenta cargar los datos existentes desde un archivo CSV. Devuelve True si tuvo éxito.
    """
    global game_records
    if os.path.exists(ruta):
        try:
            df = pd.read_csv(ruta)
            # Convertir DataFrame a lista de listas
            game_records = df.values.tolist()
            print(f"{len(game_records)} registros cargados desde '{ruta}'.")
            return True
        except Exception as e:
            print(f"Fallo leyendo CSV: {e}")
    return False

# ============================
#  ENTRENAMIENTO DE MODELOS (Adaptado de completo.py)
# ============================
def entrenar_perceptron():
    """
    Entrena un modelo de red neuronal tipo MLP usando los registros de juego.
    Retorna True si se entrena correctamente.
    """
    global ia_model_perceptron, scaler_perceptron, flag_perceptron
    # Verificar si hay suficientes datos para entrenar
    if len(game_records) < 2:
        print("No hay datos suficientes para entrenar la red neuronal.")
        return False
    try:
        # Convertir lista de registros a matriz numpy
        datos = np.array(game_records, dtype=float)
        # Asumimos que las primeras 3 columnas son X (features) y la 4a columna es y (acción)
        X = datos[:, :3]
        y = datos[:, 3].astype(int)
        # Escalar X para mejorar convergencia del MLP
        scaler_perceptron = StandardScaler()
        X = scaler_perceptron.fit_transform(X)
        # Definir y entrenar el perceptrón (MLP con una capa oculta)
        ia_model_perceptron = MLPClassifier(hidden_layer_sizes=(30,), max_iter=4000, random_state=42, activation='relu')
        print("Entrenando perceptrón …")
        ia_model_perceptron.fit(X, y)
        flag_perceptron = True
        print("Perceptrón entrenado.")
        return True
    except Exception as e:
        print(f"Error entrenando perceptrón: {e}")
        return False


def entrenar_arbol():
    """
    Entrena un modelo de árbol de decisión usando los registros de juego.
    Retorna True si se entrena correctamente.
    """
    global ia_model_tree, flag_tree
    if len(game_records) < 2:
        print("No hay datos suficientes para generar el árbol de decisión.")
        return False
    try:
        datos = np.array(game_records, dtype=float)
        X = datos[:, :3]
        y = datos[:, 3].astype(int)
        # Crear un modelo de árbol con profundidad máxima de 5 para evitar sobreajuste
        ia_model_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
        print("Entrenando árbol de decisión…")
        ia_model_tree.fit(X, y)
        flag_tree = True
        print("Árbol entrenado.")
        return True
    except Exception as e:
        print(f"Error entrenando árbol: {e}")
        return False


def entrenar_knn():
    """
    Entrena un modelo KNN usando los registros de juego.
    Retorna True si se entrena correctamente.
    """
    global ia_model_knn, scaler_knn, flag_knn
    if len(game_records) < 2:
        print("No hay datos suficientes para generar el KNN.")
        return False
    try:
        datos = np.array(game_records, dtype=float)
        X = datos[:, :3]
        y = datos[:, 3].astype(int)
        # Definir k como mínimo entre 3 y número de muestras disponibles
        n_samples = len(X)
        k = min(3, n_samples) if n_samples > 0 else 0
        if k == 0:
            print("No hay muestras suficientes para KNN.")
            return False
        # Escalar X para KNN
        scaler_knn = StandardScaler()
        X = scaler_knn.fit_transform(X)
        ia_model_knn = KNeighborsClassifier(n_neighbors=k)
        print(f"Entrenando KNN con k={k}…")
        ia_model_knn.fit(X, y)
        flag_knn = True
        print("KNN entrenado.")
        return True
    except Exception as e:
        print(f"Error entrenando KNN: {e}")
        return False

# ============================
#  INFERENCIA Y ACCIÓN IA
# ============================
def predecir_accion(modelo, rect_j, rect_b, activa_bala, vel_bala,
                    rect_b2, activa_bala2, vel_bala2):
    """
    Dada la posición y velocidad de balas y jugador, predice la acción a tomar:
    0 = nada, 1 = salto, 2 = izquierda, 3 = derecha.
    """
    try:
        # Distancia horizontal entre bala y jugador (si bala activa)
        dist_h = abs(rect_b.centerx - rect_j.centerx) if activa_bala else 999
        # Velocidad absoluta de la bala (si bala activa)
        velocidad = abs(vel_bala) if activa_bala else 0
        # Distancia vertical con la segunda bala
        dist_v = abs(rect_b2.centery - rect_j.centery) if activa_bala2 else 999
        # Preparar la fila de entrada
        X_pred = np.array([[velocidad, dist_h, dist_v]], dtype=float)
        # Escalar si corresponde (Perceptrón o KNN)
        if modelo == ia_model_perceptron and scaler_perceptron is not None:
            X_pred = scaler_perceptron.transform(X_pred)
        elif modelo == ia_model_knn and scaler_knn is not None:
            X_pred = scaler_knn.transform(X_pred)
        # Obtener la predicción del modelo y convertirla a entero
        return int(modelo.predict(X_pred)[0])
    except Exception as e:
        print(f"Error predicción: {e}")
        return 0

# ============================
#  RECOLECCIÓN DE DATOS DURANTE JUEGO MANUAL
# ============================
def recolectar_estado(jugador, bala, activa_bala, vel_bala,
                      bala2, activa_bala2, vel_bala2, keys, salto_now):
    """
    Genera una lista con [velocidad, distancia_horizontal, distancia_vertical, acción]
    basada en el estado actual (jugador, balas, teclas presionadas).
    """
    velocidad = abs(vel_bala) if activa_bala else 0
    dist_h = abs(bala.centerx - jugador.centerx) if activa_bala else 999
    dist_v = abs(bala2.centery - jugador.centery) if activa_bala2 else 999
    # Decidir acción según teclas: salto priorizado, luego izquierda, luego derecha, sino nada
    if salto_now:
        accion = 1
    elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
        accion = 2
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        accion = 3
    else:
        accion = 0
    return [velocidad, dist_h, dist_v, accion]

# ============================
#  BUCLE PRINCIPAL DEL JUEGO
# ============================
def main():
    """
    Función principal que inicializa pygame, muestra menús, ejecuta el bucle de juego,
    entrena modelos en modo automático, recopila datos en modo manual, y guarda datos.
    """
    global last_record_time, game_records

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Phaser Final – IA Adaptada")
    clock = pygame.time.Clock()

    # Fuentes para textos en pantalla
    font_title = pygame.font.SysFont(None, 48)
    font_opt = pygame.font.SysFont(None, 36)
    font_sub = pygame.font.SysFont(None, 28)

    mode = None     # "manual" o "auto"
    ia_algo = None  # "tree", "mlp" o "knn"

    # Función interna para dibujar menús simples con lista de opciones
    def draw_menu(title, options):
        screen.fill((30, 30, 40))
        surf_t = font_title.render(title, True, (255, 255, 255))
        screen.blit(surf_t, (WIDTH//2 - surf_t.get_width()//2, 50))
        for i, line in enumerate(options):
            surf = font_opt.render(line, True, (200, 240, 230))
            screen.blit(surf, (WIDTH//2 - surf.get_width()//2, 140 + i*50))
        pygame.display.flip()

    # ------------------------------
    # Selección de modo del juego
    # ------------------------------
    while mode is None:
        draw_menu("Elige el modo de juego:", ["1. Manual", "2. Automático"])
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                return
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_1:
                    mode = 'manual'
                elif ev.key == pygame.K_2:
                    mode = 'auto'

    # ------------------------------
    # Selección de IA en modo automático
    # ------------------------------
    if mode == 'auto':
        # Intentar cargar datos previos; si no hay, se fuerza modo manual
        if not cargar_registros_csv():
            draw_menu("¡No hay datos!", ["Primero juega en Manual."])
            time.sleep(2)
            mode = 'manual'
        else:
            # Ciclo para elegir algoritmo de IA
            while ia_algo is None:
                draw_menu("Elige IA:", ["

```
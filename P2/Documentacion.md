# Documentación de juego Phaser

## Algoritmo

**Tipo:** Juego en 2D con IA (Clasificación supervisada)

**Objetivo:** Implementar un entorno interactivo donde el jugador puede controlar manualmente un personaje o dejar que un modelo de aprendizaje automático tome decisiones de movimiento (saltar, moverse a izquierda o derecha) para evitar obstáculos.

**Librerías:**

* `pygame`: motor principal para gráficos y lógica del juego
* `pandas`, `numpy`: procesamiento de datos
* `sklearn`: modelos de clasificación (`MLPClassifier`, `DecisionTreeClassifier`, `KNeighborsClassifier`), escalado de características (`StandardScaler`)

**Diseño:** Patrón de arquitectura "loop del juego" con estructura modular para lógica de IA, entrenamiento, recolección de datos y predicción.

---

## Parámetros Clave

| Parámetro                                 | Descripción                                              |
| ----------------------------------------- | -------------------------------------------------------- |
| `WIDTH`, `HEIGHT`                         | Resolución de la ventana del juego                       |
| `PLAYER_SPEED`                            | Velocidad de movimiento lateral del jugador              |
| `gravity`, `player_vel_y`                 | Simulación de física vertical                            |
| `modelo_nn`, `modelo_arbol`, `modelo_knn` | Modelos de ML entrenados                                 |
| `datos_modelo`                            | Lista con registros de entrenamiento                     |
| `COLLECTION_INTERVAL`                     | Intervalo de tiempo para recolectar datos en modo manual |
| `SCALE_FACTOR`                            | Escalado de sprites                                      |
| `m_neuronal`, `m_arbol`, `m_knn`          | Banderas activas por tipo de modelo en uso               |

---

## Ejemplo de Uso

1. Ejecutar el script:

```bash
python phaser.py
```

2. Seleccionar modo de juego:

   * Manual: recolecta datos de usuario para entrenar modelos.
   * Auto: IA toma decisiones usando modelos previamente entrenados.

3. En modo auto:

   * Seleccionar modelo: Árbol de Decisión, Red Neuronal o KNN.
   * IA toma control del personaje para esquivar obstáculos.

---

## Notas

* **Entrenamiento supervisado:** El modelo requiere ejemplos previos donde se indique qué acción fue tomada en situaciones específicas.
* **Recolección activa:** Solo se recolectan datos en modo manual cada 0.05 segundos.
* **Normalización:** Es crucial para el modelo de red neuronal (`MLPClassifier`).
* **Predicción robusta:** Se protegen los métodos predictivos con manejo de excepciones para evitar errores de ejecución.
* **Animación y física:** La lógica de animación del jugador y gravedad está integrada con la toma de decisiones de la IA.

---


A continuación, se presenta el código fuente original con comentarios explicativos añadidos.


## Codigo

---

```python

# phaser_comentado.py
# Juego en Pygame con opciones de control manual o mediante modelos de ML

# === LIBRERÍAS ===
import pygame  # Motor de gráficos y sonidos
import random  # Para comportamientos aleatorios (velocidades, etc.)
import os      # Manejo de rutas de archivos
import sys     # Acceso a funciones del sistema
import time    # Manejo de tiempos
import pandas as pd  # Lectura y escritura de datasets
import numpy as np   # Operaciones numéricas

# === LIBRERÍAS DE MACHINE LEARNING ===
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

# === CONFIGURACIÓN DE PANTALLA ===
WIDTH, HEIGHT = 800, 400  # Dimensiones de la ventana
FPS = 60  # Frecuencia de actualización para suavidad

# === CONFIGURACIÓN DE SPRITES ===
FRAME_WIDTH = 32
FRAME_HEIGHT = 48
SCALE_FACTOR = 1
SPRITE_ROWS = 1
SPRITE_COLS = 4

# === FUNCIÓN PARA CARGAR IMÁGENES ===
def load_image(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No se encontró la imagen: {path}")
    return pygame.image.load(path).convert_alpha()

# === VARIABLES GLOBALES PARA MODELOS ML ===
modelo_nn = None
modelo_arbol = None
modelo_knn = None
scaler_nn = None

# === BANDERAS DE MODO ===
m_neuronal = False
m_arbol = False
m_knn = False

datos_modelo = []  # Almacenará datos de entrenamiento (características + acción)
COLLECTION_INTERVAL = 0.05  # Frecuencia de muestreo
last_collection_time = 0.0

# === FUNCIONES PARA REINICIAR Y ENTRENAR MODELOS ===
def limpiar_modelos():
    """Inicializa todos los modelos y desactiva sus banderas"""
    global modelo_nn, modelo_arbol, modelo_knn, m_neuronal, m_arbol, m_knn, scaler_nn
    modelo_nn = modelo_arbol = modelo_knn = scaler_nn = None
    m_neuronal = m_arbol = m_knn = False

def cargar_datos_csv(dataset_path):
    """Carga registros desde un CSV existente al arreglo global"""
    global datos_modelo
    if os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path)
            datos_modelo = df.values.tolist()
            print(f"Cargados {len(datos_modelo)} datos del CSV")
            return True
        except Exception as e:
            print(f"-- No se pudo cargar el CSV: {e}")
            return False
    return False

# === ENTRENAMIENTO DE MODELOS ===
def red_neuronal():
    """Entrena un modelo MLP con los datos cargados"""
    global datos_modelo, modelo_nn, scaler_nn
    if len(datos_modelo) < 20:
        print("No hay datos suficientes para entrenar la red neuronal.")
        return False

    arr = np.array(datos_modelo, dtype=float)
    X = arr[:, :-1]
    y = arr[:, -1].astype(int)

    scaler_nn = StandardScaler()
    X_norm = scaler_nn.fit_transform(X)

    modelo = MLPClassifier(
        hidden_layer_sizes=(100, 50, 25),
        max_iter=8000,
        random_state=42,
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        alpha=0.0001
    )
    print("Entrenando Red Neuronal…")
    modelo.fit(X_norm, y)
    modelo_nn = modelo
    print("Red Neuronal entrenada con éxito.")
    return True

# Entrenamiento de árbol de decisión

def generar_arbol_decision():
    """Entrena un Árbol de Decisión con los datos disponibles"""
    global datos_modelo, modelo_arbol
    if len(datos_modelo) < 20:
        print("No hay datos suficientes para entrenar el Árbol de Decisión.")
        return False

    arr = np.array(datos_modelo, dtype=float)
    X = arr[:, :-1]
    y = arr[:, -1].astype(int)

    modelo = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    modelo.fit(X, y)
    modelo_arbol = modelo
    print("Árbol de Decisión entrenado con éxito.")
    return True

# Entrenamiento de modelo KNN

def generar_knn():
    """Entrena un clasificador KNN"""
    global datos_modelo, modelo_knn
    if len(datos_modelo) < 20:
        print("No hay datos suficientes para entrenar el KNN.")
        return False

    arr = np.array(datos_modelo, dtype=float)
    X = arr[:, :-1]
    y = arr[:, -1].astype(int)

    modelo = KNeighborsClassifier(
        n_neighbors=7,
        weights='distance'
    )
    modelo.fit(X, y)
    modelo_knn = modelo
    print("-- KNN entrenado con éxito.")
    return True

# Las demás funciones de predicción, lógica automática, recolección y main() siguen en el archivo original.
# Esta versión solo incluye la inicialización y entrenamiento de modelos con comentarios pedagógicos.

# Para una versión comentada completa de todo el juego, se recomienda seguir comentando sección por sección como se ejemplifica aquí.


```

---
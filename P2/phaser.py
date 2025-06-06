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
FPS = 60                   # Cuadros por segundo para el bucle de juego

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
# Nombre del archivo CSV donde se guardan los datos\DATASET_FILE = "datos.csv"

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
    entrena modelos en modo automático, recolecta datos en modo manual, y guarda datos.
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
                draw_menu("Elige IA:", ["1. Árbol de decisión", "2. Perceptrón", "3. KNN"])
                # Mostrar cuántos registros hay disponibles
                sub = font_sub.render(f"Datos: {len(game_records)}", True, (150,210,180))
                screen.blit(sub, (WIDTH//2 - sub.get_width()//2, 110))
                pygame.display.flip()
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if ev.type == pygame.KEYDOWN:
                        if ev.key == pygame.K_1:
                            ia_algo = 'tree'
                        elif ev.key == pygame.K_2:
                            ia_algo = 'mlp'
                        elif ev.key == pygame.K_3:
                            ia_algo = 'knn'

    # ---------------------------------------------------
    # ENTRENAMIENTO PREVIO DE MODELO SI ESTAMOS EN MODO AUTO
    # ---------------------------------------------------
    model = None
    if mode == 'auto':
        limpiar_modelos()
        ok = False
        if ia_algo == 'tree':
            ok = entrenar_arbol()
            model = ia_model_tree
        elif ia_algo == 'mlp':
            ok = entrenar_perceptron()
            model = ia_model_perceptron
        elif ia_algo == 'knn':
            ok = entrenar_knn()
            model = ia_model_knn
        # Si falla el entrenamiento, volver a modo manual
        if not ok or model is None:
            print("No se entrenó IA, cambiando a manual.")
            mode = 'manual'

    # ============================
    #  CARGA DE RECURSOS (imágenes y audio)
    # ============================
    bg = load_img(os.path.join('assets', 'game', 'fondito2.png'))
    bg = pygame.transform.scale(bg, (WIDTH, HEIGHT))  # Fondo escalado al tamaño de ventana
    spritesheet = load_img(os.path.join('assets', 'sprites', 'altair2.png'))
    bullet_img = load_img(os.path.join('assets', 'sprites', 'purple_ball.png'))
    ship_img = load_img(os.path.join('assets', 'game', 'ufo.png'))

    pygame.mixer.init()
    try:
        snd_jump = pygame.mixer.Sound(os.path.join('assets', 'audio', 'jump.mp3'))
        snd_gameover = pygame.mixer.Sound(os.path.join('assets', 'audio', 'game_over.wav'))
    except pygame.error:
        snd_jump = None
        snd_gameover = None  # Si falla la carga de audio, se dejan en None

    # ============================
    #  PREPARAR ANIMACIÓN DEL JUGADOR
    # ============================
    frames = []
    for r in range(SPRITE_ROWS):
        for c in range(SPRITE_COLS):
            # Extraer cada frame del spritesheet con un rectángulo
            rect = pygame.Rect(c*FRAME_WIDTH, r*FRAME_HEIGHT, FRAME_WIDTH, FRAME_HEIGHT)
            frame = spritesheet.subsurface(rect)
            frame = pygame.transform.scale(frame, (int(FRAME_WIDTH*SCALE_FACTOR), int(FRAME_HEIGHT*SCALE_FACTOR)))
            frames.append(frame)
    anim_idx = 0
    anim_timer = 0.0
    ANIM_INTERVAL = 1/8  # Intervalo de cambio de frame (segundos)

    # ============================
    #  INICIALIZAR VARIABLES DE ESTADO DEL JUEGO
    # ============================
    running = True
    paused = False
    player_rect = frames[0].get_rect(midbottom=(50, HEIGHT))  # Posición inicial del jugador
    player_vel_y = 0.0
    gravity = 900.0
    PLAYER_SPEED = 275  # Velocidad horizontal del jugador

    # Bala horizontal (disparo desde la derecha hacia la izquierda)
    bullet_rect = bullet_img.get_rect(midbottom=(WIDTH-100, HEIGHT))
    bullet_speed = -200.0
    bullet_active = False

    # Bala vertical (disparo desde arriba hacia abajo)
    bullet2_rect = bullet_img.get_rect(topleft=(50, 0))
    bullet2_speed = 150
    bullet2_active = True

    bg_x = 0.0   # Coordenada x para scroll del fondo
    score = 0    # Puntuación del jugador (cantidad de balas esquivadas)
    session_data = []  # Lista temporal para recolectar datos en modo manual
    just_jumped = False  # Bandera para detectar que acaba de saltar el jugador
    INIT_X = 50          # Posición inicial en x para centrar al jugador gradualmente
    return_to_init = False  # Bandera para devolver al jugador al centro si no se mueve
    return_speed = 250      # Velocidad con la que regresa al centro si no se mueve

    font_small = pygame.font.SysFont(None, 28)
    pause_text = font_title.render("¡Game Over! Pulsa R", True, (255,0,0))

    # ============================
    #  BUCLE PRINCIPAL DEL JUEGO
    # ============================
    while running:
        dt = clock.tick(FPS)/1000.0  # Tiempo delta en segundos
        keys = pygame.key.get_pressed()  # Estado actual de las teclas

        # ------------------------------
        # 1. GESTIÓN DE EVENTOS (cerrar, salto, reiniciar)
        # ------------------------------
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN:
                # Salto manual si juega en modo manual y está en el suelo
                if ev.key == pygame.K_SPACE and player_rect.bottom >= HEIGHT and not paused and mode=='manual':
                    just_jumped = True
                    player_vel_y = -330.0
                    if snd_jump:
                        snd_jump.play()
                # Reiniciar partida cuando está pausado y presiona R
                if ev.key == pygame.K_r and paused:
                    # Reiniciar variables de bala y jugador
                    bullet_active = False
                    bullet2_active = True
                    bullet2_rect.topleft = (50,0)
                    player_rect.midbottom = (INIT_X, HEIGHT)
                    player_vel_y = 0.0
                    paused = False
                    score = 0
                    return_to_init = False

        # ------------------------------
        # 2. LÓGICA DE JUEGO (solo si no está en pausa)
        # ------------------------------
        if not paused:
            # Dibujo de fondo con efecto de scroll horizontal
            bg_x = (bg_x - 100*dt) % WIDTH

            # Mover jugador en modo manual
            if mode=='manual':
                moved = False
                if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                    player_rect.x -= PLAYER_SPEED * dt
                    moved = True
                    return_to_init = True
                elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                    player_rect.x += PLAYER_SPEED * dt
                    moved = True
                    return_to_init = True
                # Si dejó de mover, regresar lentamente al centro (INIT_X)
                if not moved and return_to_init:
                    if player_rect.centerx > INIT_X:
                        player_rect.x -= return_speed * dt
                        if player_rect.centerx <= INIT_X:
                            player_rect.centerx = INIT_X
                            return_to_init = False
                    elif player_rect.centerx < INIT_X:
                        player_rect.x += return_speed * dt
                        if player_rect.centerx >= INIT_X:
                            player_rect.centerx = INIT_X
                            return_to_init = False
                    # Mantener dentro de límites de la pantalla
                    player_rect.clamp_ip(pygame.Rect(0,0,WIDTH,HEIGHT))

            # Recolección de datos en modo manual cada RECORD_INTERVAL segundos
            now = time.time()
            if mode=='manual' and now - last_record_time >= RECORD_INTERVAL:
                # Agregar estado actual al session_data
                session_data.append(recolectar_estado(
                    player_rect, bullet_rect, bullet_active, bullet_speed,
                    bullet2_rect, bullet2_active, bullet2_speed,
                    keys, just_jumped))
                last_record_time = now
                just_jumped = False  # Resetear bandera de salto

            # Lógica de IA en modo automático: predecir acción y ejecutar
            if mode=='auto' and model is not None:
                action = predecir_accion(model, player_rect, bullet_rect, bullet_active, bullet_speed,
                                         bullet2_rect, bullet2_active, bullet2_speed)
                # Ejecutar acción predicha: 1 = salto, 2 = izquierda, 3 = derecha
                if action == 1 and player_rect.bottom >= HEIGHT:
                    player_vel_y = -330.0
                    if snd_jump:
                        snd_jump.play()
                elif action == 2:
                    player_rect.x = max(0, player_rect.x - PLAYER_SPEED * dt)
                elif action == 3:
                    player_rect.x = min(WIDTH - player_rect.width, player_rect.x + PLAYER_SPEED * dt)

            # Física del salto: gravedad y colisión con suelo
            player_vel_y += gravity * dt
            player_rect.y += player_vel_y * dt
            if player_rect.bottom >= HEIGHT:
                player_rect.bottom = HEIGHT
                player_vel_y = 0.0

            # Manejo de bala horizontal: si no está activa, se genera nueva bala al azar
            if not bullet_active:
                bullet_speed = random.randint(-250, -150)
                bullet_rect.midbottom = (WIDTH-50, HEIGHT)
                bullet_active = True
            else:
                bullet_rect.x += bullet_speed * dt
                # Si sale de la pantalla, desactivarla y sumar punto
                if bullet_rect.right < 0:
                    bullet_active = False
                    score += 1

            # Manejo de bala vertical similar
            if bullet2_active:
                bullet2_rect.y += bullet2_speed * dt
                if bullet2_rect.top > HEIGHT:
                    bullet2_speed = 150
                    bullet2_rect.topleft = (50, 0)

            # Verificar colisiones: si alguna bala toca al jugador, finalizar partida
            if (bullet_active and bullet_rect.colliderect(player_rect)) or \
               (bullet2_active and bullet2_rect.colliderect(player_rect)):
                paused = True
                if snd_gameover:
                    snd_gameover.play()

        # ------------------------------
        # 3. DIBUJADO DE ESCENA
        # ------------------------------
        screen.blit(bg, (bg_x - WIDTH, 0))
        screen.blit(bg, (bg_x, 0))
        # Dibujar nave estática en esquina derecha como decoración
        screen.blit(ship_img, ship_img.get_rect(midbottom=(WIDTH-100, HEIGHT-30)))
        if bullet_active:
            screen.blit(bullet_img, bullet_rect)
        if bullet2_active:
            screen.blit(bullet_img, bullet2_rect)

        # Animación del jugador: actualizar índice según ANIM_INTERVAL
        anim_timer += dt
        if anim_timer >= ANIM_INTERVAL:
            anim_timer = 0.0
            anim_idx = (anim_idx + 1) % len(frames)
        screen.blit(frames[anim_idx], player_rect)

        # Mostrar puntuación en pantalla
        screen.blit(font_small.render(f"Puntos: {score}", True, (255,255,0)), (10,10))
        # Mostrar tipo de IA elegida si está en modo automático
        if mode=='auto' and ia_algo:
            screen.blit(font_small.render(f"IA: {ia_algo.upper()}", True, (0,255,0)), (10,40))
        # Mostrar texto de "Game Over" si está en pausa
        if paused:
            screen.blit(pause_text, (WIDTH//2 - pause_text.get_width()//2, HEIGHT//2 - pause_text.get_height()//2))

        pygame.display.flip()

    # ============================
    #  GUARDAR DATOS AL FINALIZAR (solo en modo manual)
    # ============================
    if mode=='manual' and session_data:
        # Convertir session_data a DataFrame y guardar como CSV
        df = pd.DataFrame(session_data, columns=['vel_bala','dist_h','dist_v','accion'])
        df.to_csv(DATASET_FILE, index=False)
        print(f"Datos guardados: {len(df)} registros en {DATASET_FILE}")
        # Mostrar distribución de acciones en consola
        print("Distribución de acciones:")
        for a, cnt in Counter(df['accion']).items():
            print(f"  Acción {a}: {cnt}")

    pygame.quit()

if __name__ == "__main__":
    main()

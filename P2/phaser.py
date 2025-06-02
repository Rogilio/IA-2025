import pygame
import random
import os
import sys
import time
import pandas as pd
import numpy as np

# SKLEARN
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

# ---------------------------------------------------
# CONSTANTES PRINCIPALES
# ---------------------------------------------------
WIDTH, HEIGHT = 800, 400
FPS = 60  # Aumentado para mejor control

# SPRITESHEET DEL JUGADOR
FRAME_WIDTH = 32
FRAME_HEIGHT = 48
SCALE_FACTOR = 1
SPRITE_ROWS = 1
SPRITE_COLS = 4

# ---------------------------------------------------
# FUNCIONES AUXILIARES DE CARGA
# ---------------------------------------------------
def load_image(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No se encontró la imagen: {path}")
    return pygame.image.load(path).convert_alpha()

# ---------------------------------------------------
# VARIABLES GLOBALES PARA MODELOS
# ---------------------------------------------------
modelo_nn = None
modelo_arbol = None
modelo_knn = None
scaler_nn = None

m_neuronal = False
m_arbol = False
m_knn = False

datos_modelo = []    # Lista global para recolectar datos
COLLECTION_INTERVAL = 0.05  # Reducido para mejor recolección
last_collection_time = 0.0

# ---------------------------------------------------
# FUNCIONES DE ENTRENAMIENTO
# ---------------------------------------------------
def limpiar_modelos():
    """Reinicia todos los modelos y banderas"""
    global modelo_nn, modelo_arbol, modelo_knn, m_neuronal, m_arbol, m_knn, scaler_nn
    modelo_nn = None
    modelo_arbol = None
    modelo_knn = None
    scaler_nn = None
    m_neuronal = False
    m_arbol = False
    m_knn = False

def cargar_datos_csv(dataset_path):
    """Carga datos del CSV a la lista global datos_modelo"""
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

def red_neuronal():
    """Entrena una Red Neuronal con los datos"""
    global datos_modelo, modelo_nn, scaler_nn
    if len(datos_modelo) < 20:  # Mínimo más alto
        print("No hay datos suficientes para entrenar la red neuronal.")
        return False

    arr = np.array(datos_modelo, dtype=float)
    X = arr[:, :-1]           # Todas las características excepto la última
    y = arr[:, -1].astype(int) # Última columna es la acción

    # Normalización crucial para redes neuronales
    scaler_nn = StandardScaler()
    X_norm = scaler_nn.fit_transform(X)

    modelo = MLPClassifier(
        hidden_layer_sizes=(100, 50, 25),  # Arquitectura más profunda para más características
        max_iter=8000,  # Más iteraciones para mejor convergencia
        random_state=42, 
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        alpha=0.0001  # Regularización para evitar overfitting
    )
    print("Entrenando Red Neuronal…")
    modelo.fit(X_norm, y)
    modelo_nn = modelo
    print("Red Neuronal entrenada con éxito.")
    return True

def generar_arbol_decision():
    """Entrena un Árbol de Decisión"""
    global datos_modelo, modelo_arbol
    if len(datos_modelo) < 20:
        print("No hay datos suficientes para entrenar el Árbol de Decisión.")
        return False

    arr = np.array(datos_modelo, dtype=float)
    X = arr[:, :-1]           # Todas las características excepto la última
    y = arr[:, -1].astype(int) # Última columna es la acción

    modelo = DecisionTreeClassifier(
        max_depth=10,  # Más profundo para manejar más características
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # Balancear clases para mejor aprendizaje de saltos
    )
    modelo.fit(X, y)
    modelo_arbol = modelo
    print("Árbol de Decisión entrenado con éxito.")
    return True

def generar_knn():
    """Entrena un modelo KNN"""
    global datos_modelo, modelo_knn
    if len(datos_modelo) < 20:
        print("No hay datos suficientes para entrenar el KNN.")
        return False

    arr = np.array(datos_modelo, dtype=float)
    X = arr[:, :-1]           # Todas las características excepto la última
    y = arr[:, -1].astype(int) # Última columna es la acción

    modelo = KNeighborsClassifier(
        n_neighbors=7,  # Más vecinos para mayor estabilidad
        weights='distance'  # Dar más peso a vecinos cercanos
    )
    modelo.fit(X, y)
    modelo_knn = modelo
    print("-- KNN entrenado con éxito.")
    return True

# ---------------------------------------------------
# FUNCIÓN DE LÓGICA AUTOMÁTICA MEJORADA
# ---------------------------------------------------
def logica_auto(accion, player_rect, PLAYER_SPEED, WIDTH, HEIGHT, player_vel_y, gravity, jump_sound, dt):
    """Aplica la acción predicha al jugador"""
    if accion == 1 and player_rect.bottom >= HEIGHT:
        # Saltar
        player_vel_y = -350.0  # Salto más fuerte
        if jump_sound:
            jump_sound.play()
    elif accion == 2 and player_rect.left > 0:
        # Mover a la izquierda
        player_rect.x -= PLAYER_SPEED * dt
    elif accion == 3 and player_rect.right < WIDTH:
        # Mover a la derecha
        player_rect.x += PLAYER_SPEED * dt
    
    return player_vel_y

# ---------------------------------------------------
# RECOLECCIÓN DE DATOS MEJORADA CON MÁS CARACTERÍSTICAS
# ---------------------------------------------------
def collect_game_data(player_rect, bullet_rect, bullet_active, bullet_speed,
                      bullet2_rect, bullet2_active, bullet2_speed,
                      keys, just_jumped, player_vel_y):
    """Recolecta datos del juego con características más detalladas para mejor aprendizaje"""
    
    # Características mejoradas para mejor predicción de saltos
    if bullet_active:
        dist_h = bullet_rect.centerx - player_rect.centerx  # Distancia horizontal (con signo)
        vel_bala = abs(bullet_speed)
        # Tiempo estimado hasta colisión horizontal
        tiempo_colision_h = abs(dist_h / bullet_speed) if bullet_speed != 0 else 999
    else:
        dist_h = 999
        vel_bala = 0
        tiempo_colision_h = 999

    if bullet2_active:
        dist_v = bullet2_rect.centery - player_rect.centery  # Distancia vertical (con signo)
        # Tiempo estimado hasta colisión vertical
        tiempo_colision_v = abs(dist_v / bullet2_speed) if bullet2_speed != 0 else 999
    else:
        dist_v = 999
        tiempo_colision_v = 999

    # Nuevas características importantes para saltos
    player_on_ground = 1 if player_rect.bottom >= HEIGHT else 0
    player_y_velocity = player_vel_y
    
    # Peligro inmediato (balas muy cerca)
    peligro_horizontal = 1 if (bullet_active and abs(dist_h) < 100 and dist_h > 0) else 0
    peligro_vertical = 1 if (bullet2_active and abs(dist_v) < 100 and dist_v > 0) else 0

    # Acción basada en input del jugador
    if just_jumped:
        accion = 1
    elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
        accion = 2
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        accion = 3
    else:
        accion = 0

    # Retornar más características para mejor aprendizaje
    return [vel_bala, abs(dist_h), abs(dist_v), tiempo_colision_h, tiempo_colision_v, 
            player_on_ground, peligro_horizontal, peligro_vertical, accion]

# ---------------------------------------------------
# PREDICCIÓN MEJORADA CON MÁS CARACTERÍSTICAS
# ---------------------------------------------------
def sklearn_predict_improved(model, player_rect, bullet_rect, bullet_active, bullet_speed,
                             bullet2_rect, bullet2_active, bullet2_speed, player_vel_y):
    """Predicción mejorada con características más detalladas"""
    try:
        if bullet_active:
            dist_h = abs(bullet_rect.centerx - player_rect.centerx)
            vel_bala = abs(bullet_speed)
            tiempo_colision_h = dist_h / vel_bala if vel_bala > 0 else 999
        else:
            dist_h = 999
            vel_bala = 0
            tiempo_colision_h = 999

        if bullet2_active:
            dist_v = abs(bullet2_rect.centery - player_rect.centery)
            tiempo_colision_v = dist_v / bullet2_speed if bullet2_speed > 0 else 999
        else:
            dist_v = 999
            tiempo_colision_v = 999

        # Nuevas características para mejor predicción
        player_on_ground = 1 if player_rect.bottom >= HEIGHT else 0
        peligro_horizontal = 1 if (bullet_active and dist_h < 100) else 0
        peligro_vertical = 1 if (bullet2_active and dist_v < 100) else 0

        features = np.array([[vel_bala, dist_h, dist_v, tiempo_colision_h, tiempo_colision_v,
                             player_on_ground, peligro_horizontal, peligro_vertical]])
        
        # Aplicar normalización solo para redes neuronales
        if model == modelo_nn and scaler_nn is not None:
            features = scaler_nn.transform(features)
        
        accion = model.predict(features)[0]
        return int(accion)
    except Exception as e:
        print(f"-- Error en predicción: {e}")
        return 0

# ---------------------------------------------------
# FUNCIÓN PRINCIPAL DEL JUEGO
# ---------------------------------------------------
def main():
    global last_collection_time, datos_modelo

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Juego AI en Pygame con Modelos ML")
    clock = pygame.time.Clock()

    # ---------------------------------------------------
    # MENÚ INICIAL: Selección de modo
    # ---------------------------------------------------
    font_title = pygame.font.SysFont(None, 48)
    font_option = pygame.font.SysFont(None, 36)
    font_subtitle = pygame.font.SysFont(None, 28)

    mode = None  # 'normal' o 'auto'
    ai_algo = None  # 'tree', 'mlp' o 'knn'
    dataset_path = "datos.csv"

    # Bucle para seleccionar Manual vs Auto
    while mode is None:
        screen.fill((20, 20, 20))
        title_surf = font_title.render("Selecciona Modo de Juego:", True, (255, 255, 255))
        screen.blit(title_surf, (WIDTH//2 - title_surf.get_width()//2, 50))

        opts = ["1. Normal (Modo Manual)", "2. Auto (Inteligencia Artificial)"]
        for i, txt in enumerate(opts):
            surf = font_option.render(txt, True, (200, 200, 200))
            screen.blit(surf, (WIDTH//2 - surf.get_width()//2, 150 + i*50))

        pygame.display.flip()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                return
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_1:
                    mode = 'normal'
                elif ev.key == pygame.K_2:
                    mode = 'auto'

    # Si elegimos Auto, pedimos el algoritmo
    if mode == 'auto':
        # Cargar datos existentes del CSV
        if not cargar_datos_csv(dataset_path):
            # Si no hay datos, forzar modo manual
            error = True
            while error:
                screen.fill((20,20,20))
                msg1 = font_title.render("¡Error!", True, (255,0,0))
                screen.blit(msg1, (WIDTH//2 - msg1.get_width()//2, 150))
                msg2 = font_option.render("No se encontraron datos.", True, (255,255,255))
                screen.blit(msg2, (WIDTH//2 - msg2.get_width()//2, 200))
                msg3 = font_subtitle.render("Juega primero en modo Normal.", True, (150,150,150))
                screen.blit(msg3, (WIDTH//2 - msg3.get_width()//2, 250))
                pygame.display.flip()
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if ev.type == pygame.KEYDOWN:
                        error = False
                        mode = 'normal'
        else:
            # Seleccionar algoritmo
            while ai_algo is None:
                screen.fill((20, 20, 20))
                title_surf = font_title.render("Selecciona Algoritmo IA:", True, (255, 255, 255))
                screen.blit(title_surf, (WIDTH//2 - title_surf.get_width()//2, 50))
                opts = ["1. Árbol de Decisión", "2. Red Neuronal", "3. KNN"]
                for i, txt in enumerate(opts):
                    surf = font_option.render(txt, True, (200, 200, 200))
                    screen.blit(surf, (WIDTH//2 - surf.get_width()//2, 150 + i*50))
                subtitle = font_subtitle.render(f"Datos disponibles: {len(datos_modelo)}", True, (150,150,150))
                screen.blit(subtitle, (WIDTH//2 - subtitle.get_width()//2, 110))
                pygame.display.flip()
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if ev.type == pygame.KEYDOWN:
                        if ev.key == pygame.K_1:
                            ai_algo = 'tree'
                        elif ev.key == pygame.K_2:
                            ai_algo = 'mlp'
                        elif ev.key == pygame.K_3:
                            ai_algo = 'knn'

    # ---------------------------------------------------
    # ENTRENAMIENTO PREVIO
    # ---------------------------------------------------
    model = None
    if mode == 'auto':
        limpiar_modelos()
        if ai_algo == 'tree':
            success = generar_arbol_decision()
            if success:
                model = modelo_arbol
                m_arbol = True
        elif ai_algo == 'mlp':
            success = red_neuronal()
            if success:
                model = modelo_nn
                m_neuronal = True
        elif ai_algo == 'knn':
            success = generar_knn()
            if success:
                model = modelo_knn
                m_knn = True

        if model is None:
            print("-- No se pudo entrenar el modelo, cambiando a modo manual")
            mode = 'normal'

    # ---------------------------------------------------
    # CARGA DE RECURSOS
    # ---------------------------------------------------
    try:
        bg = load_image(os.path.join("assets", "game", "fondito2.png"))
        bg = pygame.transform.scale(bg, (WIDTH, HEIGHT))
        spritesheet = load_image(os.path.join("assets", "sprites", "altair2.png"))
        bullet_img = load_image(os.path.join("assets", "sprites", "purple_ball.png"))
        ship_img = load_image(os.path.join("assets", "game", "ufo.png"))
    except FileNotFoundError as e:
        print(e)
        pygame.quit()
        sys.exit()

    pygame.mixer.init()
    try:
        jump_sound = pygame.mixer.Sound(os.path.join("assets", "audio", "jump.mp3"))
        game_over_sound = pygame.mixer.Sound(os.path.join("assets", "audio", "game_over.wav"))
    except pygame.error:
        jump_sound = None
        game_over_sound = None

    font_large = pygame.font.SysFont(None, 48)
    font_small = pygame.font.SysFont(None, 28)

    # ---------------------------------------------------
    # PREPARAR ANIMACIÓN DEL JUGADOR
    # ---------------------------------------------------
    frames_run = []
    for row in range(SPRITE_ROWS):
        for col in range(SPRITE_COLS):
            x = col * FRAME_WIDTH
            y = row * FRAME_HEIGHT
            rect = pygame.Rect(x, y, FRAME_WIDTH, FRAME_HEIGHT)
            frame = spritesheet.subsurface(rect)
            frame = pygame.transform.scale(frame, (int(FRAME_WIDTH * SCALE_FACTOR), int(FRAME_HEIGHT * SCALE_FACTOR)))
            frames_run.append(frame)
    anim_index = 0
    anim_timer = 0.0
    ANIM_INTERVAL = 1/7

    # ---------------------------------------------------
    # ESTADOS INICIALES DEL JUEGO
    # ---------------------------------------------------
    running = True
    paused = False

    player_rect = frames_run[0].get_rect(midbottom=(50, HEIGHT))
    player_vel_y = 0.0
    gravity = 1000.0
    PLAYER_SPEED = 300

    # Bala horizontal
    bullet_rect = bullet_img.get_rect(midbottom=(WIDTH - 100, HEIGHT))
    bullet_speed = -200.0  # Velocidad constante inicial
    bullet_active = False

    # Bala vertical (siempre desde la esquina superior izquierda)
    bullet2_rect = bullet_img.get_rect(topleft=(0, 0))
    bullet2_speed = 200
    bullet2_active = True

    bg_x = 0.0
    score = 0

    datosEntrenamiento = []  # Para guardar en CSV al finalizar

    pause_text = font_large.render("¡Game Over! Presiona R para reiniciar", True, (255, 0, 0))
    just_jumped = False

    # ---------------------------------------------------
    # BUCLE PRINCIPAL
    # ---------------------------------------------------
    while running:
        dt = clock.tick(FPS) / 1000.0
        keys = pygame.key.get_pressed()

        # ------------------------------------
        # 1. GESTIÓN DE EVENTOS
        # ------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                # Salto manual (solo en modo Normal)
                if event.key == pygame.K_SPACE and player_rect.bottom >= HEIGHT and not paused:
                    if mode == 'normal':
                        just_jumped = True
                        player_vel_y = -350.0
                        if jump_sound:
                            jump_sound.play()
                # Reiniciar con R si estamos en pausa
                if event.key == pygame.K_r and paused:
                    # Reiniciar estado
                    bullet_active = False
                    bullet2_active = True
                    bullet2_rect.topleft = (0, 0)  # Reiniciar desde esquina superior izquierda
                    player_rect.midbottom = (50, HEIGHT)
                    player_vel_y = 0.0
                    paused = False
                    score = 0

        # ------------------------------------
        # 2. LÓGICA DE JUEGO (solo si no está pausado)
        # ------------------------------------
        if not paused:
            # Fondo en movimiento
            bg_x = (bg_x - 100.0 * dt) % WIDTH

            # Movimiento horizontal manual
            if mode == 'normal':
                if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                    player_rect.x -= PLAYER_SPEED * dt
                if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                    player_rect.x += PLAYER_SPEED * dt
                # Límites de pantalla
                if player_rect.left < 0:
                    player_rect.left = 0
                if player_rect.right > WIDTH:
                    player_rect.right = WIDTH

            # Recolección de datos en modo manual
            current_time = time.time()
            if mode == 'normal' and (current_time - last_collection_time >= COLLECTION_INTERVAL):
                data_point = collect_game_data(
                    player_rect, bullet_rect, bullet_active, bullet_speed,
                    bullet2_rect, bullet2_active, bullet2_speed,
                    keys, just_jumped, player_vel_y
                )
                datosEntrenamiento.append(data_point)
                last_collection_time = current_time
                just_jumped = False

            # Lógica automática
            if mode == 'auto' and model is not None:
                accion = sklearn_predict_improved(
                    model, player_rect, bullet_rect, bullet_active, bullet_speed,
                    bullet2_rect, bullet2_active, bullet2_speed, player_vel_y
                )
                player_vel_y = logica_auto(
                    accion, player_rect, PLAYER_SPEED, WIDTH, HEIGHT, 
                    player_vel_y, gravity, jump_sound, dt
                )

            # Física vertical (gravedad + posición)
            player_vel_y += gravity * dt
            player_rect.y += player_vel_y * dt
            if player_rect.bottom >= HEIGHT:
                player_rect.bottom = HEIGHT
                player_vel_y = 0.0

            # Lógica de bala horizontal
            if not bullet_active:
                bullet_speed = random.randint(-250, -150)  # Velocidad variable
                bullet_rect.midbottom = (WIDTH - 50, HEIGHT)
                bullet_active = True
            else:
                bullet_rect.x += bullet_speed * dt
                if bullet_rect.right < 0:
                    bullet_active = False
                    score += 1

            # Lógica de bala vertical (siempre desde esquina superior izquierda)
            if bullet2_active:
                bullet2_rect.y += bullet2_speed * dt
                if bullet2_rect.top > HEIGHT:
                    bullet2_speed = random.randint(150, 300)
                    bullet2_rect.topleft = (0, 0)  # Siempre desde esquina superior izquierda

            # Colisiones: si colisiona, pausamos
            if (bullet_active and bullet_rect.colliderect(player_rect)) or \
               (bullet2_active and bullet2_rect.colliderect(player_rect)):
                paused = True
                if game_over_sound:
                    game_over_sound.play()

        # ------------------------------------
        # 3. DIBUJADO EN PANTALLA
        # ------------------------------------
        screen.blit(bg, (bg_x - WIDTH, 0))
        screen.blit(bg, (bg_x, 0))

        ship_rect = ship_img.get_rect(midbottom=(WIDTH - 100, HEIGHT - 30))
        screen.blit(ship_img, ship_rect)

        # Dibujar balas
        if bullet_active:
            screen.blit(bullet_img, bullet_rect)
        if bullet2_active:
            screen.blit(bullet_img, bullet2_rect)

        # Animación del jugador
        anim_timer += dt
        if anim_timer >= ANIM_INTERVAL:
            anim_timer = 0.0
            anim_index = (anim_index + 1) % len(frames_run)
        current_frame = frames_run[anim_index]
        screen.blit(current_frame, player_rect)

        # Información en pantalla
        score_text = font_small.render(f"Puntos: {score}", True, (255, 255, 0))
        screen.blit(score_text, (10, 10))
        
        mode_text = font_small.render(f"Modo: {mode.upper()}", True, (255, 255, 255))
        screen.blit(mode_text, (10, 40))
        
        if mode == 'auto' and ai_algo:
            algo_text = font_small.render(f"IA: {ai_algo.upper()}", True, (0, 255, 0))
            screen.blit(algo_text, (10, 70))

        # Si está pausado, mostramos texto de reinicio
        if paused:
            screen.blit(pause_text, (WIDTH//2 - pause_text.get_width()//2, HEIGHT//2 - pause_text.get_height()//2))

        pygame.display.flip()

    # ---------------------------------------------------
    # GUARDAR DATOS AL FINALIZAR
    # ---------------------------------------------------
    if mode == 'normal' and datosEntrenamiento:
        columns = ['vel', 'dist_h', 'dist_v', 'tiempo_col_h', 'tiempo_col_v', 
                  'en_suelo', 'peligro_h', 'peligro_v', 'accion']
        df_new = pd.DataFrame(datosEntrenamiento, columns=columns)
        
        # CAMBIO PRINCIPAL: Sobrescribir en lugar de concatenar
        df_new.to_csv(dataset_path, index=False)
        print(f"Dataset creado/sobrescrito con {len(df_new)} registros")
        
        # Mostrar estadísticas de acciones para verificar balance
        print(f"Distribución de acciones:")
        print(f"  - No hacer nada (0): {len(df_new[df_new['accion'] == 0])}")
        print(f"  - Saltar (1): {len(df_new[df_new['accion'] == 1])}")
        print(f"  - Izquierda (2): {len(df_new[df_new['accion'] == 2])}")
        print(f"  - Derecha (3): {len(df_new[df_new['accion'] == 3])}")

    pygame.quit()

if __name__ == "__main__":
    main()
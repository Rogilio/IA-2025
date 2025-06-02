import pygame
import random
import os
import sys
import time
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import NotFittedError
from sklearn.multioutput import MultiOutputClassifier

# ---------------------------------------------------
# CONSTANTES PRINCIPALES
# ---------------------------------------------------
WIDTH, HEIGHT = 800, 400
FPS = 20

# ---------------------------------------------------
# CONFIGURACIÓN DEL SPRITESHEET DEL JUGADOR
# ---------------------------------------------------
FRAME_WIDTH = 32    # Ancho de cada frame en el spritesheet
FRAME_HEIGHT = 48   # Alto de cada frame
SCALE_FACTOR = 1  # Sin escalado adicional
SPRITE_ROWS = 1
SPRITE_COLS = 4

def collect_game_data(player_rect, bullet_rect, bullet_active, bullet_speed,
                     bullet2_rect, bullet2_active, bullet2_speed,
                     keys, just_jumped, HEIGHT):
    """Recolecta datos completos del estado del juego"""
    
    # Información bala horizontal
    if bullet_active:
        desp_horizontal = player_rect.centerx - bullet_rect.centerx
        vel_horizontal = abs(bullet_speed)
    else:
        desp_horizontal = 9999  # Muy lejos
        vel_horizontal = 0
    
    # Información bala vertical
    if bullet2_active:
        desp_vertical = player_rect.centery - bullet2_rect.centery
        vel_vertical = bullet2_speed
    else:
        desp_vertical = 9999  # Muy lejos
        vel_vertical = 0
    
    # Estado del jugador
    player_y = player_rect.centery
    on_ground = 1 if player_rect.bottom >= HEIGHT else 0
    
    # Acciones del jugador
    jump_flag = 1 if just_jumped else 0
    move_left_flag = 1 if (keys[pygame.K_LEFT] or keys[pygame.K_a]) else 0
    move_right_flag = 1 if (keys[pygame.K_RIGHT] or keys[pygame.K_d]) else 0
    
    return {
        'desp_horizontal': desp_horizontal,
        'vel_horizontal': vel_horizontal,
        'desp_vertical': desp_vertical,
        'vel_vertical': vel_vertical,
        'player_y': player_y,
        'on_ground': on_ground,
        'jump': jump_flag,
        'move_left': move_left_flag,
        'move_right': move_right_flag
    }

# ---------------------------------------------------
# FUNCIÓN PARA CARGAR IMÁGENES
# ---------------------------------------------------
def load_image(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No se encontró la imagen: {path}")
    return pygame.image.load(path).convert_alpha()

# ---------------------------------------------------
# FUNCIONES DE PREDICCIÓN CON SKLEARN
# ---------------------------------------------------
def train_linear_regression(dataset_path):
    df = pd.read_csv(dataset_path)
    # Características: incluye ambas balas
    X = df[['desp_horizontal', 'vel_horizontal', 'desp_vertical', 'vel_vertical', 
            'player_y', 'on_ground']].values
    y = df[['jump', 'move_left', 'move_right']].values
    
    # MultiOutputClassifier para manejar múltiples salidas binarias
    model = MultiOutputClassifier(LogisticRegression(max_iter=1000))
    model.fit(X, y)
    return model

def train_decision_tree(dataset_path):
    df = pd.read_csv(dataset_path)
    X = df[['desp_horizontal', 'vel_horizontal', 'desp_vertical', 'vel_vertical',
            'player_y', 'on_ground']].values
    y = df[['jump', 'move_left', 'move_right']].values
    
    model = MultiOutputClassifier(DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=5,
        random_state=42
    ))
    model.fit(X, y)
    return model

def train_mlp(dataset_path):
    df = pd.read_csv(dataset_path)
    X = df[['desp_horizontal', 'vel_horizontal', 'desp_vertical', 'vel_vertical',
            'player_y', 'on_ground']].values
    y = df[['jump', 'move_left', 'move_right']].values
    
    model = MultiOutputClassifier(MLPClassifier(
        hidden_layer_sizes=(20, 15, 10),
        max_iter=1000,
        random_state=42,
        early_stopping=True
    ))
    model.fit(X, y)
    return model

def predict_actions(model, desp_h, vel_h, desp_v, vel_v, player_y, on_ground):
    """Predicción unificada para todos los modelos"""
    try:
        features = [[desp_h, vel_h, desp_v, vel_v, player_y, on_ground]]
        predictions = model.predict(features)[0]
        return int(predictions[0]), int(predictions[1]), int(predictions[2])
    except (NotFittedError, Exception):
        return 0, 0, 0

# ---------------------------------------------------
# CLASE PRINCIPAL DEL JUEGO
# ---------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Juego AI en Pygame con Módos ML")
    clock = pygame.time.Clock()

    # ---------------------------------------------------
    # PANTALLA DE SELECCIÓN DE MODO
    # ---------------------------------------------------
    font_title = pygame.font.SysFont(None, 48)
    font_option = pygame.font.SysFont(None, 36)
    mode = None
    dataset_path = "datos.csv"
    options = ["1. Normal", 
               "2. Regresión Lineal", 
               "3. Árbol de Decisión",
               "4. Redes Neuronales Multicapa", 
               "5. Auto"]
    while mode is None:
        screen.fill((20, 20, 20))
        title_surf = font_title.render("Selecciona Modo de Juego:", True, (255, 255, 255))
        screen.blit(title_surf, (WIDTH//2 - title_surf.get_width()//2, 50))
        for i, opt in enumerate(options):
            opt_surf = font_option.render(opt, True, (200, 200, 200))
            screen.blit(opt_surf, (WIDTH//2 - opt_surf.get_width()//2, 150 + i*40))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    mode = 'normal'
                    if mode == 'normal' and os.path.exists(dataset_path):
                        os.remove(dataset_path)
                elif event.key == pygame.K_2:
                    mode = 'linear'
                elif event.key == pygame.K_3:
                    mode = 'tree'
                elif event.key == pygame.K_4:
                    mode = 'mlp'
                elif event.key == pygame.K_5:
                    mode = 'auto'

    # ---------------------------------------------------
    # INTENTAR CARGAR MODELOS SI NO ES MODO NORMAL
    # ---------------------------------------------------
    model = None
    
    if mode in ('linear', 'tree', 'mlp'):
        # Si no existe dataset, solo permitimos modo 'normal' o 'auto'
        if os.path.exists(dataset_path):
            if mode == 'linear':
                model = train_linear_regression(dataset_path)
            elif mode == 'tree':
                model = train_decision_tree(dataset_path)
            elif mode == 'mlp':
                model = train_mlp(dataset_path)
        else:
            mode = 'normal'  # Sin datos, regresar a normal

    # ---------------------------------------------------
    # CARGA DE RECURSOS (IMÁGENES + SONIDOS + FUENTES)
    # ---------------------------------------------------
    try:
        bg = load_image(os.path.join("assets", "game", "fondito2.png"))
    except FileNotFoundError as e:
        print(e)
        pygame.quit()
        sys.exit()
    bg = pygame.transform.scale(bg, (WIDTH, HEIGHT))

    try:
        menu_img = load_image(os.path.join("assets", "game", "menu.png"))
    except FileNotFoundError as e:
        print(e)
        pygame.quit()
        sys.exit()
    menu_img = pygame.transform.scale(menu_img, (WIDTH, HEIGHT))

    try:
        spritesheet = load_image(os.path.join("assets", "sprites", "altair2.png"))
    except FileNotFoundError as e:
        print(e)
        pygame.quit()
        sys.exit()

    try:
        bullet_img = load_image(os.path.join("assets", "sprites", "purple_ball.png"))
    except FileNotFoundError as e:
        print(e)
        pygame.quit()
        sys.exit()

    try:
        ship_img = load_image(os.path.join("assets", "game", "ufo.png"))
    except FileNotFoundError as e:
        print(e)
        pygame.quit()
        sys.exit()

    pygame.mixer.init()
    try:
        jump_sound = pygame.mixer.Sound(os.path.join("assets", "audio", "jump.mp3"))
        game_over_sound = pygame.mixer.Sound(os.path.join("assets", "audio", "game_over.wav"))
    except pygame.error as e:
        print("Error al cargar sonido:", e)
        jump_sound = None
        game_over_sound = None

    font_large = pygame.font.SysFont(None, 48)
    font_small = pygame.font.SysFont(None, 28)

    # ---------------------------------------------------
    # ANIMACIÓN DEL JUGADOR: RECORTE DE FRAMES DEL SPRITESHEET
    # ---------------------------------------------------
    frames_run = []
    for row in range(SPRITE_ROWS):
        for col in range(SPRITE_COLS):
            x = col * FRAME_WIDTH
            y = row * FRAME_HEIGHT
            rect = pygame.Rect(x, y, FRAME_WIDTH, FRAME_HEIGHT)
            frame = spritesheet.subsurface(rect)
            frame = pygame.transform.scale(
                frame,
                (int(FRAME_WIDTH * SCALE_FACTOR), int(FRAME_HEIGHT * SCALE_FACTOR))
            )
            frames_run.append(frame)

    anim_index = 0
    anim_timer = 0.0
    ANIM_INTERVAL = 1/7

    # ---------------------------------------------------
    # ESTADOS INICIALES DEL JUEGO
    # ---------------------------------------------------
    show_menu = False  # Ya seleccionamos modo
    running = True
    paused = False
    mode_auto = (mode == 'auto')

    player_rect = frames_run[0].get_rect(midbottom=(50, HEIGHT))
    player_vel_y = 0.0
    gravity = 800.0

    bullet_rect = bullet_img.get_rect(midbottom=(WIDTH - 100, HEIGHT))
    bullet_speed = 0.0
    bullet_active = False

    # ---------------------------------------------------
    # Variables para la bala vertical (parte superior izquierda)
    # ---------------------------------------------------
    bullet2_rect    = bullet_img.get_rect(topleft=(50, 0))  # margen de 50px al eje X
    bullet2_speed   = 200#random.randint(200, 400)              # velocidad en px/s hacia abajo
    bullet2_active  = True


    bg_x = 0.0
    score = 0

    datosEntrenamiento = [] # <<<--- lista local para recolectar solo en modo 'normal'

    pause_text = font_large.render("¡Game Over! Presiona R para reiniciar", True, (255, 0, 0))
    PLAYER_SPEED = 200
    just_jumped = False  # <— bandera que marca si el usuario acaba de pulsar Espacio

    # ---------------------------------------------------
    # BUCLE PRINCIPAL
    # ---------------------------------------------------
    while running:
        dt = clock.tick(FPS) / 1000.0
        keys = pygame.key.get_pressed()

        # Movimiento horizontal en todos modos salvo pausado
        if not paused:
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                player_rect.x -= PLAYER_SPEED * dt
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                player_rect.x += PLAYER_SPEED * dt
            if player_rect.left < 0:
                player_rect.left = 0
            if player_rect.right > WIDTH:
                player_rect.right = WIDTH

        # ------------------------------------
        # 1. GESTIÓN DE EVENTOS
        # ------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                # Salto manual (solo en modo 'normal' o 'auto' si se presiona Espacio)
                if event.key == pygame.K_SPACE and player_rect.bottom >= HEIGHT and not paused:
                    # En modo LIN/TREE/MLP/Auto, este Espacio no debe generar datos nuevos,
                    # así que solo permitimos Espacio → salto presencial cuando mode == 'normal'
                    if mode == 'normal':
                        just_jumped = True
                        player_vel_y = -270.0
                        if jump_sound:
                            jump_sound.play()
                # Reiniciar con R si estamos en pausa            
                if event.key == pygame.K_r and paused:
                    bullet_active = False
                    player_rect.midbottom = (50, HEIGHT)
                    player_vel_y = 0.0
                    paused = False
                    score = 0
                    datosEntrenamiento.clear()  # <-- Limpiamos solo la lista local
                    # Si hemos regresado a un modo ML, reentrenamos con el CSV
                    if mode in ('linear', 'tree', 'mlp') and os.path.exists(dataset_path):
                        # reentrenar modelo
                        if mode == 'linear':
                            model = train_linear_regression(dataset_path)
                        elif mode == 'tree':
                            model = train_decision_tree(dataset_path)
                        elif mode == 'mlp':
                            model = train_mlp(dataset_path)
                    mode_auto = (mode == 'auto')

        # ------------------------------------
        # 2. LÓGICA DE JUEGO (solo si no está en pausa)
        # ------------------------------------
        if not paused:
            bg_x = (bg_x - 100.0 * dt) % WIDTH

            # DEFINIR PREDICCIONES POR DEFECTO
            jump_pred, left_pred, right_pred = 0, 0, 0

            # IA o ML salto
            if bullet_active and (mode_auto or mode in ('linear', 'tree', 'mlp')):
                despBala = player_rect.x - bullet_rect.x
                velocidadBala = abs(bullet_speed)
                #estatusAire = 1 if player_rect.bottom < HEIGHT else 0
                #estatuSuelo = 1 if player_rect.bottom >= HEIGHT else 0
                # 1) Predecimos los tres valores (jump_pred, left_pred, right_pred)
                if mode == 'linear':
                    jump_pred, left_pred, right_pred = sklearn_predict_linear(model, despBala, velocidadBala)
                elif mode == 'tree':
                    jump_pred, left_pred, right_pred = sklearn_predict_tree(model, despBala, velocidadBala)
                elif mode == 'mlp':
                    jump_pred, left_pred, right_pred = sklearn_predict_mlp(model, despBala, velocidadBala)
                elif mode_auto and bullet_active:
                    jump_pred = random.choice([0,1])
                    left_pred = 0
                    right_pred = 0
                    


            # 2) Aplicar movimiento horizontal según predicción (siempre y cuando no estemos en pausa ni en el aire)
            if left_pred == 1 and player_rect.left > 0:
                player_rect.x -= PLAYER_SPEED * dt
            if right_pred == 1 and player_rect.right < WIDTH:
                player_rect.x += PLAYER_SPEED * dt

            # 3) Si predice salto y estamos en el suelo:
            if jump_pred == 1 and player_rect.bottom >= HEIGHT:
                player_vel_y = -270.0
                if jump_sound:
                    jump_sound.play()
            # 2.2. Física vertical (gravedad + posición)
            player_vel_y += gravity * dt
            player_rect.y += player_vel_y * dt
            if player_rect.bottom >= HEIGHT:
                player_rect.bottom = HEIGHT
                player_vel_y = 0.0
            # 2.3. Lógica de bala
            if not bullet_active:
                bullet_speed = -200#-random.randint(300, 800)
                bullet_rect.midbottom = (WIDTH - 100, HEIGHT)
                bullet_active = True
            else:
                bullet_rect.x += bullet_speed * dt
                if bullet_rect.right < 0:
                    bullet_active = False
                    score += 1

            # 2.4. RECOLECCIÓN DE DATOS (solo modo 'normal')
            #    Queremos grabar en cada frame: desp, vel y si el jugador SALTÓ (1) o NO (0).
            if mode == 'normal':
                # 1) Si la bala está activa, calculamos sus valores
                if bullet_active:
                    despBala = player_rect.x - bullet_rect.x
                    velocidadBala = abs(bullet_speed)
                else:
                    # Si no hay bala, ponemos valores neutros o por fuera de rango
                    # (esto ayuda a que el modelo aprenda que "no hay bala" => no saltar).
                    despBala = 9999     # o algún número grande
                    velocidadBala = 0
                # 2) Jump = 1 si justo saltaste en este frame, 0 si no
                jump_flag = 1 if just_jumped else 0

                # 3) Detectar movimiento horizontal este frame
                move_left_flag  = 1 if (keys[pygame.K_LEFT] or keys[pygame.K_a]) else 0
                move_right_flag = 1 if (keys[pygame.K_RIGHT] or keys[pygame.K_d]) else 0

                # 4) Añadir la fila al dataset
                datosEntrenamiento.append({
                    'desp':        despBala,
                    'vel':         velocidadBala,
                    'jump':        jump_flag,
                    'move_left':   move_left_flag,   
                    'move_right':  move_right_flag   
                })
                # Finalmente reiniciamos la bandera, para que solo se grabe una vez por pulsación:
                just_jumped = False
            # Colisión bala horizontal vs jugador
            if bullet_rect.colliderect(player_rect):
                paused = True
                if game_over_sound:
                    game_over_sound.play()

            # — BULLET VERTICAL (parte superior izquierda) —
            if bullet2_active:
                # La bala baja
                bullet2_rect.y += bullet2_speed * dt

                # Si sale por abajo, la volvemos a colocar arriba y reasignamos velocidad
                if bullet2_rect.top > HEIGHT:
                    bullet2_speed = random.randint(200, 400)   # nueva velocidad aleatoria
                    bullet2_rect.bottom = 0                   # aparece justo arriba de la pantalla

                # Colisión bala vertical ↔ jugador
                if bullet2_rect.colliderect(player_rect):
                    paused = True
                    if game_over_sound:
                        game_over_sound.play()

            #Animacion
            anim_timer += dt
            if anim_timer >= ANIM_INTERVAL:
                anim_timer = 0.0
                anim_index = (anim_index + 1) % len(frames_run)

        # ────────────────────────────────────────
        #  DIBUJADO EN PANTALLA
        # ────────────────────────────────────────
        screen.blit(bg, (bg_x - WIDTH, 0))
        screen.blit(bg, (bg_x, 0))

        ship_rect = ship_img.get_rect(midbottom=(WIDTH - 100, HEIGHT - 30))
        screen.blit(ship_img, ship_rect)
        # Dibuja bala horizontal
        screen.blit(bullet_img, bullet2_rect)
        # Dibuja bala vertical
        screen.blit(bullet_img, bullet_rect)

        current_frame = frames_run[anim_index]
        screen.blit(current_frame, player_rect)
        score_text = font_small.render(f"Puntos: {score}", True, (255, 255, 0))
        screen.blit(score_text, (10, 10))
        if paused:
            screen.blit(pause_text, (WIDTH//2 - pause_text.get_width()//2, HEIGHT//2 - pause_text.get_height()//2))
        pygame.display.flip()

    # Al cerrar, guardamos el dataset actual en "datos.csv"
    if mode == 'normal' and datosEntrenamiento:
        df_new = pd.DataFrame(datosEntrenamiento)
        if os.path.exists(dataset_path):
            df_old = pd.read_csv(dataset_path)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
            df_all.to_csv(dataset_path, index=False)
        else:
            df_new.to_csv(dataset_path, index=False)

    pygame.quit()

if __name__ == "__main__":
    main()

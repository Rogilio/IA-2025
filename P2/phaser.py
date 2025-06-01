import pygame
import random
import os
import sys
import time
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import NotFittedError

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
    X = df[['desp', 'vel']].values
    y = df['jump'].values
    model = LinearRegression()
    model.fit(X, y)
    return model

def train_decision_tree(dataset_path):
    df = pd.read_csv(dataset_path)
    X = df[['desp', 'vel']].values
    y = df['jump'].values
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model

def train_mlp(dataset_path):
    df = pd.read_csv(dataset_path)
    X = df[['desp', 'vel']].values
    y = df['jump'].values
    model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=500)
    model.fit(X, y)
    return model

def sklearn_predict(model, desp, vel):
    try:
        pred = model.predict([[desp, vel]])[0]
        return bool(round(pred))
    except NotFittedError:
        return False

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
    dataset_path = "datos.csv"
    if mode in ('linear', 'tree', 'mlp'):
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

    bg_x = 0.0
    score = 0

    datosEntrenamiento = []

    pause_text = font_large.render("¡Game Over! Presiona R para reiniciar", True, (255, 0, 0))
    PLAYER_SPEED = 200

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

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and player_rect.bottom >= HEIGHT and not paused:
                    player_vel_y = -270.0
                    if jump_sound:
                        jump_sound.play()
                if event.key == pygame.K_r and paused:
                    bullet_active = False
                    player_rect.midbottom = (50, HEIGHT)
                    player_vel_y = 0.0
                    paused = False
                    score = 0
                    datosEntrenamiento.clear()
                    if mode in ('linear', 'tree', 'mlp') and os.path.exists(dataset_path):
                        # reentrenar modelo
                        if mode == 'linear':
                            model = train_linear_regression(dataset_path)
                        elif mode == 'tree':
                            model = train_decision_tree(dataset_path)
                        elif mode == 'mlp':
                            model = train_mlp(dataset_path)
                    mode_auto = (mode == 'auto')

        # Lógica de juego
        if not paused:
            bg_x = (bg_x - 100.0 * dt) % WIDTH

            # IA o ML salto
            if bullet_active and (mode_auto or mode in ('linear', 'tree', 'mlp')):
                despBala = player_rect.x - bullet_rect.x
                velocidadBala = abs(bullet_speed)
                estatusAire = 1 if player_rect.bottom < HEIGHT else 0
                estatuSuelo = 1 if player_rect.bottom >= HEIGHT else 0
                if mode_auto:
                    decision = random.choice([True, False])
                else:
                    decision = sklearn_predict(model, despBala, velocidadBala)
                if decision and player_rect.bottom >= HEIGHT:
                    player_vel_y = -270.0
                    if jump_sound:
                        jump_sound.play()

            player_vel_y += gravity * dt
            player_rect.y += player_vel_y * dt
            if player_rect.bottom >= HEIGHT:
                player_rect.bottom = HEIGHT
                player_vel_y = 0.0

            if not bullet_active:
                bullet_speed = -random.randint(300, 800)
                bullet_rect.midbottom = (WIDTH - 100, HEIGHT)
                bullet_active = True
            else:
                bullet_rect.x += bullet_speed * dt
                if bullet_rect.right < 0:
                    bullet_active = False
                    score += 1

            if bullet_active:
                despBala = player_rect.x - bullet_rect.x
                velocidadBala = abs(bullet_speed)
                estatusAire = 1 if player_rect.bottom < HEIGHT else 0
                estatuSuelo = 1 if player_rect.bottom >= HEIGHT else 0
                datosEntrenamiento.append({
                    'desp': despBala,
                    'vel': velocidadBala,
                    'suelo': estatuSuelo
                })

            if bullet_rect.colliderect(player_rect):
                paused = True
                if game_over_sound:
                    game_over_sound.play()

            anim_timer += dt
            if anim_timer >= ANIM_INTERVAL:
                anim_timer = 0.0
                anim_index = (anim_index + 1) % len(frames_run)

        # Dibujar
        screen.blit(bg, (bg_x - WIDTH, 0))
        screen.blit(bg, (bg_x, 0))
        ship_rect = ship_img.get_rect(midbottom=(WIDTH - 100, HEIGHT - 30))
        screen.blit(ship_img, ship_rect)
        screen.blit(bullet_img, bullet_rect)
        current_frame = frames_run[anim_index]
        screen.blit(current_frame, player_rect)
        score_text = font_small.render(f"Puntos: {score}", True, (255, 255, 0))
        screen.blit(score_text, (10, 10))
        if paused:
            screen.blit(pause_text, (WIDTH//2 - pause_text.get_width()//2, HEIGHT//2 - pause_text.get_height()//2))
        pygame.display.flip()

    # Al cerrar, guardamos el dataset actual en "datos.csv"
    if datosEntrenamiento:
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

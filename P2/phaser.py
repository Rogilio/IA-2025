import pygame
import random
import os
import sys
import time
import pandas as pd
import numpy as np
from collections import Counter

# ML
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

# ============================
#  CONSTANTES DEL JUEGO
# ============================
WIDTH, HEIGHT = 800, 400
FPS = 60

# SPRITE DEL JUGADOR
FRAME_WIDTH = 32
FRAME_HEIGHT = 48
SPRITE_ROWS = 1
SPRITE_COLS = 4
SCALE_FACTOR = 1

# ============================
#  VARIABLES GLOBALES IA & DATOS
# ============================
ia_model_perceptron = None
ia_model_tree = None
ia_model_knn = None
scaler_perceptron = None
scaler_knn = None

flag_perceptron = False
flag_tree = False
flag_knn = False

game_records = []
RECORD_INTERVAL = 0.05
last_record_time = 0.0
DATASET_FILE = "datos.csv"

# ============================
#  UTILIDADES DE CARGA DE IMÁGENES
# ============================
def load_img(path):
    if not os.path.isfile(path):
        tmp = pygame.Surface((32, 32))
        tmp.fill((255, 0, 255))
        pygame.draw.line(tmp, (0, 0, 0), (0, 0), (32, 32))
        pygame.draw.line(tmp, (0, 0, 0), (0, 32), (32, 0))
        print(f" Imagen no encontrada: {path}, usando placeholder")
        return tmp.convert_alpha()
    return pygame.image.load(path).convert_alpha()

# ============================
#  LIMPIEZA DE MODELOS
# ============================
def limpiar_modelos():
    global ia_model_perceptron, ia_model_tree, ia_model_knn, scaler_perceptron, scaler_knn
    global flag_perceptron, flag_tree, flag_knn
    ia_model_perceptron = ia_model_tree = ia_model_knn = None
    scaler_perceptron = scaler_knn = None
    flag_perceptron = flag_tree = flag_knn = False

# ============================
#  CARGA DE PARTIDAS PASADAS
# ============================
def cargar_registros_csv(ruta: str = DATASET_FILE) -> bool:
    global game_records
    if os.path.exists(ruta):
        try:
            df = pd.read_csv(ruta)
            game_records = df.values.tolist()
            print(f" {len(game_records)} registros cargados desde '{ruta}'.")
            return True
        except Exception as e:
            print(f"Fallo leyendo CSV: {e}")
    return False

# ============================
#  ENTRENAMIENTO DE MODELOS
# ============================
def _get_xy(min_samples: int = 40):
    if len(game_records) < min_samples:
        raise ValueError(f"Al menos {min_samples} muestras requeridas, hay {len(game_records)}.")
    arr = np.array(game_records, dtype=float)
    X = arr[:, :3]
    y = arr[:, 3].astype(int)
    return X, y

def entrenar_perceptron():
    global ia_model_perceptron, scaler_perceptron, flag_perceptron
    try:
        X, y = _get_xy(40)
        sm = SMOTE(sampling_strategy={1: 150}, random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        scaler_perceptron = StandardScaler()
        X_norm = scaler_perceptron.fit_transform(X_res)
        ia_model_perceptron = MLPClassifier(
            hidden_layer_sizes=(50,),
            max_iter=3000,
            random_state=24,
            activation='relu'
        )
        print("Entrenando perceptrón …")
        ia_model_perceptron.fit(X_norm, y_res)
        flag_perceptron = True
        print("Perceptrón entrenado.")
        return True
    except Exception as e:
        print(f"Fallo perceptrón: {e}")
    return False

def entrenar_arbol():
    global ia_model_tree, flag_tree
    try:
        X, y = _get_xy(40)
        sm = SMOTE(sampling_strategy={1: 100}, random_state=45)
        X_res, y_res = sm.fit_resample(X, y)
        params = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [4, 7],
            'min_samples_split': [4, 8],
            'min_samples_leaf': [2, 5]
        }
        grid = GridSearchCV(
            DecisionTreeClassifier(random_state=45, class_weight='balanced'),
            params,
            cv=3,
            scoring='f1_macro',
            n_jobs=-1
        )
        print("Buscando hiperparámetros Árbol …")
        grid.fit(X_res, y_res)
        best = grid.best_params_
        print("Árbol mejor config:", best)
        ia_model_tree = DecisionTreeClassifier(
            **best, class_weight='balanced', random_state=45)
        ia_model_tree.fit(X_res, y_res)
        flag_tree = True
        print("Árbol entrenado.")
        return True
    except Exception as e:
        print(f"Fallo árbol: {e}")
    return False

def entrenar_knn():
    global ia_model_knn, scaler_knn, flag_knn
    try:
        X, y = _get_xy(55)
        sm = SMOTE(sampling_strategy={1: 90}, random_state=25)
        X_res, y_res = sm.fit_resample(X, y)
        scaler_knn = StandardScaler()
        X_norm = scaler_knn.fit_transform(X_res)
        params = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'p': [1, 2],
        }
        grid = GridSearchCV(KNeighborsClassifier(), params, cv=3, scoring='f1_macro', n_jobs=-1)
        print("Buscando hiperparámetros KNN …")
        grid.fit(X_norm, y_res)
        best = grid.best_params_
        print("KNN mejor config:", best)
        ia_model_knn = KNeighborsClassifier(**best, metric='minkowski')
        ia_model_knn.fit(X_norm, y_res)
        flag_knn = True
        print("KNN entrenado.")
        return True
    except Exception as e:
        print(f"Fallo KNN: {e}")
    return False

# ============================
#  INFERENCIA Y ACCIÓN IA
# ============================
def predecir_accion(modelo, rect_j, rect_b, act_b, vel_b,
                    rect_b2, act_b2, vel_b2):
    try:
        dist_h = abs(rect_b.centerx - rect_j.centerx) if act_b else 999
        velocidad = abs(vel_b) if act_b else 0
        dist_v = abs(rect_b2.centery - rect_j.centery) if act_b2 else 999
        X_pred = np.array([[velocidad, dist_h, dist_v]], dtype=float)
        if modelo == ia_model_perceptron and scaler_perceptron is not None:
            X_pred = scaler_perceptron.transform(X_pred)
        elif modelo == ia_model_knn and scaler_knn is not None:
            X_pred = scaler_knn.transform(X_pred)
        return int(modelo.predict(X_pred)[0])
    except Exception as e:
        print(f"Error predicción: {e}")
        return 0

# ============================
#  RECOLECCIÓN DE DATOS MANUAL
# ============================
def recolectar_estado(jugador, bala, activa_bala, vel_bala,
                      bala2, activa_bala2, vel_bala2, keys, salto_now):
    velocidad = abs(vel_bala) if activa_bala else 0
    dist_h = abs(bala.centerx - jugador.centerx) if activa_bala else 999
    dist_v = abs(bala2.centery - jugador.centery) if activa_bala2 else 999
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
#  LOOP PRINCIPAL DEL JUEGO
# ============================
def main():
    global last_record_time, game_records

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Phaser Final – IA Adaptada")
    clock = pygame.time.Clock()

    # Fuentes y menú
    font_title = pygame.font.SysFont(None, 48)
    font_opt = pygame.font.SysFont(None, 36)
    font_sub = pygame.font.SysFont(None, 28)

    mode = None
    ia_algo = None

    def draw_menu(title, options):
        screen.fill((30, 30, 40))
        surf_t = font_title.render(title, True, (255, 255, 255))
        screen.blit(surf_t, (WIDTH//2 - surf_t.get_width()//2, 50))
        for i, line in enumerate(options):
            surf = font_opt.render(line, True, (200, 240, 230))
            screen.blit(surf, (WIDTH//2 - surf.get_width()//2, 140 + i*50))
        pygame.display.flip()

    # Menú principal
    while mode is None:
        draw_menu("Elige el modo de juego:", ["1. Manual", "2. Automático"])
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); return
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_1: mode = 'manual'
                elif ev.key == pygame.K_2: mode = 'auto'

    # Menú IA si corresponde
    if mode == 'auto':
        if not cargar_registros_csv():
            draw_menu("¡No hay datos!", ["Primero juega en Manual."])
            time.sleep(2)
            mode = 'manual'
        else:
            while ia_algo is None:
                draw_menu("Elige IA:", ["1. Árbol de decisión", "2. Perceptrón", "3. KNN"])
                sub = font_sub.render(f"Datos: {len(game_records)}", True, (150,210,180))
                screen.blit(sub, (WIDTH//2 - sub.get_width()//2, 110))
                pygame.display.flip()
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT: pygame.quit(); return
                    if ev.type == pygame.KEYDOWN:
                        if ev.key == pygame.K_1: ia_algo = 'tree'
                        elif ev.key == pygame.K_2: ia_algo = 'mlp'
                        elif ev.key == pygame.K_3: ia_algo = 'knn'

    # ---------------------------------------------------
    # ENTRENAMIENTO PREVIO
    # ---------------------------------------------------
    model = None
    if mode == 'auto':
        limpiar_modelos()
        ok = False
        if ia_algo == 'tree': ok = entrenar_arbol(); model = ia_model_tree
        if ia_algo == 'mlp':  ok = entrenar_perceptron(); model = ia_model_perceptron
        if ia_algo == 'knn':  ok = entrenar_knn(); model = ia_model_knn
        if not ok or model is None:
            print("No se entrenó IA, cambiando a manual.")
            mode = 'manual'

    # ---------------------------------------------------
    # CARGA DE RECURSOS
    # ---------------------------------------------------
    bg = load_img(os.path.join('assets', 'game', 'fondito2.png'))
    bg = pygame.transform.scale(bg, (WIDTH, HEIGHT))
    spritesheet = load_img(os.path.join('assets', 'sprites', 'altair2.png'))
    bullet_img = load_img(os.path.join('assets', 'sprites', 'purple_ball.png'))
    ship_img = load_img(os.path.join('assets', 'game', 'ufo.png'))

    pygame.mixer.init()
    try:
        snd_jump = pygame.mixer.Sound(os.path.join('assets', 'audio', 'jump.mp3'))
        snd_gameover = pygame.mixer.Sound(os.path.join('assets', 'audio', 'game_over.wav'))
    except pygame.error:
        snd_jump = snd_gameover = None

    # ---------------------------------------------------
    # PREPARAR ANIMACIÓN DEL JUGADOR
    # ---------------------------------------------------
    frames = []
    for r in range(SPRITE_ROWS):
        for c in range(SPRITE_COLS):
            rect = pygame.Rect(c*FRAME_WIDTH, r*FRAME_HEIGHT, FRAME_WIDTH, FRAME_HEIGHT)
            frame = spritesheet.subsurface(rect)
            frame = pygame.transform.scale(frame, (int(FRAME_WIDTH*SCALE_FACTOR), int(FRAME_HEIGHT*SCALE_FACTOR)))
            frames.append(frame)
    anim_idx, anim_timer, ANIM_INTERVAL = 0, 0.0, 1/8

    # ---------------------------------------------------
    # ESTADOS INICIALES DEL JUEGO
    # ---------------------------------------------------
    running, paused = True, False
    player_rect = frames[0].get_rect(midbottom=(50, HEIGHT))
    player_vel_y, gravity = 0.0, 900.0
    PLAYER_SPEED = 275

    bullet_rect = bullet_img.get_rect(midbottom=(WIDTH-100, HEIGHT))
    bullet_speed, bullet_active = -200.0, False

    bullet2_rect = bullet_img.get_rect(topleft=(50, 0))
    bullet2_speed, bullet2_active = 150, True

    bg_x, score = 0.0, 0
    session_data, just_jumped = [], False
    INIT_X, return_to_init, return_speed = 50, False, 250

    font_small = pygame.font.SysFont(None, 28)
    pause_text = font_title.render("¡Game Over! Pulsa R", True, (255,0,0))

    # ---------------------------------------------------
    # BUCLE PRINCIPAL
    # ---------------------------------------------------

    while running:
        dt = clock.tick(FPS)/1000.0
        keys = pygame.key.get_pressed()

         # ------------------------------------
        # 1. GESTIÓN DE EVENTOS
        # ------------------------------------
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE and player_rect.bottom >= HEIGHT and not paused and mode=='manual':
                    just_jumped = True
                    player_vel_y = -330.0
                    if snd_jump: snd_jump.play()
                if ev.key == pygame.K_r and paused:
                    bullet_active = False
                    bullet2_active, bullet2_rect.topleft = True, (50,0)
                    player_rect.midbottom, player_vel_y = (INIT_X, HEIGHT), 0.0
                    paused, score, return_to_init = False, 0, False

        # ------------------------------------
        # 2. LÓGICA DE JUEGO (solo si no está pausado)
        # ------------------------------------

        if not paused:
            # Fondo scroll
            bg_x = (bg_x - 100*dt) % WIDTH

            # Movimiento manual y regreso al centro
            if mode=='manual':
                moved = False
                if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                    player_rect.x -= PLAYER_SPEED*dt; moved=True; return_to_init=True
                elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                    player_rect.x += PLAYER_SPEED*dt; moved=True; return_to_init=True
                if not moved and return_to_init:
                    if player_rect.centerx > INIT_X:
                        player_rect.x -= return_speed*dt
                        if player_rect.centerx <= INIT_X:
                            player_rect.centerx = INIT_X; return_to_init=False
                    elif player_rect.centerx < INIT_X:
                        player_rect.x += return_speed*dt
                        if player_rect.centerx >= INIT_X:
                            player_rect.centerx = INIT_X; return_to_init=False
                player_rect.clamp_ip(pygame.Rect(0,0,WIDTH,HEIGHT))

            # Recolección datos (solo manual)
            now = time.time()
            if mode=='manual' and now-last_record_time>=RECORD_INTERVAL:
                session_data.append(recolectar_estado(
                    player_rect, bullet_rect, bullet_active, bullet_speed,
                    bullet2_rect, bullet2_active, bullet2_speed,
                    keys, just_jumped))
                last_record_time, just_jumped = now, False

            # Control IA
            if mode=='auto' and model is not None:
                action = predecir_accion(model, player_rect, bullet_rect, bullet_active, bullet_speed,
                                         bullet2_rect, bullet2_active, bullet2_speed)
                # Acción: 0 = nada, 1 = salto, 2 = izq, 3 = der
                if action == 1 and player_rect.bottom >= HEIGHT:
                    player_vel_y = -330.0
                    if snd_jump: snd_jump.play()
                elif action == 2:
                    player_rect.x = max(0, player_rect.x - PLAYER_SPEED * dt)
                elif action == 3:
                    player_rect.x = min(WIDTH - player_rect.width, player_rect.x + PLAYER_SPEED * dt)

            # Física
            player_vel_y += gravity*dt
            player_rect.y += player_vel_y*dt
            if player_rect.bottom >= HEIGHT: player_rect.bottom, player_vel_y = HEIGHT, 0.0

            # Bala horizontal
            if not bullet_active:
                bullet_speed = random.randint(-250, -150)
                bullet_rect.midbottom = (WIDTH-50, HEIGHT)
                bullet_active=True
            else:
                bullet_rect.x += bullet_speed*dt
                if bullet_rect.right < 0:
                    bullet_active=False; score+=1

            # Bala vertical
            if bullet2_active:
                bullet2_rect.y += bullet2_speed*dt
                if bullet2_rect.top>HEIGHT:
                    bullet2_speed = 150
                    bullet2_rect.topleft = (50,0)

            # Colisiones
            if (bullet_active and bullet_rect.colliderect(player_rect)) or \
               (bullet2_active and bullet2_rect.colliderect(player_rect)):
                paused=True
                if snd_gameover: snd_gameover.play()

        # Dibujado
        screen.blit(bg, (bg_x-WIDTH,0)); screen.blit(bg, (bg_x,0))
        screen.blit(ship_img, ship_img.get_rect(midbottom=(WIDTH-100, HEIGHT-30)))
        if bullet_active: screen.blit(bullet_img, bullet_rect)
        if bullet2_active: screen.blit(bullet_img, bullet2_rect)

        anim_timer += dt
        if anim_timer>=ANIM_INTERVAL:
            anim_timer=0.0; anim_idx=(anim_idx+1)%len(frames)
        screen.blit(frames[anim_idx], player_rect)

        screen.blit(font_small.render(f"Puntos: {score}", True, (255,255,0)), (10,10))
        if mode=='auto' and ia_algo:
            screen.blit(font_small.render(f"IA: {ia_algo.upper()}", True, (0,255,0)), (10,40))
        if paused:
            screen.blit(pause_text, (WIDTH//2 - pause_text.get_width()//2, HEIGHT//2 - pause_text.get_height()//2))

        pygame.display.flip()

    # ---------------------------------------------------
    # GUARDAR DATOS AL FINALIZAR
    # ---------------------------------------------------

    if mode=='manual' and session_data:
        df = pd.DataFrame(session_data, columns=['vel_bala','dist_h','dist_v','accion'])
        df.to_csv(DATASET_FILE, index=False)
        print(f"Datos guardados: {len(df)} registros en {DATASET_FILE}")
        print("Distribución de acciones:")
        for a, cnt in Counter(df['accion']).items():
            print(f"  Acción {a}: {cnt}")

    pygame.quit()

if __name__ == "__main__":
    main()

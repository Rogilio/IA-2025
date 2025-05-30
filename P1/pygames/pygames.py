import pygame
import time
import random
from queue import PriorityQueue

# configuración de animación (en segundos)
ANIMATION_DELAY = 0.01

# Colores
BLANCO  = (255, 255, 255)
NEGRO   = (  0,   0,   0)
GRIS    = (128, 128, 128)
VERDE   = (  0, 255,   0)
ROJO    = (255,   0,   0)
NARANJA = (255, 165,   0)
PURPURA = (128,   0, 128)
AZUL    = (  0,   0, 255)

class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = col * ancho
        self.y = fila * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas
        self.vecinos = []
        self.g = float("inf")
        self.h = float("inf")
        self.f = float("inf")
        self.padre = None

    def get_pos(self): return self.fila, self.col
    def es_cerrado(self):   return self.color == ROJO
    def es_abierto(self):   return self.color == VERDE
    def es_pared(self):     return self.color == NEGRO
    def es_inicio(self):    return self.color == NARANJA
    def es_fin(self):       return self.color == PURPURA

    def restablecer(self):  self.color = BLANCO
    def hacer_inicio(self): self.color = NARANJA
    def hacer_cerrado(self):self.color = ROJO
    def hacer_abierto(self):self.color = VERDE
    def hacer_pared(self):  self.color = NEGRO
    def hacer_fin(self):    self.color = PURPURA
    def hacer_camino(self): self.color = AZUL

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color,
                         (self.x, self.y, self.ancho, self.ancho))

    def actualizar_vecinos(self, grid):
        self.vecinos = []
        # abajo, arriba, derecha, izquierda
        if self.fila < self.total_filas-1 and not grid[self.fila+1][self.col].es_pared():
            self.vecinos.append(grid[self.fila+1][self.col])
        if self.fila > 0 and not grid[self.fila-1][self.col].es_pared():
            self.vecinos.append(grid[self.fila-1][self.col])
        if self.col < self.total_filas-1 and not grid[self.fila][self.col+1].es_pared():
            self.vecinos.append(grid[self.fila][self.col+1])
        if self.col > 0 and not grid[self.fila][self.col-1].es_pared():
            self.vecinos.append(grid[self.fila][self.col-1])

    def __lt__(self, otro): return False

def heuristica(p1, p2):
    x1,y1 = p1; x2,y2 = p2
    return abs(x1-x2) + abs(y1-y2)

def reconstruir_camino(came_from, actual, dibujar):
    longitud = 0
    while actual in came_from:
        actual = came_from[actual]
        actual.hacer_camino()
        longitud += 1
        dibujar()
        time.sleep(ANIMATION_DELAY)
    return longitud

def algoritmo_a_estrella(dibujar, grid, inicio, fin):
    # --- Preparar estructuras ---
    contador = 0
    open_set = PriorityQueue()
    open_set.put((0, contador, inicio))
    open_set_hash = {inicio}
    # nodos ya evaluados
    closed_set = set()
    came_from = {}
    inicio.g = 0
    inicio.f = heuristica(inicio.get_pos(), fin.get_pos())
    nodos_visitados = 0
    t0 = time.time()

    while not open_set.empty():
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
        # ─── 1) IMPRIMIR listas antes de sacar el siguiente nodo ───
        abierta = [f"{n.fila},{n.col}" for n in open_set_hash]
        cerrada = [f"{n.fila},{n.col}" for n in closed_set]
        print("Lista Abierta:", ", ".join(abierta))
        print("Lista Cerrada:",  ", ".join(cerrada))

        # ─── 2) Extraer nodo con menor f ───
        actual = open_set.get()[2]
        open_set_hash.remove(actual)
        closed_set.add(actual)          # marcamos actual como cerrado

        if actual == fin:
            dur = time.time() - t0
            # colorea el camino y devuelve la longitud 
            longitud = reconstruir_camino(came_from, fin, dibujar)
            # Restaurar colores de inicio/fin
            fin.hacer_fin(); inicio.hacer_inicio()

            # Reconstruir lista de nodos en el camino
            nodo = fin
            camino = []
            while nodo in came_from:
                camino.append(nodo)
                nodo = came_from[nodo]
            camino.reverse()

            # Imprimir g, h y f de cada nodo
            print("\n--- Detalle del camino ---")
            for n in camino:
                nombre = f"{n.fila},{n.col}"
                print(f"Nodo: {nombre}, g: {n.g:.2f}, h: {n.h:.2f}, f: {n.f:.2f}")

            # Costo total = g de la meta
            print(f"Costo Total del Camino: {fin.g:.2f}\n")
            print(f"Tiempo: {dur:.4f}s  Nodos: {nodos_visitados}  Longitud: {longitud}")
            print("/////////// CAMINO ENCONTRADO ///////////")
            return True

        # ─── 4) Procesar vecinos ───
        for vecino in actual.vecinos:
            temp_g = actual.g + 1
            if temp_g < vecino.g:
                came_from[vecino] = actual
                vecino.g = temp_g
                vecino.h = heuristica(vecino.get_pos(), fin.get_pos())
                vecino.f = vecino.g + vecino.h
                if vecino not in open_set_hash:
                    contador += 1
                    open_set.put((vecino.f, contador, vecino))
                    open_set_hash.add(vecino)
                    vecino.hacer_abierto()

        dibujar(); time.sleep(ANIMATION_DELAY)
        nodos_visitados += 1
        if actual != inicio:
            actual.hacer_cerrado()

    print("/////////// No se encontró camino ///////////")
    return False

def crear_grid(filas, ancho):
    grid = []; nodo_ancho = ancho // filas
    for i in range(filas):
        grid.append([Nodo(i,j,nodo_ancho,filas) for j in range(filas)])
    return grid

def dibujar_grid(ventana, filas, ancho):
    nodo_ancho = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, GRIS, (0,i*nodo_ancho),(ancho,i*nodo_ancho))
        pygame.draw.line(ventana, GRIS, (i*nodo_ancho,0),(i*nodo_ancho,ancho))

def dibujar(ventana, grid, filas, ancho):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)
    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()

def obtener_click_pos(pos, filas, ancho):
    nodo_ancho = ancho // filas
    x,y = pos
    return y//nodo_ancho, x//nodo_ancho

def reiniciar(grid, filas):
    for fila in grid:
        for nodo in fila:
            nodo.restablecer()
            nodo.g = nodo.h = nodo.f = float("inf")
            nodo.padre = None
    return None, None

def agregar_obstaculos(grid, prob=0.2):
    for fila in grid:
        for nodo in fila:
            if not nodo.es_inicio() and not nodo.es_fin():
                if random.random() < prob:
                    nodo.hacer_pared()

def guardar_mapa(grid, nombre="mapa.txt"):
    with open(nombre,"w") as f:
        for fila in grid:
            linea = "".join(
              "0" if nodo.color==BLANCO else
              "1" if nodo.color==NEGRO else
              #"2" if nodo.color==NARANJA else
              #"3" if nodo.color==PURPURA else
              "0"
              for nodo in fila)
            f.write(linea+"\n")
    print(f"Mapa guardado en {nombre}.")

def cargar_mapa(grid, filas, ancho, nombre="mapa.txt"):
    try:
        with open(nombre) as f:
            lines = f.read().splitlines()
        inicio = fin = None
        for i,line in enumerate(lines):
            for j,ch in enumerate(line):
                nodo = grid[i][j]
                nodo.restablecer()
                if ch=="1": nodo.hacer_pared()
                elif ch=="2": nodo.hacer_inicio(); inicio=nodo
                elif ch=="3": nodo.hacer_fin();    fin=nodo
        print(f"Mapa cargado desde {nombre}.")
        return inicio, fin
    except FileNotFoundError:
        print("No existe el archivo de mapa.")
        return None, None
    

def main(ventana, ancho):
    FILAS = 20

    # Estado: "edit" = editar (colocar start/goal/obstáculos)
    #         "searching" = A* corriendo
    #         "done" = A* terminado (solo R para reiniciar)
    estado = "edit"
    grid   = crear_grid(FILAS, ancho)
    mostrar_stats = True      # flag para imprimir estadísticas
    inicio = None
    fin    = None
    corriendo = True

    def reset_grid():
        nonlocal grid, inicio, fin, estado
        grid      = crear_grid(FILAS, ancho)
        inicio    = None
        fin       = None
        estado    = "edit"

    def random_obstacles(densidad=0.2):
            # convierte aleatoriamente un % de nodos en paredes
            for fila in grid:
                for nodo in fila:
                    if nodo not in (inicio, fin) and random.random() < densidad:
                        nodo.hacer_pared()

    corriendo = True
    while corriendo:
        dibujar(ventana, grid, FILAS, ancho)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            # —————— TECLAS ——————
            if event.type == pygame.KEYDOWN:
                # Lanzar A*
                if event.key == pygame.K_SPACE and inicio and fin and estado == "edit":
                    estado = "searching"
                    for fila in grid:
                        for nodo in fila:
                            nodo.actualizar_vecinos(grid)
                    res = algoritmo_a_estrella(
                        lambda: dibujar(ventana, grid, FILAS, ancho),
                        grid, inicio, fin
                    )
                    estado = "done"

                # R = Reiniciar todo
                elif event.key == pygame.K_r:
                    reset_grid()

                # O = obstáculos aleatorios
                elif event.key == pygame.K_o and estado == "edit":
                    random_obstacles(densidad=0.25)

                # S = Guardar mapa
                elif event.key == pygame.K_s:
                    guardar_mapa(grid, "mapa.txt")


                # L = cargar mapa (implementación de ejemplo)
                elif event.key == pygame.K_l and estado == "edit":
                    try:
                        with open("mapa.txt") as f:
                            lines = f.read().splitlines()
                        for i, line in enumerate(lines):
                            for j, c in enumerate(line):
                                if c == "1":
                                    grid[i][j].hacer_pared()
                                else:
                                    grid[i][j].restablecer()
                        print("Mapa cargado desde 'mapa.txt'")
                    except Exception as e:
                        print("Error al cargar mapa:", e)


            # —————— CLICKS SOLO EN MODO EDIT y EVENTO DOWN ——————
            elif event.type == pygame.MOUSEBUTTONDOWN and estado == "edit":
                mx, my = event.pos
                # Solo si está dentro de la cuadrícula
                if 0 <= mx < ancho and 0 <= my < ancho:
                    fila, col = obtener_click_pos((mx, my), FILAS, ancho)
                    nodo = grid[fila][col]
                    # Botón izquierdo
                    if event.button == 1:
                        if not inicio and nodo != fin:
                            inicio = nodo
                            nodo.hacer_inicio()
                        elif not fin and nodo != inicio:
                            fin = nodo
                            nodo.hacer_fin()
                        elif nodo not in (inicio, fin):
                            nodo.hacer_pared()
                    # Botón derecho
                    elif event.button == 3:
                        nodo.restablecer()
                        if nodo == inicio:
                            inicio = None
                        elif nodo == fin:
                            fin = None

            # ———— RATÓN ————
            # Pintar/arrastrar paredes SOLO en modo edit
            elif estado == "edit" and event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
                buttons = pygame.mouse.get_pressed()
                mx, my = pygame.mouse.get_pos()
                # si sale fuera, ignoramos
                if not (0 <= mx < ancho and 0 <= my < ancho):
                    continue
                fila, col = obtener_click_pos((mx, my), FILAS, ancho)
                nodo = grid[fila][col]
                # botón izquierdo arrastra paredes
                if buttons[0]:
                    if not inicio and nodo != fin:
                        inicio = nodo; nodo.hacer_inicio()
                    elif inicio and not fin and nodo not in (inicio,):
                        fin = nodo; nodo.hacer_fin()
                    elif nodo not in (inicio, fin):
                        nodo.hacer_pared()
                # botón derecho arrastra borrando
                if buttons[2]:
                    nodo.restablecer()
                    if nodo == inicio: inicio = None
                    if nodo == fin:    fin    = None

        pygame.display.update()

    pygame.quit()


if __name__=="__main__":
    pygame.init()
    info = pygame.display.Info()
    mw, mh = info.current_w, info.current_h
    MARGEN = 100
    ANCHO = min(mw,mh) - MARGEN
    ventana = pygame.display.set_mode((ANCHO,ANCHO))
    pygame.display.set_caption("A* Visualizador de rutas")
    main(ventana, ANCHO)

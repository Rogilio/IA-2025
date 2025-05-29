import pygame
from queue import PriorityQueue
import time

# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)
VERDE = (0, 255, 0)
ROJO = (255, 0, 0)
NARANJA = (255, 165, 0)
PURPURA = (128, 0, 128)
AZUL = (0, 0, 255)

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

    def get_pos(self):
        return self.fila, self.col

    def es_cerrado(self):
        return self.color == ROJO

    def es_abierto(self):
        return self.color == VERDE

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA

    def restablecer(self):
        self.color = BLANCO

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_cerrado(self):
        self.color = ROJO

    def hacer_abierto(self):
        self.color = VERDE

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA

    def hacer_camino(self):
        self.color = AZUL

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))

    def actualizar_vecinos(self, grid):
        self.vecinos = []
        if self.fila < self.total_filas - 1 and not grid[self.fila + 1][self.col].es_pared():
            self.vecinos.append(grid[self.fila + 1][self.col])
        if self.fila > 0 and not grid[self.fila - 1][self.col].es_pared():
            self.vecinos.append(grid[self.fila - 1][self.col])
        if self.col < self.total_filas - 1 and not grid[self.fila][self.col + 1].es_pared():
            self.vecinos.append(grid[self.fila][self.col + 1])
        if self.col > 0 and not grid[self.fila][self.col - 1].es_pared():
            self.vecinos.append(grid[self.fila][self.col - 1])

    def __lt__(self, other):
        return False

def heuristica(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruir_camino(came_from, actual, dibujar):
    longitud = 0
    while actual in came_from:
        actual = came_from[actual]
        actual.hacer_camino()
        longitud += 1
        dibujar()
    return longitud

def algoritmo_a_estrella(dibujar, grid, inicio, fin):
    contador = 0
    open_set = PriorityQueue()
    open_set.put((0, contador, inicio))
    came_from = {}
    inicio.g = 0
    inicio.f = heuristica(inicio.get_pos(), fin.get_pos())

    open_set_hash = {inicio}
    nodos_visitados = 0
    inicio_tiempo = time.time()

    while not open_set.empty():
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()

        actual = open_set.get()[2]
        open_set_hash.remove(actual)

        if actual == fin:
            duracion = time.time() - inicio_tiempo
            longitud = reconstruir_camino(came_from, fin, dibujar)
            fin.hacer_fin()
            inicio.hacer_inicio()
            print(f"✔ Camino encontrado.")
            print(f"🕒 Tiempo de ejecución: {duracion:.4f} segundos")
            print(f"🔁 Nodos visitados: {nodos_visitados}")
            print(f"📏 Longitud del camino: {longitud} nodos")
            return True

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

        dibujar()
        nodos_visitados += 1
        if actual != inicio:
            actual.hacer_cerrado()

    print("❌ No se encontró camino.")
    return False

def crear_grid(filas, ancho):
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    return grid

def dibujar_grid(ventana, filas, ancho):
    ancho_nodo = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, GRIS, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, GRIS, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))

def dibujar(ventana, grid, filas, ancho):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)
    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()

def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    x, y = pos
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col

def main(ventana, ancho):
    FILAS = 20
    grid = crear_grid(FILAS, ancho)

    inicio = None
    fin = None

    corriendo = True

    while corriendo:
        dibujar(ventana, grid, FILAS, ancho)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                if not inicio and nodo != fin:
                    inicio = nodo
                    inicio.hacer_inicio()
                elif not fin and nodo != inicio:
                    fin = nodo
                    fin.hacer_fin()
                elif nodo != fin and nodo != inicio:
                    nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                nodo.restablecer()
                if nodo == inicio:
                    inicio = None
                elif nodo == fin:
                    fin = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and inicio and fin:
                    for fila in grid:
                        for nodo in fila:
                            nodo.actualizar_vecinos(grid)
                    algoritmo_a_estrella(lambda: dibujar(ventana, grid, FILAS, ancho), grid, inicio, fin)

    pygame.quit()

# ─── IMPORTANTE: inicializar Pygame antes de usar el display ───
pygame.init()

info = pygame.display.Info()
screen_w, screen_h = info.current_w, info.current_h
#ANCHO_VENTANA = 800
MARGEN = 100
ANCHO_VENTANA = min(screen_w, screen_h) - MARGEN
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Visualización de Nodos - A*")
main(VENTANA, ANCHO_VENTANA)

# Documentación del Proyecto: Visualizador del Algoritmo A\*

Este documento describe el funcionamiento interno del código contenido en `pygames.py`, que implementa una visualización interactiva del algoritmo A\* utilizando `pygame`.

---

## Informacion

**Alumno:** Rogelio Cristian Punzo Castro  
**Materia:** Inteligencia Artificial  
**No. Control:** 21120245
**Docente:** Jesus Eduardo Alcaraz Chavez  
**Institución:** Instituto Tecnologico de Morelia
**Fecha:** Junio 2025

---

## Estructura General

El programa se divide en varias secciones claves:

1. **Importaciones y Configuraciones**
2. **Definición de la clase `Nodo`**
3. **Funciones auxiliares para dibujo y eventos**
4. **Implementación del algoritmo A**\*
5. **Gestores de mapas y obstáculos**
6. **Bucle principal `main()` y control de eventos**
7. **Punto de entrada del programa**

---

## 1. Importaciones y Configuraciones

```python
import pygame
import time
import random
from queue import PriorityQueue
```

Se importan librerías necesarias: `pygame` para interfaz gráfica, `time` para demoras animadas, `random` para obstáculos aleatorios y `PriorityQueue` para el conjunto abierto del algoritmo A\*.

Colores y parámetros visuales están definidos como constantes globales.

---

## 2. Clase `Nodo`

La clase `Nodo` representa cada celda de la cuadrícula. Guarda información como:

* Posición (fila, columna, coordenadas en pantalla)
* Tipo de nodo (inicio, fin, pared, camino, etc.)
* Costos `g`, `h`, `f` para A\*
* Vecinos accesibles

Contiene métodos para cambiar su color/estado, dibujarse en pantalla, y actualizar vecinos accesibles (movimiento en 4 direcciones).

---

## 3. Funciones Auxiliares

* `heuristica(p1, p2)`: Calcula distancia Manhattan.
* `reconstruir_camino(came_from, actual, dibujar)`: Marca visualmente el camino encontrado.
* `crear_grid(filas, ancho)`: Crea una matriz de nodos.
* `dibujar_grid` y `dibujar`: Renderizan la cuadrícula en pantalla.
* `obtener_click_pos`: Convierte un clic en coordenadas de nodo.
* `reiniciar`, `agregar_obstaculos`, `guardar_mapa`, `cargar_mapa`: Gestores del estado del tablero.

---

## 4. Algoritmo A\*

La función `algoritmo_a_estrella` implementa la lógica principal del algoritmo:

* Usa `PriorityQueue` para almacenar nodos abiertos (según `f`).
* `came_from` para reconstruir camino.
* Revisa nodos vecinos y actualiza sus costos si se encuentra una mejor ruta.
* Termina al alcanzar el nodo fin o si no hay solución.

Incluye impresión en consola del detalle de nodos abiertos/cerrados y del camino final con sus costos.

---

## 5. `main()` y Control de Eventos

El bucle principal de `main()` inicializa `pygame`, crea la ventana y la cuadrícula, y escucha eventos:

* **Clic izquierdo/derecho:** Coloca/borrar nodos.
* **Teclas:**

  * SPACE: ejecuta A\*
  * R: reinicia tablero
  * O: obstáculos aleatorios
  * S/L: guardar/cargar mapas

Incluye modo de dibujo al arrastrar el mouse.

---

## 6. Punto de Entrada

```python
if __name__=="__main__":
    pygame.init()
    ...
    main(ventana, ANCHO)
```

Este bloque detecta si el script está siendo ejecutado directamente e inicia la aplicación.

---

## Observaciones

* La clase `Nodo` debería tener una implementación adecuada de `__lt__` para funcionar bien con `PriorityQueue`.
* El código está bien modularizado y permite una buena extensibilidad.
* Las animaciones y estados visuales hacen que el algoritmo sea fácil de seguir.

---

## Codigo

---

```python
...
# (Sección previa del código documentado omitida por brevedad)
...

# Función principal que controla la ejecución del programa
def main(ventana, ancho):
    FILAS = 20

    # Estados posibles del programa
    # 'edit': colocar inicio/fin/obstáculos
    # 'searching': algoritmo en ejecución
    # 'done': búsqueda terminada
    estado = "edit"
    grid = crear_grid(FILAS, ancho)
    mostrar_stats = True
    inicio = None
    fin = None
    corriendo = True

    def reset_grid():
        nonlocal grid, inicio, fin, estado
        grid = crear_grid(FILAS, ancho)
        inicio = None
        fin = None
        estado = "edit"

    def random_obstacles(densidad=0.2):
        # Genera obstáculos aleatorios en nodos libres
        for fila in grid:
            for nodo in fila:
                if nodo not in (inicio, fin) and random.random() < densidad:
                    nodo.hacer_pared()

    while corriendo:
        dibujar(ventana, grid, FILAS, ancho)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if event.type == pygame.KEYDOWN:
                # Ejecutar A*
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

                # Reiniciar todo
                elif event.key == pygame.K_r:
                    reset_grid()

                # Obstáculos aleatorios
                elif event.key == pygame.K_o and estado == "edit":
                    random_obstacles(densidad=0.25)

                # Guardar mapa
                elif event.key == pygame.K_s:
                    guardar_mapa(grid, "mapa.txt")

                # Cargar mapa
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

            # Clicks del mouse en modo edición
            elif event.type == pygame.MOUSEBUTTONDOWN and estado == "edit":
                mx, my = event.pos
                if 0 <= mx < ancho and 0 <= my < ancho:
                    fila, col = obtener_click_pos((mx, my), FILAS, ancho)
                    nodo = grid[fila][col]
                    if event.button == 1:
                        if not inicio and nodo != fin:
                            inicio = nodo
                            nodo.hacer_inicio()
                        elif not fin and nodo != inicio:
                            fin = nodo
                            nodo.hacer_fin()
                        elif nodo not in (inicio, fin):
                            nodo.hacer_pared()
                    elif event.button == 3:
                        nodo.restablecer()
                        if nodo == inicio: inicio = None
                        elif nodo == fin: fin = None

            # Arrastrar para pintar/borrar nodos
            elif estado == "edit" and event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
                buttons = pygame.mouse.get_pressed()
                mx, my = pygame.mouse.get_pos()
                if not (0 <= mx < ancho and 0 <= my < ancho):
                    continue
                fila, col = obtener_click_pos((mx, my), FILAS, ancho)
                nodo = grid[fila][col]
                if buttons[0]:
                    if not inicio and nodo != fin:
                        inicio = nodo; nodo.hacer_inicio()
                    elif inicio and not fin and nodo not in (inicio,):
                        fin = nodo; nodo.hacer_fin()
                    elif nodo not in (inicio, fin):
                        nodo.hacer_pared()
                if buttons[2]:
                    nodo.restablecer()
                    if nodo == inicio: inicio = None
                    if nodo == fin:    fin = None

        pygame.display.update()

    pygame.quit()

# Punto de entrada principal del programa
if __name__ == "__main__":
    pygame.init()
    info = pygame.display.Info()
    mw, mh = info.current_w, info.current_h
    MARGEN = 100
    ANCHO = min(mw, mh) - MARGEN
    ventana = pygame.display.set_mode((ANCHO, ANCHO))
    pygame.display.set_caption("A* Visualizador de rutas")
    main(ventana, ANCHO)
```

---

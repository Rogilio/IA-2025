# Modelar una red neuronal que pueda jugar al 5 en línea sin gravedad en un tablero de 20x20

## Definir el tipo de red neuronal y describir cada una de sus partes

Una **red neuronal convolucional**:

- **Entrada**: el tablero 20x20, unas 400 posibles posiciones. En cada celda del tablero hay un valor que indica si la celda está vacía o si está ocupada por un jugador o un oponente.
- **Capas ocultas**: encargadas de extraer características del tablero.
- **Salida**: nos da la probabilidad de que cada movimiento sea bueno.

## Definir los patrones a utilizar

Se deben identificar los patrones en el tablero que llevan a una victoria o una situación ventajosa.

## Definir la función de activación necesaria para este problema

- **ReLU**: debido a que en este problema los patrones, como las líneas de fichas, no son lineales.
- **Softmax**: convierte un vector de valores en probabilidades. En nuestro problema, permite decidir cuál es la mejor celda para colocar la ficha, de acuerdo con la que tenga la probabilidad más alta.

## Definir el número máximo de entradas

Dado que el tablero es de **20x20**, en total hay **400 posiciones posibles**, por lo que el número máximo de entradas será **400**.  
Cada casilla puede estar vacía, ocupada por un jugador o por un oponente.

## ¿Qué valores se podrían esperar a la salida de la red?

- Un valor que representa la probabilidad en cada casilla de ser la mejor opción para colocar la ficha.

## ¿Cuáles son los valores máximos que puede tener el bias?

- Se usarán valores pequeños entre **-1 y 1**, aunque pueden variar en cada entrenamiento.

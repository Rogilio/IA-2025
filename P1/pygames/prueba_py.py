import pygame

pygame.init()
ventana = pygame.display.set_mode((400, 400))
pygame.display.set_caption("Prueba de Pygame")

corriendo = True
while corriendo:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            corriendo = False

pygame.quit()

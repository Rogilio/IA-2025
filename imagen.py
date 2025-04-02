import cv2 as cv
import numpy as np

# La función videocapture busca alguna entrada de video (el 0 significa que va a hacer un tipo broadcast, se puede
#                                                       poner el indice de la camara)
cap = cv.VideoCapture(0)

# En el while true se tienen que poner dos valores
while True:
    ret, img = cap.read()
    if(ret): # El ret lo que va a decir es si efectivamente estoy leyendo la cámara
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        cv.imshow('salida', img) #Cuidado con poner espacios porque generaria muchos frames
    k=cv.waitKey(1) & 0xff 
    if k == 27: 
        break

cap.release()
cv.destroyAllWindows()
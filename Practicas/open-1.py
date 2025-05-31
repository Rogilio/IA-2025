#+BEGIN_SRC python
import cv2
import numpy as np

img = cv2.imread("C://Users//rogel//OneDrive//Imágenes//tuxis.jpg", 1)
imgn = np.zeros(img.shape[:2], np.uint8)
print(img.shape)
b,g,r = cv2.split(img)
imgb = cv2.merge([b, imgn, imgn])
imgg = cv2.merge([imgn, g, imgn])
imgr = cv2.merge([imgn, imgn, r])
imgnn = cv2.merge([a,r,b])
#img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img4 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#cv2.imshow('salida',img)
cv2.imshow('salidab',imgb)
cv2.imshow('salidag',imgg)
cv2.imshow('salidar',imgr)
cv2.imshow('salidan',imgnn)
#cv2.imshow('salida5',imgn)

cv2.waitKey(0)
cv2.destroyAllWindows()

#+END_SRC

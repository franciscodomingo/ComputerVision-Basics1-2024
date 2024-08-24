import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================================
# PROBLEMA 1 – Ecualización local de histograma
# =============================================

"""
Problema 1: Desarrolle una función para implementar la ecualización local del histograma, que reciba como 
parámetros de entrada la imagen a procesar, y el tamaño de la ventana de procesamiento (M x N). 
Utilice dicha función para analizar la imagen e informe cuales son los detalles escondidos en las diferentes 
zonas de la misma. Analice la influencia del tamaño de la ventana en los resultados obtenidos.
"""

# Imagen Original
img1 = cv2.imread("Imagen_con_detalles_escondidos.tif")
plt.imshow(img1), 
plt.show(block=False) 
# plt.show() 

cantidad_de_pixeles = 10

img1_p = cv2.copyMakeBorder(img1, cantidad_de_pixeles, cantidad_de_pixeles, cantidad_de_pixeles, cantidad_de_pixeles, cv2.BORDER_CONSTANT)

cv2.imshow('Imagen Original', img1)
plt.show(block=False) 
cv2.imshow('Imagen con Bordes', img1_p)
cv2.waitKey(0)

def local_histogram_equalization(image, window_size):
    # Convertimos la imagen a escala de grises si no lo está
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # else:
    #     gray_image = image

    # Dimensiones de la imagen
    height, width = gray_image.shape

    # Iterar sobre la imagen con el tamaño de la ventana especificado
    for y in range(0, height - window_size[0] + 1, window_size[0]):
        for x in range(0, width - window_size[1] + 1, window_size[1]):
            # Obtener la región de interés
            roi = gray_image[y:y+window_size[0], x:x+window_size[1]]
            equalized_roi = cv2.equalizeHist(roi)
            gray_image[y:y+window_size[0], x:x+window_size[1]] = equalized_roi

    return gray_image


# Definimos diferentes tamaños de ventana para la ecualización local del histograma
tamanos_ventana = [(32, 32), (64, 64), (128, 128)]

# Analizamos la imagen con diferentes tamaños de ventana
for tv in tamanos_ventana:
    # Realizamos la ecualización local del histograma
    image_eq = local_histogram_equalization(img1, tv)

    # Imagen resultante
    cv2.imshow(f'Imagen con ventana {tv}', image_eq)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

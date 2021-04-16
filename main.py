"""
 Aalgoritmo Shi-Tomasi Corner Detector & Good Features to Track

 Vamos a buscar aquellas zonas con una variacion de imagenes muy grande. De esta forma obtendremos los puntos de interés.
 La idea del algoritmo es obtener aquellos puntos que hacen “esquina” mirando si tiene dos gradientes muy grandes en perpendicular.
 La peculiardidad de este algoritmo es que podemos quedarnos con las N mejor esquinas. Por tanto, los puntos estarán distribuidos más
 uniformemente que en el detector de Harrris-Stephan.

"""
import numpy as np
import cv2

camara = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    _, imagen = camara.read()

    # La imágnen a color se trnsforma en una imágen en gris
    imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    """
     Se obtienen las N mejor imágenes.
     
      1 - Imágen sobre la que buscar las esquinas. 
    
      2 - Cantidad de esquinas deseadas. Son las mejores esquinas.
      
      3 - Calidad mínima que debe tener una esquina. Se obtiene la mejor esquina y a partir de ella se obtiene la calidad.
          Si la mejor esquina tiene una calidad de 1500 y nosotros metemos como parámetro 0.01, todas las esquinas con una 
          calidad inferior a 15 se rechazarán (1500 * 0.01). 
          
      4 - Distancia mímina que ha de existrir entre las esquinas. Es lo que nos permite crear una distribución uniforme de
          las esquinas.    
    """
    esquinas = cv2.goodFeaturesToTrack(imagenGris, 25, 0.01, 10)

    # Introducimos las esquinas en un vector de tipo int64
    esquinas = np.int0(esquinas)

    # Se dibujan círculos donde se situen las esquinas
    for i in esquinas:
        x,y = i.ravel()
        cv2.circle(imagen, (x,y), 3, 255, -1)

    cv2.imshow("Harris - color", imagen);
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
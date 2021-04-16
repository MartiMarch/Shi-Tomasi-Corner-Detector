"""
 Aalgoritmo Shi-Tomasi Corner Detector & Good Features to Track

 Vamos a buscar aquellas zonas con una variacion de imagenes muy grande. De esta forma obtendremos los puntos de interés.
 La idea del algoritmo es obtener aquellos puntos que hacen “esquina” mirando si tiene dos gradientes muy grandes en perpendicular.
 La peculiardidad de este algoritmo es que podemos quedarnos con las N mejor esquinas. Por tanto, los puntos estarán distribuidos más
 uniformemente que en el detector de Harrris-Stephan.

"""
import numpy as np
import cv2
from datetime import datetime

camara = cv2.VideoCapture(1, cv2.CAP_DSHOW)
umbralSuperior = 1.10
umbralInferior = 0.90
now = 0
tiempoActual = 0
nEsquinas = 0

while True:
    _, imagen = camara.read()

    # La imagnen a color se transforma en una imagen en gris
    imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    """
     Se obtienen las N mejor imágenes.
     
      1 - Imagen sobre la que buscar las esquinas. 
    
      2 - Cantidad de esquinas deseadas. Son las mejores esquinas.
      
      3 - Calidad mínima que debe tener una esquina. Se obtiene la mejor esquina y a partir de ella se obtiene la calidad.
          Si la mejor esquina tiene una calidad de 1500 y nosotros metemos como parámetro 0.01, todas las esquinas con una 
          calidad inferior a 15 se rechazarán (1500 * 0.01). 
          
      4 - Distancia mímina que ha de existrir entre las esquinas. Es lo que nos permite crear una distribución uniforme de
          las esquinas a lo largo de la imágen.    
    """
    esquinas = cv2.goodFeaturesToTrack(imagenGris, cv2.countNonZero(imagenGris), 0.01, 10)

    # Introducimos las esquinas en un vector de tipo int64
    esquinas = np.int0(esquinas)

    # Se dibujan círculos allá donde se encuentre una esquina y se cuenta cuantas esquinas existen.
    nEsquinas = 0
    for i in esquinas:
        x,y = i.ravel()
        cv2.circle(imagen, (x,y), 3, 255, -1)
        nEsquinas = nEsquinas + 1

    # Cada 15 minutos se renuevan los puntos de referencia usados para comparar
    now = datetime.now()
    if now.timestamp() - tiempoActual > 9000000 or tiempoActual == 0:
        tiempoActual = now.timestamp()
        puntosRelevantes = nEsquinas

    # Para saber si se ha producido movimiento lo que se hace es comparar si la cantidad de puntos entra dentro de los umbrales.
    if nEsquinas > (puntosRelevantes * umbralSuperior) or nEsquinas < (puntosRelevantes * umbralInferior):
        print("MOVIMIENTO")

    cv2.imshow("Shi-Tomasi - color", imagen);
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
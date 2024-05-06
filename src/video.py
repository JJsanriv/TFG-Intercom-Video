#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

import os
import signal
import argparse
import sounddevice as sd  # If "pip install sounddevice" fails, install the "libportaudio2" system package
import numpy as np
import cv2
import socket
import time
import psutil
import logging
import soundfile as sf
import logging
from pygrabber.dshow_graph import FilterGraph

FORMAT = "(%(levelname)s) %(module)s: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


def spinning_cursor():
    ''' https://stackoverflow.com/questions/4995733/how-to-create-a-spinning-command-line-cursor'''
    while True:
        for cursor in '|/-\\':
            yield cursor

spinner = spinning_cursor()

class Minimal_video:

    @staticmethod
    def check_webcam_available():
        # Intenta abrir la primera cámara (índice 0)
        cap = cv2.VideoCapture(0)

        # Verifica si la cámara se abrió correctamente
        if not cap.isOpened():
            print("No se pudo abrir la webcam. Asegúrate de tener una webcam disponible.")
            return False
        else:
            print("Webcam detectada")
            # Obtener el ancho y alto de la imagen capturada
            
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print("Resolución de la cámara:", width, "x", height)
            cap.release()  # Liberar la cámara
            return True

    @staticmethod
    def capture_video():
        cap = cv2.VideoCapture(0)  # 0 indica la primera cámara (webcam) disponible

        while True:
            ret, frame = cap.read()  # Captura un fotograma desde la cámara
            cv2.imshow('Video', frame)  # Muestra el fotograma en una ventana llamada 'Video'
            print(next(spinner), end='\b', flush=True)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Si se presiona 'q', sal del bucle
                break

        cap.release()  # Libera la cámara
        cv2.destroyAllWindows()  # Cierra todas las ventanas
        

try:
    import argcomplete  # <tab> completion for argparse.
except ImportError:
    logging.warning("Unable to import argcomplete (optional)")

if __name__ == "__main__":   

    # Ejecutar la clase Minimal_video solo si se establece el flag -video
    if Minimal_video.check_webcam_available():
        Minimal_video.capture_video()
    else:
        print("No es posible realizar la conexión debido a que no hay una webcam disponible.")

#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""Over minimal, implements the use of a video stream while minimal runs normally."""

import socket
import threading
import argparse
import cv2
import numpy as np

from minimal import Minimal

class Minimal_Video(Minimal):
    def __init__(self, args=None):
        # Parsear argumentos propios (chunk size, puerto de video)
        parser = argparse.ArgumentParser(parents=[Minimal.get_arg_parser()], add_help=False)
        parser.add_argument('--video_port', type=int, default=5001, help="Puerto UDP para vídeo")
        parser.add_argument('--video_chunk_lines', type=int, default=10, help="Número de líneas por chunk de vídeo")
        self.args, _ = parser.parse_known_args(args)

        super().__init__(self.args)

        self.video_port = self.args.video_port
        self.video_chunk_lines = self.args.video_chunk_lines
        self.video_ip = self.args.address  # Heredado de Minimal (-a)
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') else 0)

        # Obtener resolución nativa
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Resolución nativa de la cámara: {self.width}x{self.height}")

        # Lanzar hilo para enviar vídeo
        self.stop_event = threading.Event()
        self.video_thread = threading.Thread(target=self.send_video_loop)
        self.video_thread.start()

    @staticmethod
    def get_arg_parser():
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--video_port', type=int, default=5001)
        parser.add_argument('--video_chunk_lines', type=int, default=10)
        return parser

    def send_video_loop(self):
        chunk_lines = self.video_chunk_lines
        while not self.stop_event.is_set():
            ret, frame = self.capture.read()
            if not ret:
                continue
            # Convertir el frame a RGB si es necesario
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Trocear el frame en chunks de líneas
            for start_line in range(0, self.height, chunk_lines):
                end_line = min(start_line + chunk_lines, self.height)
                chunk = frame[start_line:end_line, :, :].tobytes()
                header = np.array([start_line, end_line], dtype=np.uint16).tobytes()
                # El receptor debe saber reconstruir el frame con estos índices
                self.video_socket.sendto(header + chunk, (self.video_ip, self.video_port))
            # Opcional: sleep para limitar FPS
            # time.sleep(1/30)

    def close(self):
        self.stop_event.set()
        self.video_thread.join()
        self.capture.release()
        self.video_socket.close()
        super().close()

    # Sobrescribir main o método de arranque si procede
    def run(self):
        try:
            super().run()
        finally:
            self.close()

# Bloque para ejecución directa desde terminal
if __name__ == '__main__':
    min_video = Minimal_Video()
    min_video.run()





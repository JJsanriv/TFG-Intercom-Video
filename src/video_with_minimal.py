#!/usr/bin/env python
import os
import signal
import argparse
import sys
import sounddevice as sd
import numpy as np
import socket
import logging
import soundfile as sf
import cv2
import struct
import time
import threading

FORMAT = "(%(levelname)s) %(module)s: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

def spinning_cursor():
    while True:
        for cursor in '|/-\\':
            yield cursor

spinner = spinning_cursor()

def int_or_str(text):
    try:
        return int(text)
    except ValueError:
        return text

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input-device", type=int_or_str, help="Input device ID or substring")
parser.add_argument("-o", "--output-device", type=int_or_str, help="Output device ID or substring")
parser.add_argument("-d", "--list-devices", action="store_true", help="Print the available audio devices and quit")
parser.add_argument("-s", "--frames_per_second", type=float, default=44100, help="sampling rate in frames/second")
parser.add_argument("-c", "--frames_per_chunk", type=int, default=1024, help="Number of frames in a chunk")
parser.add_argument("-l", "--listening_port", type=int, default=4444, help="My listening port for audio")
parser.add_argument("-a", "--destination_address", type=str, default=None, help="Destination (interlocutor's listening) address")
parser.add_argument("-p", "--destination_port", type=int, default=4444, help="Destination (interlocutor's listening) port for audio")
parser.add_argument("-f", "--filename", type=str, help="Use a wav/oga/... file instead of the mic data")
parser.add_argument("-t", "--reading_time", type=int, help="Time reading data (mic or file) (only with effect if --show_stats or --show_data is used)")
parser.add_argument("--client", action='store_true', help="Set the device as client")
parser.add_argument("--server", action='store_true', help="Set the device as server")
args = parser.parse_args()

MAX_PAYLOAD_BYTES = 2800
VIDEO_PORT = 4445
VIDEO_FPS = 10
NUMBER_OF_CHANNELS = 2

class VideoAudioIntercom:

    # Inicializa los sockets, variables de configuración y prepara las fuentes de audio y video
    def __init__(self, args):
        self.args = args
        self.sock_audio = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_video = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listening_endpoint_audio = ("0.0.0.0", self.args.listening_port)
        self.listening_endpoint_video = ("0.0.0.0", VIDEO_PORT)
        self.sock_audio.bind(self.listening_endpoint_audio)
        if self.args.server:
            self.sock_video.bind(self.listening_endpoint_video)
            self.sock_video.listen(1)
        self.chunk_time = self.args.frames_per_chunk / self.args.frames_per_second
        self.destination_address = self.args.destination_address
        self.destination_port_audio = self.args.destination_port
        self.destination_port_video = VIDEO_PORT
        self.shutdown_flag = False
        self.zero_chunk = np.zeros((self.args.frames_per_chunk, NUMBER_OF_CHANNELS), np.int16)
        self.cap = None
        logging.info(f"chunk_time = {self.chunk_time} seconds")
        if self.args.filename:
            logging.info(f"Using \"{self.args.filename}\" as input")
            self.wavfile = sf.SoundFile(self.args.filename, 'r')
            self.audio_stream = self.file_stream
        else:
            self.audio_stream = self.mic_stream

    # Empaqueta el chunk de audio en formato de bytes
    def pack_audio(self, audio_chunk):
        return audio_chunk.tobytes()

    # Desempaqueta los bytes del chunk de audio en un array numpy
    def unpack_audio(self, packed_chunk):
        return np.frombuffer(packed_chunk, np.int16).reshape(-1, NUMBER_OF_CHANNELS)

    # Envía el chunk de audio empaquetado al destino
    def send_audio(self, packed_chunk):
        if packed_chunk is not None:
            try:
                self.sock_audio.sendto(packed_chunk, (self.destination_address, self.destination_port_audio))
            except Exception as e:
                logging.error(f"Failed to send audio: {e}")

    # Recibe un chunk de audio del destino y lo devuelve empaquetado
    def receive_audio(self):
        try:
            audio_chunk_size = self.args.frames_per_chunk * NUMBER_OF_CHANNELS * np.dtype(np.int16).itemsize
            audio_packed, _ = self.sock_audio.recvfrom(audio_chunk_size)
            return audio_packed
        except Exception as e:
            logging.error(f"Failed to receive audio: {e}")
            return None

    # Callback para grabar audio, enviarlo al destino, recibir audio del destino y reproducirlo
    def _record_IO_and_play(self, ADC, DAC, frames, time, status):
        if self.shutdown_flag:
            raise sd.CallbackAbort
        try:
            data = ADC.copy()
            packed_chunk = self.pack_audio(data)
            if self.destination_address:
                self.send_audio(packed_chunk)
                packed_chunk = self.receive_audio()
                if packed_chunk:
                    chunk = self.unpack_audio(packed_chunk)
                else:
                    chunk = self.zero_chunk
            else:
                chunk = data
            DAC[:] = chunk
        except Exception as e:
            logging.error(f"Error in audio processing: {e}")
        if __debug__:
            print(next(spinner), end='\b', flush=True)

    # Callback para grabar audio y enviarlo al destino siendo el dispositivo servidor
    def server_video(self, client_socket):
        # Configura la captura de video desde la cámara y el envío de frames al cliente
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)

        try:
            while not self.shutdown_flag:
                # Captura un frame de la cámara
                ret, frame = self.cap.read()
                if not ret:
                    break
                # Redimensiona el frame a 320x240
                frame = cv2.resize(frame, (320, 240))
                # Empaqueta el frame crudo
                data = frame.tobytes()

                try:
                    # Envía el tamaño del frame y los datos del frame al cliente
                    client_socket.sendall(struct.pack('<L', len(data)) + data)
                except Exception as e:
                    logging.error(f"Error sending video: {e}")
                    break

                try:
                    # Recibe el tamaño del frame del cliente
                    packed_length = client_socket.recv(4)
                    if not packed_length:
                        break
                    (length,) = struct.unpack('<L', packed_length)
                    data = b''
                    while len(data) < length:
                        to_read = length - len(data)
                        # Recibe los datos del frame del cliente
                        data += client_socket.recv(4096 if to_read > 4096 else to_read)

                    # Validar el tamaño del frame antes de continuar
                    if len(data) != 320 * 240 * 3:
                        logging.error(f"Received incorrect data size: {len(data)}. Expected: {320 * 240 * 3}")
                        continue

                    # Decodifica los datos del frame crudo y los convierte de nuevo a imagen
                    img = np.frombuffer(data, dtype=np.uint8).reshape(240, 320, 3)
                    cv2.imshow('Server', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    logging.error(f"Error receiving video: {e}")
                    break
        finally:
            self.cap.release()
            client_socket.close()
            cv2.destroyAllWindows()

    # Callback para grabar audio y enviarlo al destino siendo el dispositivo cliente
    def client_video(self):
        # Intenta conectar al servidor de video
        retry_count = 0
        max_retries = 5
        while retry_count < max_retries:
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect((self.destination_address, VIDEO_PORT))
                break
            except socket.error as e:
                logging.error(f"Connection attempt {retry_count + 1} failed: {e}")
                retry_count += 1
                time.sleep(2)
        else:
            logging.error(f"Failed to connect to video server after {max_retries} attempts")
            return

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)

        try:
            while not self.shutdown_flag:
                # Captura un frame de la cámara
                ret, frame = self.cap.read()
                if not ret:
                    break
                # Redimensiona el frame a 320x240
                frame = cv2.resize(frame, (320, 240))
                # Empaqueta el frame crudo
                data = frame.tobytes()

                try:
                    # Envía el tamaño del frame y los datos del frame al servidor
                    client_socket.sendall(struct.pack('<L', len(data)) + data)
                except Exception as e:
                    logging.error(f"Error sending video: {e}")
                    break

                try:
                    # Recibe el tamaño del frame del servidor
                    packed_length = client_socket.recv(4)
                    if not packed_length:
                        break
                    (length,) = struct.unpack('<L', packed_length)
                    data = b''
                    while len(data) < length:
                        to_read = length - len(data)
                        # Recibe los datos del frame del servidor
                        data += client_socket.recv(4096 if to_read > 4096 else to_read)

                    # Validar el tamaño del frame antes de continuar
                    if len(data) != 320 * 240 * 3:
                        logging.error(f"Received incorrect data size: {len(data)}. Expected: {320 * 240 * 3}")
                        continue

                    # Decodifica los datos del frame crudo y los convierte de nuevo a imagen
                    img = np.frombuffer(data, dtype=np.uint8).reshape(240, 320, 3)
                    cv2.imshow('Client', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    logging.error(f"Error receiving video: {e}")
                    break
        finally:
            self.cap.release()
            client_socket.close()
            cv2.destroyAllWindows()


        # Funcion para ejecutar video en modo exclusivamente local (sin flags de servidor o cliente)
        def local_video(self):
            # Configura la captura de video desde la cámara y muestra el video localmente
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            self.cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)

            try:
                while not self.shutdown_flag:
                    # Captura un frame de la cámara
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    # Muestra el frame capturado localmente
                    cv2.imshow('Local Video', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            finally:
                self.cap.release()
                cv2.destroyAllWindows()

    # Función para ejecutar el modo de video dependiendo de si es servidor, cliente o local
    def run_video(self):
        # Ejecuta el modo de video dependiendo de si es servidor, cliente o local
        if self.args.server:
            client_socket, client_address = self.sock_video.accept()
            logging.info(f"Accepted connection from {client_address}")
            self.server_video(client_socket)
        elif self.args.client:
            self.client_video()
        else:
            self.local_video()

    # Función para ejecutar el stream de audio
    def mic_stream(self, callback_function):
        try:
            with sd.Stream(device=(self.args.input_device, self.args.output_device),
                           dtype=np.int16,
                           samplerate=self.args.frames_per_second,
                           blocksize=self.args.frames_per_chunk,
                           channels=NUMBER_OF_CHANNELS,
                           callback=callback_function):
                self.run_video()
        except Exception as e:
            logging.error(f"Error in mic_stream: {e}")

    # Función para ejecutar el stream de audio desde un archivo
    def file_stream(self):
        try:
            self.run_video()
        except Exception as e:
            logging.error(f"Error in file_stream: {e}")

    # Función para ejecutar el programa
    def run(self):
        try:
            if self.args.filename:
                self.file_stream()
            else:
                self.mic_stream(self._record_IO_and_play)
        except Exception as e:
            logging.error(f"Error in run: {e}")

    # Función para cerrar los sockets y liberar recursos
    def shutdown(self):
        self.shutdown_flag = True
        self.sock_audio.close()
        self.sock_video.close()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

# Función para manejar la señal de interrupción (Ctrl+C)
def shutdown_handler(signum, frame):
    global intercom
    logging.info("Shutting down...")
    intercom.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown_handler)
    intercom = VideoAudioIntercom(args)
    
    def check_for_enter_key():
        while True:
            if input() == "":
                logging.info("Enter key pressed. Shutting down...")
                intercom.shutdown()
                sys.exit(0)
    
    enter_key_thread = threading.Thread(target=check_for_enter_key)
    enter_key_thread.daemon = True
    enter_key_thread.start()
    
    intercom.run()

    
    
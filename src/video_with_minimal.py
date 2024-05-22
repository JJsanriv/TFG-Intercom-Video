import signal
import argparse
import sounddevice as sd
import numpy as np
import socket
import time
import psutil
import logging
import soundfile as sf
import cv2
import threading
import sys
from queue import Queue
from threading import Event

# Definir el formato del log
FORMAT = "(%(levelname)s) %(module)s: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)

# Función para el cursor giratorio
def spinning_cursor():
    ''' https://stackoverflow.com/questions/4995733/how-to-create-a-spinning-command-line-cursor'''
    while True:
        for cursor in '|/-\\':
            yield cursor

spinner = spinning_cursor()

# Helper function for argument parsing
def int_or_str(text):
    try:
        return int(text)
    except ValueError:
        return text

# Configuración del parser para argumentos
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input-device", type=int_or_str, help="Input device ID or substring")
parser.add_argument("-o", "--output-device", type=int_or_str, help="Output device ID or substring")
parser.add_argument("-d", "--list-devices", action="store_true", help="Print the available audio devices and quit")
parser.add_argument("-s", "--frames_per_second", type=float, default=44100, help="sampling rate in frames/second")
parser.add_argument("-c", "--frames_per_chunk", type=int, default=4096, help="Number of frames in a chunk")
parser.add_argument("-l", "--listening_port", type=int, default=4444, help="My listening port")
parser.add_argument("-a", "--destination_address", type=str, help="Destination (interlocutor's listening) address")
parser.add_argument("-p", "--destination_port", type=int, default=4444, help="Destination (interlocutor's listening) port")
parser.add_argument("-f", "--filename", type=str, help="Use a wav/oga/... file instead of the mic data")
parser.add_argument("-t", "--reading_time", type=int, help="Time reading data (mic or file) (only with effect if --show_stats or --show_data is used)")
parser.add_argument("--show_stats", action="store_true", help="shows bandwidth, CPU and quality statistics")
parser.add_argument("--show_samples", action="store_true", help="shows samples values")

import cv2
import numpy as np
import socket
import threading

class VideoStream:
    def __init__(self, capture_device, server_socket):
        # Inicializar la captura de video y el socket del servidor
        self.capture = cv2.VideoCapture(capture_device)
        self.server_socket = server_socket

    def pack_video(self, frame):
        # Codificar el frame en formato JPEG y convertirlo a bytes
        encimg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
        return encimg.tobytes()

    def unpack_video(self, packed_frame):
        # Decodificar los bytes del frame a una imagen
        data = np.frombuffer(packed_frame, dtype=np.uint8)
        decimg = cv2.imdecode(data, 1)
        return decimg

    def send_video(self, send_address):
        # Enviar el video en trozos para evitar el error "Message too long"
        while True:
            ret, frame = self.capture.read()
            packed_frame = self.pack_video(frame)
            chunks = [packed_frame[i:i+65507] for i in range(0, len(packed_frame), 65507)]
            for chunk in chunks:
                try:
                    self.server_socket.sendto(chunk, send_address)
                except Exception as e:
                    print(f"Error sending video frame: {e}")
                    break

    def receive_video(self, receive_address):
        # Recibir y mostrar el video
        while True:
            try:
                packed_frame, addr = self.server_socket.recvfrom(65507)
                frame = self.unpack_video(packed_frame)
                if frame is not None:
                    cv2.imshow('Video connected!', frame)
                    cv2.waitKey(1)
                else:
                    print("Error decoding video frame")
                    break
            except Exception as e:
                print(f"Error receiving video frame: {e}")
                break

    def send_and_receive_video(self, send_address, receive_address):
        # Hilos para enviar y recibir video simultáneamente
        send_thread = threading.Thread(target=self.send_video, args=(send_address,))
        receive_thread = threading.Thread(target=self.receive_video, args=(receive_address,))
        send_thread.start()
        receive_thread.start()

    def close(self):
        # Cerrar el socket del servidor
        self.server_socket.close()

class Minimal:
    NUMBER_OF_CHANNELS = 2

    def __init__(self, args):
        logging.info(__doc__)
        logging.info(f"NUMBER_OF_CHANNELS = {self.NUMBER_OF_CHANNELS}")
        # Configuración del socket para audio
        self.sock_audio = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.listening_endpoint = ("0.0.0.0", args.listening_port)
        self.sock_audio.bind(self.listening_endpoint)
        self.chunk_time = args.frames_per_chunk / args.frames_per_second
        logging.info(f"chunk_time = {self.chunk_time} seconds")
        self.zero_chunk = self.generate_zero_chunk()

        # Configuración del socket para video
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.video_socket.bind(('0.0.0.0', args.listening_port + 1))
        self.video_stream = VideoStream(0, self.video_socket)

        # Determinar el modo de operación (grabar y enviar, o reproducir)
        if args.destination_address:
            self._handler = self._record_and_send  # Usar este handler para evitar escuchar el propio audio
            self.start_video_stream = self._send_and_receive_video  # Cambiar esto para enviar y recibir video
        else:
            self._handler = self._record_and_play
            self.start_video_stream = self._send_and_receive_video  # Cambiar esto para enviar y recibir video

        # Usar un archivo de audio en lugar del micrófono si se especifica
        if args.filename:
            logging.info(f"Using \"{args.filename}\" as input")
            self.wavfile = sf.SoundFile(args.filename, 'r')
            self._handler = self._read_IO_and_play
            self.stream = self.file_stream
        else:
            self._handler = self._record_IO_and_play
            self.stream = self.mic_stream

    def pack_audio(self, audio_chunk):
        # Convertir un chunk de audio a bytes
        return audio_chunk.tobytes()

    def unpack_audio(self, packed_chunk):
        # Convertir bytes a un chunk de audio
        return np.frombuffer(packed_chunk, np.int16)

    def send_audio(self, packed_chunk):
        # Enviar un chunk de audio
        self.sock_audio.sendto(packed_chunk, (args.destination_address, args.destination_port))

    def receive_audio(self):
        # Recibir un chunk de audio
        audio_chunk_size = args.frames_per_chunk * self.NUMBER_OF_CHANNELS * np.dtype(np.int16).itemsize
        audio_packed, _ = self.sock_audio.recvfrom(audio_chunk_size)
        return audio_packed

    def _send_video(self):
        # Iniciar un hilo para enviar video
        send_thread = threading.Thread(target=self.video_stream.send_video, args=((args.destination_address, args.destination_port + 1),))
        send_thread.start()
        send_thread.join()

    def _send_and_receive_video(self):
        # Iniciar hilos para enviar y recibir video simultáneamente
        send_thread = threading.Thread(target=self.video_stream.send_video, args=((args.destination_address, args.destination_port + 1),))
        receive_thread = threading.Thread(target=self.video_stream.receive_video)
        send_thread.start()
        receive_thread.start()
        send_thread.join()
        receive_thread.join()

    def _receive_video(self):
        # Iniciar un hilo para recibir video
        receive_thread = threading.Thread(target=self.video_stream.receive_video)
        receive_thread.start()
        receive_thread.join()

    def generate_zero_chunk(self):
        # Generar un chunk de audio con ceros
        return np.zeros((args.frames_per_chunk, self.NUMBER_OF_CHANNELS), np.int16)

    def _record_IO_and_play(self, ADC, DAC, frames, time, status):
        # Grabar audio del micrófono, enviar y reproducir
        data = ADC.copy()
        packed_chunk = self.pack_audio(data)
        self.send_audio(packed_chunk)

        try:
            packed_chunk = self.receive_audio()
            chunk = self.unpack_audio(packed_chunk)
        except (socket.timeout, BlockingIOError):
            chunk = self.zero_chunk
            logging.debug("playing zero chunk")

        chunk = chunk.reshape((int(len(chunk) / 2), 2))
        DAC[:] = chunk
        if __debug__:
            print(next(spinner), end='\b', flush=True)

    def _read_IO_and_play(self, DAC, frames, time, status):
        # Leer y reproducir audio desde un archivo
        chunk = self.read_chunk_from_file()
        packed_chunk = self.pack(chunk)
        self.send(packed_chunk)
        try:
            packed_chunk = self.receive()
            chunk = self.unpack(packed_chunk)
        except (socket.timeout, BlockingIOError, ValueError):
            chunk = self.zero_chunk
            logging.debug("playing zero chunk")
        DAC[:] = chunk
        if __debug__:
            print(next(spinner), end='\b', flush=True)
        return chunk

    def _record_and_send(self, ADC, DAC, frames, time, status):
        # Grabar audio del micrófono y enviar
        data = ADC.copy()
        packed_chunk = self.pack_audio(data)
        self.send_audio(packed_chunk)
        # DAC[:] = data  # Comentado para evitar escuchar el propio audio

    def _record_send_and_play(self, ADC, DAC, frames, time, status):
        # Grabar, enviar y reproducir audio
        data = ADC.copy()
        packed_chunk = self.pack_audio(data)
        self.send_audio(packed_chunk)
        DAC[:] = data

    def _record_and_play(self, ADC, DAC, frames, time, status):
        # Grabar y reproducir audio sin enviar
        data = ADC.copy()
        DAC[:] = data

    def mic_stream(self):
        # Transmisión de audio desde el micrófono
        try:
            with sd.InputStream(channels=self.NUMBER_OF_CHANNELS, dtype='int16', callback=self.audio_callback):
                with sd.OutputStream(samplerate=args.frames_per_second, blocksize=args.frames_per_chunk, dtype=np.int16, channels=self.NUMBER_OF_CHANNELS) as stream:
                    while True:
                        self.event.wait()
                        self.event.clear()
                        packed_audio = self.pack_audio(self.audio_data)
                        self.queue.put(packed_audio)
                        stream.write(self.audio_data)
        except Exception as e:
            print(f"Error in mic_stream: {e}")

    def audio_callback(self, indata, frames, time, status):
        # Callback para manejar el audio del micrófono
        self.audio_data = indata
        self.event.set()

    def file_stream(self):
        # Transmisión de audio desde un archivo
        with sd.OutputStream(device=args.output_device, samplerate=args.frames_per_second, blocksize=args.frames_per_chunk, dtype=np.int16, channels=self.NUMBER_OF_CHANNELS, callback=self._handler):
            print("Press 'q' to quit...")
            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(0.1)

    def run(self):
        # Ejecución principal del programa
        self.queue = Queue()
        self.event = Event()

        audio_thread = threading.Thread(target=self.mic_stream)
        audio_thread.start()

        if args.destination_address:
            video_thread = threading.Thread(target=self.video_stream.send_and_receive_video, args=((args.destination_address, args.destination_port + 1), ("0.0.0.0", args.listening_port + 1)))
            video_thread.start()
        else:
            video_thread = threading.Thread(target=self.video_stream.send_and_receive_video, args=(("0.0.0.0", args.listening_port + 1), (args.destination_address, args.destination_port + 1)))
            video_thread.start()
            
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Interrupted by user, closing...")
            audio_thread.join()
            video_thread.join()
            self.video_stream.close()
            sys.exit(0)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.list_devices:
        logging.info(sd.query_devices())
        exit()

    if args.destination_address:
        logging.info(f"Waiting for connection to {args.destination_address}:{args.destination_port}")
        logging.info(f"Waiting for connection to {args.destination_address}:{args.destination_port+1}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(5)
        try:
            sock.connect((args.destination_address, args.destination_port))
            sock.connect((args.destination_address, args.destination_port+1))
            logging.info("Connection established!")
            sock.close()
        except socket.error as e:
            logging.error(f"Connection failed: {e}")
        finally:
            sock.close()
        time.sleep(5)  # Wait 5 seconds for the other device to connect

    intercom = Minimal(args)
    intercom.run()

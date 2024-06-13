#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

'''A minimal InterCom for video and audio (no compression, no quantization, no transform, ... only provides a bidirectional (full-duplex) transmission of raw (playable) chunks and video frames).'''

import os
import signal
import argparse
import sys
import sounddevice as sd  # If "pip install sounddevice" fails, install the "libportaudio2" system package
import numpy as np
import socket
import logging
import soundfile as sf
import cv2  # OpenCV for video capture and transmission
import threading
import time
import struct
import traceback
from queue import Queue
from threading import Event

FORMAT = "(%(levelname)s) %(module)s: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)

def spinning_cursor():
    ''' https://stackoverflow.com/questions/4995733/how-to-create-a-spinning-command-line-cursor'''
    while True:
        for cursor in '|/-\\':
            yield cursor
spinner = spinning_cursor()

def int_or_str(text):
    '''Helper function for argument parsing.'''
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
parser.add_argument("-l", "--listening_port", type=int, default=4444, help="My listening port")
parser.add_argument("-a", "--destination_address", type=str, default=None, help="Destination (interlocutor's listening) address")
parser.add_argument("-p", "--destination_port", type=int, default=4444, help="Destination (interlocutor's listening) port")
parser.add_argument("-f", "--filename", type=str, help="Use a wav/oga/... file instead of the mic data")
parser.add_argument("-t", "--reading_time", type=int, help="Time reading data (mic or file) (only with effect if --show_stats or --show_data is used)")

args = parser.parse_args()

class VideoAudioIntercom:
    MAX_PAYLOAD_BYTES = 1500  # Tamaño máximo del paquete UDP de video ajustado (puede necesitar ajuste según la red)
    VIDEO_PORT = 4445  # Default port for video transmission.
    VIDEO_FPS = 10  # Reduced FPS
    NUMBER_OF_CHANNELS = 2  # Number of audio channels

    def __init__(self):
        ''' Constructor. Initializes the sockets and other necessary components. '''
        logging.info(__doc__)
        self.args = args
        self.sock_audio = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.listening_endpoint_audio = ("0.0.0.0", args.listening_port)
        self.listening_endpoint_video = ("0.0.0.0", self.VIDEO_PORT)
        self.sock_audio.bind(self.listening_endpoint_audio)
        self.sock_video.bind(self.listening_endpoint_video)
        self.chunk_time = args.frames_per_chunk / args.frames_per_second
        self.zero_chunk = self.generate_zero_chunk()
        self.destination_address = args.destination_address
        self.destination_port_audio = args.destination_port
        self.destination_port_video = self.VIDEO_PORT
        self.shutdown_flag = threading.Event()

        if args.filename:
            logging.info(f"Using \"{args.filename}\" as input")
            self.wavfile = sf.SoundFile(args.filename, 'r')
            self._audio_handler = self._record_IO_and_play
            self.audio_stream = self.file_stream
        else:
            self._audio_handler = self._record_IO_and_play
            self.audio_stream = self.mic_stream

    def pack_audio(self, audio_chunk):
        # Convertir un chunk de audio a bytes
        return audio_chunk.tobytes()

    def unpack_audio(self, packed_chunk):
        # Convertir bytes a un chunk de audio
        return np.frombuffer(packed_chunk, np.int16).reshape(-1, self.NUMBER_OF_CHANNELS)

    def send_audio(self, packed_chunk):
        # Enviar un chunk de audio
        self.sock_audio.sendto(packed_chunk, (self.destination_address, self.destination_port_audio))

    def receive_audio(self):
        # Recibir un chunk de audio
        audio_chunk_size = args.frames_per_chunk * self.NUMBER_OF_CHANNELS * np.dtype(np.int16).itemsize
        audio_packed, _ = self.sock_audio.recvfrom(audio_chunk_size)
        return audio_packed

    def generate_zero_chunk(self):
        # Generar un chunk de audio con ceros
        return np.zeros((args.frames_per_chunk, self.NUMBER_OF_CHANNELS), np.int16)

    def _record_IO_and_play(self, ADC, DAC, frames, time, status):
        # Grabar audio del micrófono y reproducir
        data = ADC.copy()
        packed_chunk = self.pack_audio(data)

        if self.destination_address:  # Si se ha especificado una dirección IP de destino
            self.send_audio(packed_chunk)  # Enviar audio
            try:
                packed_chunk = self.receive_audio()  # Recibir audio
                chunk = self.unpack_audio(packed_chunk)
            except (socket.timeout, BlockingIOError):
                chunk = self.zero_chunk
                logging.debug("playing zero chunk")
        else:  # Si no se ha especificado una dirección IP de destino
            chunk = data  # Solo usar el audio grabado

        # Ajustar el tamaño del chunk al tamaño del buffer DAC
        if len(chunk) < len(DAC):
            # Rellenar con ceros si el chunk es más pequeño
            chunk = np.pad(chunk, ((0, len(DAC) - len(chunk)), (0, 0)), mode='constant')

        chunk = chunk.reshape((len(chunk) // 2, 2))
        DAC[:] = chunk
        if __debug__:
            print(next(spinner), end='\b', flush=True)

    def _record_and_send(self, ADC, frames, time, status):
        # Grabar audio del micrófono y enviarlo
        data = ADC.copy()
        packed_chunk = self.pack_audio(data)
        self.send_audio(packed_chunk)

    def _play_received_audio(self, DAC, frames, time, status):
        try:
            packed_chunk = self.receive_audio()  # Recibir audio
            chunk = self.unpack_audio(packed_chunk)
        except (socket.timeout, BlockingIOError):
            chunk = self.zero_chunk
            logging.debug("playing zero chunk")

        # Asegurarse de que el chunk tenga el número correcto de frames y canales
        if chunk.ndim == 1:
            chunk = np.tile(chunk[:, np.newaxis], (1, 2))  # Duplicar los datos de un solo canal para crear dos canales

        if len(chunk) < len(DAC):
            # Rellenar el chunk con ceros si es demasiado pequeño
            chunk = np.pad(chunk, ((0, len(DAC) - len(chunk)), (0, 0)), mode='constant')
        elif len(chunk) > len(DAC):
            # Truncar el chunk si es demasiado grande
            chunk = chunk[:len(DAC)]

        DAC[:] = chunk

        if __debug__:
            print(next(spinner), end='\b', flush=True)

    def mic_stream(self):
        # Transmisión de audio desde el micrófono
        try:
            if self.destination_address:
                with sd.InputStream(channels=self.NUMBER_OF_CHANNELS, dtype='int16', callback=self._record_and_send) as input_stream:
                    with sd.OutputStream(samplerate=args.frames_per_second, blocksize=args.frames_per_chunk, dtype=np.int16, channels=self.NUMBER_OF_CHANNELS, callback=self._play_received_audio) as output_stream:
                        while not self.shutdown_flag.is_set():
                            time.sleep(0.1)
            else:
                with sd.InputStream(channels=self.NUMBER_OF_CHANNELS, dtype='int16', callback=self.audio_callback) as input_stream:
                    with sd.OutputStream(samplerate=args.frames_per_second, blocksize=args.frames_per_chunk, dtype=np.int16, channels=self.NUMBER_OF_CHANNELS) as output_stream:
                        while not self.shutdown_flag.is_set():
                            self.event.wait()
                            self.event.clear()
                            packed_audio = self.pack_audio(self.audio_data)
                            self.queue.put(packed_audio)
                            output_stream.write(self.audio_data)
        except Exception as e:
            logging.error(f"Error in mic_stream: {e}")

    def audio_callback(self, indata, frames, time, status):
        self.audio_data = indata
        self.event.set()

    def file_stream(self):
        # Transmisión de audio desde un archivo
        with sd.OutputStream(device=args.output_device, samplerate=args.frames_per_second, blocksize=args.frames_per_chunk, dtype=np.int16, channels=self.NUMBER_OF_CHANNELS, callback=self._handler):
            print("Press 'q' to quit...")
            while not self.shutdown_flag.is_set():
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(0.1)

    def send_video(self, frame):
        '''Sends a video frame.'''
        if self.destination_address:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
            frame_data = frame.tobytes()  # Convert frame to bytes
            header = struct.pack('>I', len(frame_data))
            frame_data = header + frame_data
            num_chunks = len(frame_data) // self.MAX_PAYLOAD_BYTES + 1
            # Send the number of chunks first
            self.sock_video.sendto(struct.pack('>I', num_chunks), (self.destination_address, self.destination_port_video))
            for i in range(num_chunks):
                chunk = frame_data[i * self.MAX_PAYLOAD_BYTES: (i + 1) * self.MAX_PAYLOAD_BYTES]
                try:
                    self.sock_video.sendto(chunk, (self.destination_address, self.destination_port_video))
                except Exception as e:
                    logging.error(f"Failed to send video chunk: {e}")

    def receive_video(self):
        '''Receives a video frame.'''
        frame_chunks = {}
        expected_chunk = 0
        frame_size = 0
        # Receive the number of chunks first
        num_chunks, _ = self.sock_video.recvfrom(4)
        num_chunks = struct.unpack('>I', num_chunks)[0]
        while True:
            try:
                chunk, _ = self.sock_video.recvfrom(self.MAX_PAYLOAD_BYTES)
                frame_chunks[expected_chunk] = chunk
                if expected_chunk == 0:  # This is the first chunk
                    frame_size = struct.unpack('>I', chunk[:4])[0] + 4  # Update frame size
                expected_chunk += 1
                if len(frame_chunks) >= num_chunks:
                    break
            except socket.timeout as e:
                logging.debug(f"Video receive timeout: {e}")
                break
            except Exception as e:
                logging.error(f"Failed to receive video chunk: {e}")
                logging.error(traceback.format_exc())
                break
            
        frame_data = b''
        for i in range(expected_chunk):
            if i in frame_chunks:
                frame_data += frame_chunks[i]
            else:
                frame_data += b'\x00' * self.MAX_PAYLOAD_BYTES  # Fill missing chunks with zeros

        if len(frame_data) > 4:
            frame_size = struct.unpack('>I', frame_data[:4])[0]
            frame_data = frame_data[4:]
            if len(frame_data) == frame_size:
                try:
                    frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(240, 320, 3)
                    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    logging.error(f"Failed to process received frame: {e}")
                    return None
        return None

    def run_video(self):
        '''Runs the video stream.'''
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Could not open video device")
            return

        if args.destination_address:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Ajustar la resolución a 320x240
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            cap.set(cv2.CAP_PROP_FPS, self.VIDEO_FPS)  # Usar el FPS especificado
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:
            while not self.shutdown_flag.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                if args.destination_address:
                    self.send_video(frame)
                    received_frame = self.receive_video()
                    if received_frame is not None:
                        cv2.imshow('Received Video', received_frame)
                else:
                    cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            logging.error(f"Error in run_video: {e}")

        cap.release()
        cv2.destroyAllWindows()
        self.sock_video.close()

    def cleanup(self):
        '''Cleans up the resources.'''
        logging.info("Cleaning up resources...")
        self.shutdown_flag.set()
        try:
            self.sock_audio.close()
            self.sock_video.close()
        except Exception as e:
            logging.error(f"Failed to close sockets: {e}")

    def run(self):
        self.queue = Queue()
        self.event = Event()

        audio_thread = threading.Thread(target=self.mic_stream)
        video_thread = threading.Thread(target=self.run_video, daemon=True)
        audio_thread.start()
        video_thread.start()
        
        try:
            audio_thread.join()
            video_thread.join()
        except KeyboardInterrupt:
            logging.info("Interrupt received, stopping...")
            self.cleanup()

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
        time.sleep(5)

    def signal_handler(sig, frame):
        logging.info("Signal received, shutting down...")
        intercom.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        intercom = VideoAudioIntercom()
        intercom.run()
    except KeyboardInterrupt:
        logging.info("Closing program...")
        intercom.cleanup()
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        intercom.cleanup()
        sys.exit(1)

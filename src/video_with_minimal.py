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


# Configuración de logging
FORMAT = "(%(levelname)s) %(module)s: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)


def int_or_str(text):
   try:
       return int(text)
   except ValueError:
       return text


# Configuración de argumentos de línea de comandos
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input-device", type=int_or_str, help="Input device ID or substring")
parser.add_argument("-o", "--output-device", type=int_or_str, help="Output device ID or substring")
parser.add_argument("-d", "--list-devices", action="store_true", help="Print the available audio devices and quit")
parser.add_argument("-s", "--frames_per_second", type=float, default=22050, help="Sampling rate in frames/second")
parser.add_argument("-c", "--frames_per_chunk", type=int, default=1024, help="Number of frames in a chunk")
parser.add_argument("-l", "--listening_port", type=int, default=4444, help="My listening port for audio")
parser.add_argument("-a", "--destination_address", type=str, default=None, help="Destination (interlocutor's listening) address")
parser.add_argument("-p", "--destination_port", type=int, default=4444, help="Destination (interlocutor's listening) port for audio")
parser.add_argument("-f", "--filename", type=str, help="Use a wav/oga/... file instead of the mic data")
args = parser.parse_args()


# Configuración de video
VIDEO_PORT = 4445
VIDEO_FPS = 10
NUMBER_OF_CHANNELS = 2
VIDEO_FRAME_SIZE = 320 * 240 * 3  # Tamaño de un frame de video (en bytes) para 320x240 resolución
MAX_PAYLOAD_BYTES = 60000 # Tamaño máximo de un paquete UDP


class VideoAudioIntercom:
   def __init__(self, args):
       self.args = args
       self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
       self.listening_endpoint = ("0.0.0.0", self.args.listening_port)
       self.sock.bind(self.listening_endpoint)
       self.sock.settimeout(1.0)  # Establece un tiempo de espera de 1 segundo para recvfrom
       self.chunk_time = self.args.frames_per_chunk / self.args.frames_per_second
       self.destination_address = self.args.destination_address
       self.destination_port = self.args.destination_port
       self.shutdown_flag = threading.Event()  # Usamos un threading.Event para gestionar la señal de cierre
       self.zero_chunk_audio = np.zeros((self.args.frames_per_chunk, NUMBER_OF_CHANNELS), np.int16)
       self.zero_chunk_video = np.zeros((240, 320, 3), dtype=np.uint8)  # Frame en negro para 320x240 resolución
       self.cap = None
       logging.info(f"chunk_time = {self.chunk_time} seconds")
      
       if self.args.filename:
           logging.info(f"Using \"{self.args.filename}\" as input")
           self.wavfile = sf.SoundFile(self.args.filename, 'r')
           self.audio_stream = self.file_stream
       else:
           self.audio_stream = self.mic_stream


   # Empaqueta el chunk de audio y video juntos
   def pack_audio_video(self, audio_chunk, video_frame):
       packed_audio = audio_chunk.tobytes()
       packed_video = video_frame.tobytes()
       return struct.pack('<L', len(packed_audio)) + packed_audio + packed_video


   # Desempaqueta los datos del audio y video
   def unpack_audio_video(self, packed_chunk):
       audio_size = struct.unpack('<L', packed_chunk[:4])[0]
       audio_chunk = np.frombuffer(packed_chunk[4:4 + audio_size], np.int16).reshape(-1, NUMBER_OF_CHANNELS)
       video_chunk = np.frombuffer(packed_chunk[4 + audio_size:], dtype=np.uint8).reshape(240, 320, 3)
       return audio_chunk, video_chunk


   # Envía el chunk empaquetado al destino
   def send_audio_video(self, packed_chunk):
       if packed_chunk is not None:
           try:
               num_fragments = (len(packed_chunk) + MAX_PAYLOAD_BYTES - 1) // MAX_PAYLOAD_BYTES
               for i in range(num_fragments):
                   fragment = packed_chunk[i * MAX_PAYLOAD_BYTES:(i + 1) * MAX_PAYLOAD_BYTES]
                   header = struct.pack('<H', i) + struct.pack('<H', num_fragments)
                   self.sock.sendto(header + fragment, (self.destination_address, self.destination_port))
                   time.sleep(0.01)  # Retardo pequeño para evitar la saturación de la red
           except Exception as e:
               logging.error(f"Failed to send audio/video: {e}")


   # Recibe un chunk empaquetado del destino
   def receive_audio_video(self):
       try:
           fragments = {}
           expected_fragments = None
           while True:
               try:
                   packed_data, _ = self.sock.recvfrom(MAX_PAYLOAD_BYTES + 4)  # Recibe el fragmento con el encabezado
               except socket.timeout:
                   if self.shutdown_flag.is_set():
                       return None
                   else:
                       continue  # Seguir esperando


               fragment_index, total_fragments = struct.unpack('<H', packed_data[:2])[0], struct.unpack('<H', packed_data[2:4])[0]
               fragments[fragment_index] = packed_data[4:]
               if expected_fragments is None:
                   expected_fragments = total_fragments
               if len(fragments) == expected_fragments:
                   break
           # Reconstruir el paquete completo
           full_packet = b''.join(fragments[i] for i in range(expected_fragments))
           return full_packet
       except Exception as e:
           logging.error(f"Failed to receive audio/video: {e}")
           return None


   # Callback para grabar audio, video, enviarlos al destino, recibir audio y video del destino y reproducirlos
   def _record_IO_and_play(self, ADC, DAC, frames, time_info, status):
       if self.shutdown_flag.is_set():
           raise sd.CallbackAbort

       try:
           # Captura el audio del micrófono
           audio_chunk = ADC.copy()

           # Captura el video desde la cámara
           ret, frame = self.cap.read()
           if not ret:
               frame = self.zero_chunk_video  # Si falla la captura de video, usa un frame en negro

           # Empaqueta audio y video juntos
           packed_chunk = self.pack_audio_video(audio_chunk, frame)

           if self.destination_address:
               # Enviar el paquete audio-video
               self.send_audio_video(packed_chunk)

               # Recibir el paquete de audio y video del interlocutor
               packed_chunk = self.receive_audio_video()

               if packed_chunk:
                   audio_chunk, video_frame = self.unpack_audio_video(packed_chunk)
               else:
                   audio_chunk = self.zero_chunk_audio  # Relleno en caso de fallo
                   video_frame = self.zero_chunk_video  # Frame negro en caso de fallo
           else:
               audio_chunk = ADC  # Si no hay interlocutor, usa el mismo audio
               video_frame = frame  # Y el mismo video

           # Reproducir el audio recibido
           DAC[:] = audio_chunk

           # Mostrar el video recibido
           cv2.imshow('Video', video_frame)
           if cv2.waitKey(1) & 0xFF == ord('q'):
               self.shutdown_flag.set()

       except sd.CallbackAbort:
           logging.info("Callback aborted")
       except Exception as e:
           logging.error(f"Error in audio/video processing: {e}")


   # Configura la cámara para el video
   def setup_video_capture(self):
       self.cap = cv2.VideoCapture(0)
       self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Cambiar la resolución a 320x240
       self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
       self.cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
       if not self.cap.isOpened():
           logging.error("Failed to open video capture")


   # Función para ejecutar el stream de audio desde el micrófono
   def mic_stream(self, callback_function):
       try:
           self.setup_video_capture()  # Inicia la captura de video
           with sd.Stream(device=(self.args.input_device, self.args.output_device),
                          dtype=np.int16,
                          samplerate=self.args.frames_per_second,
                          blocksize=self.args.frames_per_chunk,
                          channels=NUMBER_OF_CHANNELS,
                          callback=callback_function):
               while not self.shutdown_flag.is_set():
                   time.sleep(0.1)  # Mantiene el stream activo
       except Exception as e:
           logging.error(f"Error in mic_stream: {e}")


   # Función principal de ejecución
   def run(self):
       try:
           self.audio_stream(self._record_IO_and_play)
       except KeyboardInterrupt:
           logging.info("Interruption by user")
       except Exception as e:
           logging.error(f"Exception in run: {e}")
       finally:
           self.shutdown()


   # Cierre ordenado del stream
   def shutdown(self):
       if not self.shutdown_flag.is_set():
           logging.info("Shutting down...")
           self.shutdown_flag.set()
           try:
               self.sock.close()
           except Exception as e:
               logging.error(f"Error closing socket: {e}")
           try:
               if self.cap:
                   self.cap.release()
           except Exception as e:
               logging.error(f"Error releasing video capture: {e}")
           try:
               cv2.destroyAllWindows()
           except Exception as e:
               logging.error(f"Error destroying OpenCV windows: {e}")
           logging.info("Shutdown complete.")


# Función para manejar señales como SIGINT
def signal_handler(sig, frame):
   logging.info("SIGINT received, shutting down...")
   intercom.shutdown()


if __name__ == "__main__":
   intercom = VideoAudioIntercom(args)

   # Registrar el handler de la señal SIGINT (Ctrl+C)
   signal.signal(signal.SIGINT, signal_handler)

   try:
       intercom.run()
   except KeyboardInterrupt:
       logging.info("KeyboardInterrupt caught in main")
       intercom.shutdown()
       parser.exit("\nSIGINT received")


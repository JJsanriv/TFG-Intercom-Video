import signal
import argparse
import sys
import sounddevice as sd
import numpy as np
import socket
import logging
import cv2
import struct
import threading
import collections
import queue
import time

# Configuración básica del logger
FORMAT = "(%(levelname)s) %(module)s: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)

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
parser.add_argument("-s", "--frames_per_second", type=float, default=44100, help="Sampling rate in frames/second")
parser.add_argument("-c", "--frames_per_chunk", type=int, default=1024, help="Number of frames in a chunk")
parser.add_argument("-l", "--listening_port", type=int, default=4444, help="My listening port for audio/video")
parser.add_argument("-a", "--destination_address", type=str, default=None, help="Destination (interlocutor's listening) address")
parser.add_argument("-p", "--destination_port", type=int, default=4444, help="Destination (interlocutor's listening) port for audio/video")
parser.add_argument("-f", "--filename", type=str, help="Use a wav/oga/... file instead of the mic data")
parser.add_argument("-j", "--jitter_buffer_size", type=int, default=5, help="Size of jitter buffer (in packets)")
args = parser.parse_args()

# Constantes de configuración para video y audio
VIDEO_FPS = 10
NUMBER_OF_CHANNELS = 2
VIDEO_WIDTH = 320
VIDEO_HEIGHT = 240
MAX_PAYLOAD_BYTES = 32768
AUDIO_SAMPLE_SIZE = 2  # 16 bits = 2 bytes por muestra

class VideoAudioIntercom:
  def __init__(self, args):
      self.args = args
      self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
      # Enlazar al puerto de escucha
      self.sock.bind(("0.0.0.0", self.args.listening_port))
      self.sock.settimeout(0.01)
      self.destination_address = self.args.destination_address
      self.destination_port = self.args.destination_port
      self.shutdown_flag = threading.Event()

      # Chunks de audio y video nulos para cuando no hay datos válidos
      self.zero_chunk_audio = np.zeros((self.args.frames_per_chunk, NUMBER_OF_CHANNELS), np.int16)
      self.zero_chunk_video = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
      self.zero_audio_bytes = self.zero_chunk_audio.tobytes()
      self.zero_video_bytes = self.zero_chunk_video.tobytes()

      self.cap = None
      self.jitter_buffer = collections.deque(maxlen=self.args.jitter_buffer_size)
      self.packet_queue = queue.Queue(maxsize=5)
      self.fragments_received = {}
      self.latest_packet = None
      self.packet_counter = 0
      self.current_video_frame = self.zero_chunk_video.copy()
      self.last_video_display_time = 0
      self.next_video_capture_time = 0

      self.new_frame_ready = threading.Event()

      self.sender_fragmenter_thread = None
      self.receiver_thread = None
      self.display_thread = None
      self.audio_stream_instance = None

      # Selección entre entrada de micrófono o archivo
      if self.args.filename:
          logging.error("File playback not implemented with video")
          self.audio_stream = self.file_stream
      else:
          self.audio_stream = self.mic_stream

      # Formatos de cabecera para paquetes y fragmentos
      self.PACKET_HEADER_FORMAT = '<LQB'  # packet_number, timestamp, has_video
      self.PACKET_HEADER_SIZE = struct.calcsize(self.PACKET_HEADER_FORMAT)
      self.FRAGMENT_HEADER_FORMAT = '<III'  # fragment_id, total_fragments, fragment_index
      self.FRAGMENT_HEADER_SIZE = struct.calcsize(self.FRAGMENT_HEADER_FORMAT)

  def pack_audio_video(self, audio_chunk, video_frame):
    try:
        # Validación de audio
        if audio_chunk is None or audio_chunk.size == 0 or not np.isfinite(audio_chunk).all() or \
           audio_chunk.shape != (self.args.frames_per_chunk, NUMBER_OF_CHANNELS):
            if audio_chunk is not None:
                 logging.warning(f"Invalid audio chunk shape/content: {audio_chunk.shape if audio_chunk is not None else 'None'}. Using silence.")
            packed_audio = self.zero_audio_bytes
        else:
            packed_audio = audio_chunk.tobytes()

        audio_size = len(packed_audio)

        # Validación de video
        if video_frame is None or video_frame.size == 0 or not np.isfinite(video_frame).all() or \
           video_frame.shape != (VIDEO_HEIGHT, VIDEO_WIDTH, 3):
            if video_frame is not None:
                logging.warning(f"Invalid video frame shape/content: {video_frame.shape if video_frame is not None else 'None'}. Omitting video.")
            # Paquete sin video (has_video=0)
            return struct.pack(self.PACKET_HEADER_FORMAT, audio_size, 0, 0) + packed_audio
        else:
            packed_video = video_frame.tobytes()
            # Paquete con audio y video (has_video=1)
            return struct.pack(self.PACKET_HEADER_FORMAT, audio_size, 0, 1) + packed_audio + packed_video

    except Exception as e:
        logging.error(f"Error in pack_audio_video: {e}")
        # Fallback a paquete solo de silencio en caso de error
        return struct.pack(self.PACKET_HEADER_FORMAT, len(self.zero_audio_bytes), 0, 0) + self.zero_audio_bytes


  def unpack_audio_video(self, packed_chunk):
    try:
        if packed_chunk is None or len(packed_chunk) < self.PACKET_HEADER_SIZE:
            logging.warning(f"Received packet too short ({len(packed_chunk) if packed_chunk is not None else 'None'}) for header. Returning silence/blank.")
            return self.zero_chunk_audio, None

        audio_size, packet_number, has_video = struct.unpack(self.PACKET_HEADER_FORMAT, packed_chunk[:self.PACKET_HEADER_SIZE])

        if not (0 < audio_size <= (MAX_PAYLOAD_BYTES - self.PACKET_HEADER_SIZE)):
            logging.warning(f"Invalid audio size in packet {packet_number}: {audio_size}. Returning silence.")
            return self.zero_chunk_audio, None

        audio_end_index = self.PACKET_HEADER_SIZE + audio_size
        if audio_end_index > len(packed_chunk):
            logging.warning(f"Reported audio size ({audio_size}) for packet {packet_number} exceeds packet length ({len(packed_chunk)}). Returning silence.")
            return self.zero_chunk_audio, None

        audio_data = packed_chunk[self.PACKET_HEADER_SIZE:audio_end_index]
        audio_chunk = self.zero_chunk_audio  # Default a silencio
        # Cálculo de bytes esperados para el audio
        expected_audio_bytes = self.args.frames_per_chunk * NUMBER_OF_CHANNELS * AUDIO_SAMPLE_SIZE
        if len(audio_data) != expected_audio_bytes:
            logging.warning(f"Audio data size mismatch (got {len(audio_data)}, expected {expected_audio_bytes}) for packet {packet_number}. Using silence.")
        else:
            try:
                audio_chunk_candidate = np.frombuffer(audio_data, np.int16).reshape(-1, NUMBER_OF_CHANNELS)
                # Verificar forma tras el reshape
                if audio_chunk_candidate.shape == (self.args.frames_per_chunk, NUMBER_OF_CHANNELS):
                    audio_chunk = audio_chunk_candidate
                else:
                     logging.warning(f"Reshaped audio data has wrong shape {audio_chunk_candidate.shape} for packet {packet_number}. Using silence.")
            except ValueError as e:
                logging.warning(f"Could not reshape audio data (size {len(audio_data)}) for packet {packet_number}: {e}. Using silence.")

        # Si no hay video, devuelve solo el audio
        if has_video == 0:
            return audio_chunk, None

        # Procesa datos de video si están presentes
        video_data = packed_chunk[audio_end_index:]
        expected_video_bytes = VIDEO_HEIGHT * VIDEO_WIDTH * 3
        if len(video_data) != expected_video_bytes:
            logging.warning(f"Received video data size mismatch (got {len(video_data)}, expected {expected_video_bytes}) for packet {packet_number}. Returning audio with blank frame.")
            return audio_chunk, self.zero_chunk_video.copy()

        try:
            video_frame = np.frombuffer(video_data, dtype=np.uint8).reshape(VIDEO_HEIGHT, VIDEO_WIDTH, 3)
            return audio_chunk, video_frame
        except ValueError:
            logging.warning(f"Received malformed video data (packet {packet_number}, size {len(video_data)}). Returning blank frame.")
            return audio_chunk, self.zero_chunk_video.copy()

    except struct.error as e:
         logging.warning(f"Error unpacking packet header: {e}. Returning silence/blank.")
         return self.zero_chunk_audio, None
    except Exception as e:
        logging.error(f"Error in unpack_audio_video: {e}")
        return self.zero_chunk_audio, None


  def reassemble_packet(self, fragment, addr):
    try:
        # Validación básica del fragmento
        if fragment is None or len(fragment) < self.FRAGMENT_HEADER_SIZE:
            logging.warning(f"Received fragment too short ({len(fragment) if fragment is not None else 'None'} bytes) from {addr}. Discarding.")
            return None

        fragment_id, total_fragments, fragment_index = struct.unpack(self.FRAGMENT_HEADER_FORMAT, fragment[:self.FRAGMENT_HEADER_SIZE])

        # Validación de límites de fragmentación
        if not (0 < total_fragments <= 1000):
            logging.warning(f"Invalid total_fragments value ({total_fragments}) for ID {fragment_id} from {addr}. Discarding.")
            return None
        if not (0 <= fragment_index < total_fragments):
            logging.warning(f"Invalid fragment_index ({fragment_index}) for ID {fragment_id}, total {total_fragments} from {addr}. Discarding.")
            return None

        fragment_data = fragment[self.FRAGMENT_HEADER_SIZE:]
        if not fragment_data:
            logging.warning(f"Empty fragment data for ID {fragment_id}, index {fragment_index} from {addr}. Discarding.")
            return None

        # Gestión de fragmentos recibidos
        current_time = time.time()
        if fragment_id not in self.fragments_received:
             # Limpieza si hay demasiados conjuntos incompletos
            if len(self.fragments_received) > 50:
                oldest_id = min(self.fragments_received, key=lambda k: self.fragments_received[k]['time'])
                del self.fragments_received[oldest_id]
                logging.debug(f"Cleaned up oldest incomplete fragment set {oldest_id} due to memory limit.")

            self.fragments_received[fragment_id] = {'total': total_fragments, 'received': 0, 'fragments': {}, 'time': current_time}

        # Evita procesar fragmentos duplicados
        if fragment_index not in self.fragments_received[fragment_id]['fragments']:
            self.fragments_received[fragment_id]['fragments'][fragment_index] = fragment_data
            self.fragments_received[fragment_id]['received'] += 1

        # Intenta reconstruir el paquete si están todos los fragmentos
        fragment_info = self.fragments_received[fragment_id]
        if fragment_info['received'] == fragment_info['total']:
            try:
                ordered_fragments = [fragment_info['fragments'][i] for i in range(fragment_info['total'])]
                complete_packet = b"".join(ordered_fragments)
            except KeyError:
                 logging.warning(f"Missing fragment during final assembly for packet ID {fragment_id}. Discarding.")
                 del self.fragments_received[fragment_id]
                 return None

            # Validación del paquete reconstruido
            if not complete_packet:
                logging.warning(f"Reassembled empty packet from fragments for ID {fragment_id}. Discarding.")
                del self.fragments_received[fragment_id]
                return None
            if len(complete_packet) < self.PACKET_HEADER_SIZE:
                logging.warning(f"Reassembled packet too small (size {len(complete_packet)}) for ID {fragment_id}. Discarding.")
                del self.fragments_received[fragment_id]
                return None

            del self.fragments_received[fragment_id]  # Limpieza tras éxito
            return complete_packet

    except struct.error as e:
         logging.warning(f"Received malformed fragment header from {addr}: {e}. Discarding.")
    except KeyError:
         logging.debug(f"Received fragment for already processed/cleaned ID {fragment_id} from {addr}. Discarding.")
    except Exception as e:
        logging.error(f"Error reassembling packet: {e}")

    return None


  def clean_old_fragments(self, max_time=0.8):
      """Elimina conjuntos de fragmentos no completados después de max_age segundos."""
      current_time = time.time()
      fragments_to_delete = [
          fid for fid, data in self.fragments_received.items()
          if current_time - data.get('time', current_time) > max_time
      ]
      for fragment_id in fragments_to_delete:
          if self.fragments_received.pop(fragment_id, None):
               logging.debug(f"Cleaned up old/incomplete fragment set {fragment_id}")


  def sender_fragmenter_loop(self):
      """Bucle que toma paquetes de la cola y los envía fragmentados."""
      fragment_id_counter = 0
      try:
          # Verificar destino válido
          dest_addr = (self.destination_address, self.destination_port) if self.destination_address else None
          if not dest_addr:
              logging.warning("SenderFragmenter Thread started without destination address. Exiting.")
              return

          while not self.shutdown_flag.is_set():
              try:
                  packet_data, packet_number = self.packet_queue.get(timeout=0.1)

                  if packet_data is None:  # Ignorar paquetes nulos
                      self.packet_queue.task_done()
                      continue

                  # Actualizar cabecera con número de paquete
                  try:
                      audio_size, _, has_video = struct.unpack(self.PACKET_HEADER_FORMAT, packet_data[:self.PACKET_HEADER_SIZE])
                      packet_header = struct.pack(self.PACKET_HEADER_FORMAT, audio_size, packet_number, has_video)
                      packet_payload = packet_data[self.PACKET_HEADER_SIZE:]
                      full_packet_to_send = packet_header + packet_payload
                  except struct.error:
                      logging.error(f"Could not unpack/repack packet header for packet #{packet_number}. Skipping send.")
                      self.packet_queue.task_done()
                      continue

                  if not full_packet_to_send:
                       logging.warning(f"Packet #{packet_number} resulted in empty data before fragmentation. Skipping send.")
                       self.packet_queue.task_done()
                       continue

                  # Cálculo de fragmentación
                  header_size = self.FRAGMENT_HEADER_SIZE
                  payload_size = MAX_PAYLOAD_BYTES - header_size
                  total_size = len(full_packet_to_send)
                  total_fragments = max(1, (total_size + payload_size - 1) // payload_size)
                  fragment_id = fragment_id_counter
                  fragment_id_counter += 1

                  # Envío de fragmentos
                  for i in range(total_fragments):
                      start = i * payload_size
                      end = min(start + payload_size, total_size)
                      if start >= end:
                          logging.warning(f"Fragment calculation invalid slice ({start}:{end}) for packet {packet_number}, frag {i}. Stopping.")
                          break

                      fragment_payload = full_packet_to_send[start:end]
                      fragment_header = struct.pack(self.FRAGMENT_HEADER_FORMAT, fragment_id, total_fragments, i)
                      fragment_to_send = fragment_header + fragment_payload

                      try:
                          self.sock.sendto(fragment_to_send, dest_addr)
                      except socket.error as e:
                          if not self.shutdown_flag.is_set():
                              logging.error(f"Socket error sending fragment {i} (packet {packet_number}): {e}")
                          break
                      except Exception as e:
                           if not self.shutdown_flag.is_set():
                               logging.error(f"Error sending fragment {i} (packet {packet_number}): {e}")
                           break

                  self.packet_queue.task_done()

              except queue.Empty:
                  continue  # Timeout normal de la cola
              except Exception as e:
                  if not self.shutdown_flag.is_set():
                      logging.error(f"Error in sender_fragmenter_loop's main try: {e}")

      except Exception as e:
          if not self.shutdown_flag.is_set():
              logging.error(f"Unhandled error in sender_fragmenter_loop setup: {e}")
      finally:
          logging.info("Sender/Fragmenter loop finished.")


  def receiver_loop(self):
      """Bucle que recibe fragmentos, los reensambla y llena el jitter buffer."""
      last_cleanup_time = time.time()
      CLEANUP_INTERVAL = 1.0
      try:
          while not self.shutdown_flag.is_set():
              try:
                  # Recibir fragmento
                  fragment, addr = self.sock.recvfrom(MAX_PAYLOAD_BYTES + self.FRAGMENT_HEADER_SIZE + 100)
                  packet = self.reassemble_packet(fragment, addr)
                  if packet:
                      # Añadir paquete completo al buffer
                      self.jitter_buffer.append(packet)
                      # Proporcionar al callback el próximo paquete si no hay ninguno en proceso
                      if self.latest_packet is None and self.jitter_buffer:
                          try:
                            self.latest_packet = self.jitter_buffer.popleft()
                          except IndexError:
                            pass  # El buffer podría vaciarse entre la comprobación y el pop

                  # Limpieza periódica de fragmentos antiguos
                  current_time = time.time()
                  if current_time - last_cleanup_time > CLEANUP_INTERVAL:
                      self.clean_old_fragments()
                      last_cleanup_time = current_time

              except socket.timeout:
                  continue  # Timeout normal
              except socket.error as e:
                   if self.shutdown_flag.is_set(): break
                   logging.error(f"Socket error in receiver_loop: {e}")
                   time.sleep(0.1)  # Pausa breve en caso de error
              except Exception as e:
                  if not self.shutdown_flag.is_set():
                      logging.error(f"Error in receiver_loop: {e}", exc_info=False)
      except Exception as e:
          if not self.shutdown_flag.is_set():
              logging.error(f"Unhandled error in receiver_loop: {e}")
      finally:
          logging.info("Receiver loop finished.")


  def send_audio_video(self, packed_chunk_no_pkt_num):
      """Encola un paquete de audio/video para envío fragmentado."""
      if self.destination_address and packed_chunk_no_pkt_num is not None:
          try:
              current_packet_number = self.packet_counter
              self.packet_counter += 1
              packet_to_queue = (packed_chunk_no_pkt_num, current_packet_number)

              # Intenta encolar sin bloqueo
              self.packet_queue.put_nowait(packet_to_queue)

          except queue.Full:
              # Si la cola está llena, descarta el paquete más antiguo y reintenta
              try:
                  dropped_packet, dropped_num = self.packet_queue.get_nowait()
                  self.packet_queue.task_done()
                  logging.debug(f"Packet queue full, dropped oldest packet #{dropped_num}. Retrying put.")
                  self.packet_queue.put_nowait(packet_to_queue)
              except queue.Empty:
                  logging.warning(f"Packet queue race condition? Packet #{current_packet_number} dropped.")
              except queue.Full:
                   logging.warning(f"Packet queue still full after dropping. Packet #{current_packet_number} dropped.")
              except Exception as e_inner:
                   logging.error(f"Error handling full queue for packet #{current_packet_number}: {e_inner}")
          except Exception as e:
              logging.error(f"Error adding packet #{current_packet_number} to queue: {e}")


  def receive_audio_video(self):
      """Obtiene el siguiente paquete disponible para procesamiento."""
      packet = self.latest_packet
      self.latest_packet = None  # Consumir el 'latest'

      # Si no hay paquete pendiente, intentar sacar del buffer
      if packet is None and self.jitter_buffer:
          try:
              packet = self.jitter_buffer.popleft()
          except IndexError:
              packet = None

      return packet


  def _record_IO_and_play(self, indata, outdata, frames, time_info, status):
    """Callback principal de audio que maneja captura, envío, recepción y reproducción."""
    if self.shutdown_flag.is_set():
        outdata[:] = self.zero_chunk_audio
        raise sd.CallbackAbort

    if status: logging.warning(f"Audio callback status: {status}")

    try:
        # 1. Validar audio de entrada para enviar/reproducir
        expected_shape = (self.args.frames_per_chunk, NUMBER_OF_CHANNELS)
        if indata is None or not np.isfinite(indata).all() or indata.shape != expected_shape:
            if indata is not None:
                 logging.warning(f"Invalid audio input shape/content: {indata.shape if indata is not None else 'None'}. Using silence.")
            audio_chunk_to_send = self.zero_chunk_audio.copy()
        else:
            audio_chunk_to_send = indata

        # 2. Capturar video si es el momento adecuado
        current_frame_to_send = None
        video_capture = self.cap and self.cap.isOpened()
        current_time = time.time()

        if video_capture and current_time >= self.next_video_capture_time:
            ret, frame = self.cap.read()
            self.next_video_capture_time = current_time + (1.0 / VIDEO_FPS)

            if ret and frame is not None and frame.shape == (VIDEO_HEIGHT, VIDEO_WIDTH, 3):
                current_frame_to_send = frame
            else:
                logging.warning(f"Invalid video frame captured (ret={ret}, shape={frame.shape if frame is not None else 'None'}). Using blank.")
                current_frame_to_send = self.zero_chunk_video.copy()

            # En modo local, actualizar frame inmediatamente para display
            if not self.destination_address and current_frame_to_send is not None:
                self.current_video_frame = current_frame_to_send
                self.new_frame_ready.set()

        # 3. Modo red (con destino remoto)
        if self.destination_address:
            # Empaquetar y enviar
            packed_chunk = self.pack_audio_video(audio_chunk_to_send, current_frame_to_send)
            self.send_audio_video(packed_chunk)

            # Recibir y procesar
            received_packet = self.receive_audio_video()
            audio_chunk_to_play = self.zero_chunk_audio  # Default a silencio
            received_video_frame = None

            if received_packet:
                unpacked_audio, unpacked_video = self.unpack_audio_video(received_packet)
                if unpacked_audio is not None and unpacked_audio.shape == expected_shape:
                    audio_chunk_to_play = unpacked_audio
                if unpacked_video is not None:
                    received_video_frame = unpacked_video

            # Reproducir audio recibido
            outdata[:] = audio_chunk_to_play

            # Actualizar frame de video si se recibió
            if received_video_frame is not None:
                 self.current_video_frame = received_video_frame
                 self.new_frame_ready.set()

        # 4. Modo local (sin destino remoto)
        else:
            # Reproducir audio local directamente
            outdata[:] = audio_chunk_to_send
            # (El video se maneja en la sección de captura)

    except sd.CallbackAbort:
        logging.info("Audio callback aborted by shutdown flag.")
        outdata[:] = self.zero_chunk_audio
        raise
    except Exception as e:
        logging.error(f"Error in audio/video callback: {e}", exc_info=False)
        try:
            outdata[:] = self.zero_chunk_audio
        except Exception as e_out:
            logging.error(f"Failed to output silence after error: {e_out}")


  def video_display_loop(self):
      """Bucle que muestra frames de video en una ventana OpenCV."""
      try:
          while not self.shutdown_flag.is_set():
              # Esperar por un nuevo frame o timeout
              frame = self.new_frame_ready.wait(timeout=0.05)
              if frame:
                  frame_to_show = self.current_video_frame
                  # Validar frame antes de mostrar
                  if frame_to_show is not None and frame_to_show.shape == (VIDEO_HEIGHT, VIDEO_WIDTH, 3):
                      cv2.imshow('Video', frame_to_show)
                  else:
                      logging.warning("Invalid frame in current_video_frame before display. Showing black.")
                      cv2.imshow('Video', self.zero_chunk_video)
                  self.new_frame_ready.clear()

              # Detectar si se presiona 'q' para salir
              key = cv2.waitKey(1) & 0xFF
              if key == ord('q'):
                  print("'Q' pressed in video window, initiating shutdown...")
                  self.shutdown_flag.set()
                  break

      except cv2.error as e:
           if not self.shutdown_flag.is_set(): logging.error(f"OpenCV error in video display: {e}")
      except Exception as e:
           if not self.shutdown_flag.is_set(): logging.error(f"Error in video display loop: {e}")
      finally:
          logging.info("Video display loop finished.")
          try: cv2.destroyWindow('Video'); cv2.waitKey(1)
          except: pass


  def setup_video_capture(self):
      """Inicializa y configura la captura de video."""
      try:
          self.cap = cv2.VideoCapture(0)  # Usa cámara predeterminada
          # Configurar resolución y FPS
          self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
          self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
          self.cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
          self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimizar buffer para menor latencia

          if not self.cap.isOpened():
              logging.error("Failed to open video capture device (index 0)")
              self.cap = None
              return False

          # Verificar que la cámara funciona leyendo un frame
          ret, frame = self.cap.read()
          if ret and frame is not None:
              h, w, _ = frame.shape
              logging.info(f"Video capture started. Actual resolution: {w}x{h}")
              if w != VIDEO_WIDTH or h != VIDEO_HEIGHT:
                  logging.warning(f"Camera resolution {w}x{h} differs from requested {VIDEO_WIDTH}x{VIDEO_HEIGHT}. Using actual.")
          else:
              logging.error("Failed to read initial frame from camera.")
              self.cap.release()
              self.cap = None
              return False

          self.next_video_capture_time = time.time()
          return True
      except Exception as e:
          logging.error(f"Exception during video capture setup: {e}")
          if self.cap: 
            try: self.cap.release()
            except: pass
          self.cap = None
          return False


  def mic_stream(self, callback_function):
      """Inicializa y gestiona los streams de audio/video y la red en modo micrófono."""
      self.audio_stream_instance = None
      stream_started = False
      try:
          # 1. Inicializar captura de video
          if not self.setup_video_capture():
              parser.exit(1, "Error: Failed to initialize video capture.")

          # 2. Iniciar hilo de visualización de video
          self.display_thread = threading.Thread(target=self.video_display_loop, name="VideoDisplayThread")
          self.display_thread.daemon = True
          self.display_thread.start()

          # 3. Iniciar hilos de red si hay destino
          if self.destination_address:
              self.sender_fragmenter_thread = threading.Thread(target=self.sender_fragmenter_loop, name="SenderFragmenterThread")
              self.sender_fragmenter_thread.daemon = True
              self.sender_fragmenter_thread.start()

              self.receiver_thread = threading.Thread(target=self.receiver_loop, name="ReceiverThread")
              self.receiver_thread.daemon = True
              self.receiver_thread.start()

          # 4. Verificar configuración de audio
          try:
               sd.check_input_settings(device=self.args.input_device, channels=NUMBER_OF_CHANNELS, dtype=np.int16, samplerate=self.args.frames_per_second)
               sd.check_output_settings(device=self.args.output_device, channels=NUMBER_OF_CHANNELS, dtype=np.int16, samplerate=self.args.frames_per_second)
          except (ValueError, sd.PortAudioError) as e:
               logging.error(f"Invalid audio device settings: {e}")
               try: print("\nAvailable audio devices:\n", sd.query_devices())
               except Exception as e_dev: print(f"Could not query audio devices: {e_dev}")
               parser.exit(1, f"Error with audio device settings: {e}")

          # 5. Crear e iniciar stream de audio
          self.audio_stream_instance = sd.Stream(
              device=(self.args.input_device, self.args.output_device),
              dtype=np.int16,
              samplerate=self.args.frames_per_second,
              blocksize=self.args.frames_per_chunk,
              channels=NUMBER_OF_CHANNELS,
              callback=callback_function,
              finished_callback=self.stream_finished_callback
          )

          # 6. Bucle principal
          with self.audio_stream_instance:
              stream_started = True
              rate = self.audio_stream_instance.samplerate
              chans = self.audio_stream_instance.channels
              print(f"Audio stream started. Sample Rate: {rate:.0f} Hz, Channels: {chans}")
              if self.destination_address:
                  dest_info = f"{self.destination_address}:{self.destination_port}"
                  print(f"Mode: Remote. Listening on {self.args.listening_port}, Sending to {dest_info}")
              else:
                  print("Mode: Local Loopback (Audio/Video).")
              print("Press Ctrl+C or 'q' in the video window to quit.")

              # Mantenerse en ejecución hasta shutdown
              while self.audio_stream_instance.active and not self.shutdown_flag.is_set():
                  time.sleep(0.1)

      except sd.PortAudioError as e:
          logging.error(f"PortAudio error during stream setup: {e}")
          try: print("\nAvailable audio devices:\n", sd.query_devices())
          except Exception as e_dev: print(f"Could not query audio devices: {e_dev}")
          parser.exit(1, f"\nError initializing audio stream: {e}")
      except Exception as e:
          logging.error(f"Error in mic_stream setup or main loop: {e}", exc_info=True)
          self.shutdown_flag.set()
      finally:
          logging.info("Exiting mic_stream function.")
          self.shutdown_flag.set()
          if not stream_started:
              self.audio_stream_instance = None


  def stream_finished_callback(self):
      """Callback llamado cuando termina el stream de audio."""
      logging.info("Audio stream finished callback triggered.")


  def file_stream(self, callback_function):
      """Implementación para reproducción desde archivo (no se puede con video)."""
      logging.error("File playback not implemented with video")
      print("Error: File playback is not supported in this version of Minimal.")
      parser.exit(1, "File playback is not supported.")


  def run(self):
      """Método principal que inicia la aplicación."""
      try:
          if self.args.list_devices:
              print("Available audio devices:")
              print(sd.query_devices())
              parser.exit(0)

          print("Starting Intercom... This time the 'Minimal' with Video version...")
          self.audio_stream(self._record_IO_and_play)

      except KeyboardInterrupt:
          print("\nInterrupted by user (Ctrl+C)")
      except Exception as e:
          logging.error(f"Unhandled exception in run: {e}", exc_info=True)
          self.shutdown_flag.set()


  def shutdown(self):
      """Realiza un apagado ordenado liberando todos los recursos."""
      print("\nShutting down gracefully...")
      self.shutdown_flag.set()

      # 1. Detener stream de audio
      print("  - Stopping audio stream...")
      if self.audio_stream_instance:
          try:
              self.audio_stream_instance.stop()
              self.audio_stream_instance.close()
              logging.debug("Audio stream stopped and closed.")
          except sd.PortAudioError as e:
               if "Stream is not active" not in str(e):
                   logging.warning(f"  - Ignored PortAudioError stopping stream: {e}")
          except Exception as e:
              logging.error(f"  - Error stopping/closing audio stream: {e}")
          self.audio_stream_instance = None

      # 2. Esperar a que terminen los hilos
      print("  - Waiting for worker threads to terminate...")
      threads_to_join = [
          (self.receiver_thread, "ReceiverThread"),
          (self.sender_fragmenter_thread, "SenderFragmenterThread"),
          (self.display_thread, "VideoDisplayThread")
      ]
      timeout_time = 5.0

      for thread, name in threads_to_join:
          if thread and thread.is_alive():
              logging.debug(f"  - Joining {name}...")
              try:
                  thread.join(timeout=timeout_time)
                  if thread.is_alive():
                      logging.warning(f"  - Thread {name} did not terminate cleanly.")
                  else:
                      logging.debug(f"  - Thread {name} terminated.")
              except Exception as e:
                  logging.error(f"  - Error joining thread {name}: {e}")

      # 3. Liberar recursos de video
      print("  - Releasing video capture device...")
      if self.cap:
          try: self.cap.release(); logging.debug("Video capture released.")
          except Exception as e: logging.error(f"  - Error releasing video capture: {e}")
          self.cap = None

      # 4. Cerrar ventanas OpenCV
      print("  - Closing OpenCV windows...")
      try: cv2.destroyAllWindows(); cv2.waitKey(5); logging.debug("OpenCV windows destroyed.")
      except Exception as e: logging.warning(f"  - Non-critical error closing OpenCV windows: {e}")

      # 5. Cerrar socket de red
      print("  - Closing network socket...")
      if self.sock:
          try: self.sock.close(); logging.debug("Network socket closed.")
          except Exception as e: logging.error(f"  - Error closing socket: {e}")
          self.sock = None

      # 6. Limpiar colas
      print("  - Clearing queues...")
      if hasattr(self, 'packet_queue') and self.packet_queue:
          while not self.packet_queue.empty():
              try: self.packet_queue.get_nowait(); self.packet_queue.task_done()
              except queue.Empty: break
              except Exception as e: logging.warning(f"Error clearing packet_queue: {e}"); break
          logging.debug("Packet queue cleared.")

      print("Shutdown complete.")


# Referencia global para el manejador de señales
intercom_instance = None

def signal_handler(sig, frame):
  """Manejador de señales para cierre ordenado al recibir Ctrl+C o SIGTERM."""
  global intercom_instance
  print(f"\nSignal {sig} received.")
  if intercom_instance and not intercom_instance.shutdown_flag.is_set():
      print("Initiating shutdown sequence from signal handler...")
      intercom_instance.shutdown_flag.set()
  elif intercom_instance and intercom_instance.shutdown_flag.is_set():
       print("Shutdown already in progress.")
  else:
       print("Intercom instance not available or shutdown signal received before instance creation. Exiting.")
       sys.exit(1)

if __name__ == "__main__":
 
  intercom = VideoAudioIntercom(args)

  #Registrar manejadores de señal
  signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
  signal.signal(signal.SIGTERM, signal_handler) # kill/system shutdown

  #Ejecutar con manejo de errores y limpieza garantizada
  exit_code = 0
  try:
      intercom.run()
  except SystemExit as e:
      msg = f"Exiting application ({e})" if str(e) != '0' else "Exiting application."
      print(msg)
      exit_code = e.code if isinstance(e.code, int) else 0
  except Exception as e:
      logging.critical(f"Critical unhandled exception in main execution: {e}", exc_info=True)
      exit_code = 1
      if intercom and not intercom.shutdown_flag.is_set():
          intercom.shutdown_flag.set()
  finally:
      if intercom:
          intercom.shutdown()

      print("Intercom execution finished!")
      sys.exit(exit_code)
      
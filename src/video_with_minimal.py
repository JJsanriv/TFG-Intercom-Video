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

# Configuración de argumentos para la línea de comandos
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

# Constantes para la configuración de video y audio
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
      # Enlazar socket al puerto de escucha
      self.sock.bind(("0.0.0.0", self.args.listening_port))
      self.sock.settimeout(0.01)
      self.destination_address = self.args.destination_address
      self.destination_port = self.args.destination_port
      self.shutdown_flag = threading.Event()

      # Preparar chunks de audio y video nulos para usar cuando no hay datos válidos
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

      self.new_frame_ready = threading.Event()  # Para sincronizar proceso de display

      # Cola para pasar frames del hilo de captura al hilo principal/de envío
      self.video_frame_queue = queue.Queue(maxsize=2)  # Tamaño pequeño para mantener frames recientes
      self.video_capture_thread = None

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
        # Validar audio y usar silencio si es inválido
        if audio_chunk is None or audio_chunk.size == 0 or not np.isfinite(audio_chunk).all() or \
           audio_chunk.shape != (self.args.frames_per_chunk, NUMBER_OF_CHANNELS):
            if audio_chunk is not None:
                 logging.warning(f"Invalid audio chunk shape/content: {audio_chunk.shape if audio_chunk is not None else 'None'}. Using silence.")
            packed_audio = self.zero_audio_bytes
        else:
            packed_audio = audio_chunk.tobytes()

        audio_size = len(packed_audio)

        # Validar video y enviarlo solo si es válido
        if video_frame is None or video_frame.size == 0 or not np.isfinite(video_frame).all() or \
           video_frame.shape != (VIDEO_HEIGHT, VIDEO_WIDTH, 3):
            if video_frame is not None:
                logging.warning(f"Invalid video frame shape/content: {video_frame.shape if video_frame is not None else 'None'}. Omitting video.")
            # Paquete solo con audio (has_video=0)
            return struct.pack(self.PACKET_HEADER_FORMAT, audio_size, 0, 0) + packed_audio
        else:
            packed_video = video_frame.tobytes()
            # Paquete con audio y video (has_video=1)
            return struct.pack(self.PACKET_HEADER_FORMAT, audio_size, 0, 1) + packed_audio + packed_video

    except Exception as e:
        logging.error(f"Error in pack_audio_video: {e}")
        # Fallback a paquete de silencio en caso de error
        return struct.pack(self.PACKET_HEADER_FORMAT, len(self.zero_audio_bytes), 0, 0) + self.zero_audio_bytes


  def unpack_audio_video(self, packed_chunk):
    try:
        # Validar tamaño mínimo del paquete
        if packed_chunk is None or len(packed_chunk) < self.PACKET_HEADER_SIZE:
            logging.warning(f"Received packet too short ({len(packed_chunk) if packed_chunk is not None else 'None'}) for header. Returning silence/blank.")
            return self.zero_chunk_audio, None

        # Extraer información de la cabecera del paquete
        audio_size, packet_number, has_video = struct.unpack(self.PACKET_HEADER_FORMAT, packed_chunk[:self.PACKET_HEADER_SIZE])

        # Validar tamaño de audio recibido
        if not (0 < audio_size <= (MAX_PAYLOAD_BYTES - self.PACKET_HEADER_SIZE)):
            logging.warning(f"Invalid audio size in packet {packet_number}: {audio_size}. Returning silence.")
            return self.zero_chunk_audio, None

        # Verificar si el paquete contiene todos los datos de audio indicados
        audio_end_index = self.PACKET_HEADER_SIZE + audio_size
        if audio_end_index > len(packed_chunk):
            logging.warning(f"Reported audio size ({audio_size}) for packet {packet_number} exceeds packet length ({len(packed_chunk)}). Returning silence.")
            return self.zero_chunk_audio, None

        # Procesar datos de audio
        audio_data = packed_chunk[self.PACKET_HEADER_SIZE:audio_end_index]
        audio_chunk = self.zero_chunk_audio  # Valor por defecto
        expected_audio_bytes = self.args.frames_per_chunk * NUMBER_OF_CHANNELS * AUDIO_SAMPLE_SIZE
        
        if len(audio_data) != expected_audio_bytes:
            logging.warning(f"Audio data size mismatch (got {len(audio_data)}, expected {expected_audio_bytes}) for packet {packet_number}. Using silence.")
        else:
            try:
                audio_chunk_candidate = np.frombuffer(audio_data, np.int16).reshape(-1, NUMBER_OF_CHANNELS)
                if audio_chunk_candidate.shape == (self.args.frames_per_chunk, NUMBER_OF_CHANNELS):
                    audio_chunk = audio_chunk_candidate
                else:
                     logging.warning(f"Reshaped audio data has wrong shape {audio_chunk_candidate.shape} for packet {packet_number}. Using silence.")
            except ValueError as e:
                logging.warning(f"Could not reshape audio data (size {len(audio_data)}) for packet {packet_number}: {e}. Using silence.")

        # Si no hay video, devolver solo audio
        if has_video == 0:
            return audio_chunk, None

        # Procesar datos de video si están presentes
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
        # Validar tamaño mínimo del fragmento
        if fragment is None or len(fragment) < self.FRAGMENT_HEADER_SIZE:
            logging.warning(f"Received fragment too short ({len(fragment) if fragment is not None else 'None'} bytes) from {addr}. Discarding.")
            return None

        # Extraer información de la cabecera del fragmento
        fragment_id, total_fragments, fragment_index = struct.unpack(self.FRAGMENT_HEADER_FORMAT, fragment[:self.FRAGMENT_HEADER_SIZE])

        # Validar parámetros de fragmentación
        if not (0 < total_fragments <= 1000):
            logging.warning(f"Invalid total_fragments value ({total_fragments}) for ID {fragment_id} from {addr}. Discarding.")
            return None
        if not (0 <= fragment_index < total_fragments):
            logging.warning(f"Invalid fragment_index ({fragment_index}) for ID {fragment_id}, total {total_fragments} from {addr}. Discarding.")
            return None

        # Extraer datos del fragmento
        fragment_data = fragment[self.FRAGMENT_HEADER_SIZE:]
        if not fragment_data:
            logging.warning(f"Empty fragment data for ID {fragment_id}, index {fragment_index} from {addr}. Discarding.")
            return None

        # Gestionar conjunto de fragmentos para un paquete
        current_time = time.time()
        if fragment_id not in self.fragments_received:
             # Limpiar conjuntos antiguos si hay demasiados
            if len(self.fragments_received) > 50:
                oldest_id = min(self.fragments_received, key=lambda k: self.fragments_received[k]['time'])
                del self.fragments_received[oldest_id]
                logging.debug(f"Cleaned up oldest incomplete fragment set {oldest_id} due to memory limit.")

            self.fragments_received[fragment_id] = {'total': total_fragments, 'received': 0, 'fragments': {}, 'time': current_time}

        # Procesar solo fragmentos no duplicados
        if fragment_index not in self.fragments_received[fragment_id]['fragments']:
            self.fragments_received[fragment_id]['fragments'][fragment_index] = fragment_data
            self.fragments_received[fragment_id]['received'] += 1

        # Verificar si se han recibido todos los fragmentos
        fragment_info = self.fragments_received[fragment_id]
        if fragment_info['received'] == fragment_info['total']:
            try:
                # Reensamblar fragmentos en orden
                ordered_fragments = [fragment_info['fragments'][i] for i in range(fragment_info['total'])]
                complete_packet = b"".join(ordered_fragments)
            except KeyError:
                 logging.warning(f"Missing fragment during final assembly for packet ID {fragment_id}. Discarding.")
                 del self.fragments_received[fragment_id]
                 return None

            # Verificar paquete reensamblado
            if not complete_packet:
                logging.warning(f"Reassembled empty packet from fragments for ID {fragment_id}. Discarding.")
                del self.fragments_received[fragment_id]
                return None
            if len(complete_packet) < self.PACKET_HEADER_SIZE:
                logging.warning(f"Reassembled packet too small (size {len(complete_packet)}) for ID {fragment_id}. Discarding.")
                del self.fragments_received[fragment_id]
                return None

            # Limpiar entrada y retornar paquete completo
            del self.fragments_received[fragment_id]
            return complete_packet

    except struct.error as e:
         logging.warning(f"Received malformed fragment header from {addr}: {e}. Discarding.")
    except KeyError:
         logging.debug(f"Received fragment for already processed/cleaned ID {fragment_id} from {addr}. Discarding.")
    except Exception as e:
        logging.error(f"Error reassembling packet: {e}")

    return None


  def clean_old_fragments(self, max_age=0.5):
      """Elimina conjuntos de fragmentos incompletos después de max_age segundos."""
      current_time = time.time()
      to_delete = [
          fid for fid, data in self.fragments_received.items()
          if current_time - data.get('time', current_time) > max_age
      ]
      for fragment_id in to_delete:
          if self.fragments_received.pop(fragment_id, None):
               logging.debug(f"Cleaned up old/incomplete fragment set {fragment_id}")


  def sender_fragmenter_loop(self):
      """Toma paquetes de la cola y los envía fragmentados a la dirección destino."""
      fragment_id_counter = 0
      try:
          # Verificar existencia de destino
          dest_addr = (self.destination_address, self.destination_port) if self.destination_address else None
          if not dest_addr:
              logging.warning("SenderFragmenterThread started without destination address. Exiting.")
              return

          while not self.shutdown_flag.is_set():
              try:
                  # Obtener próximo paquete a enviar
                  packet_data, packet_number = self.packet_queue.get(timeout=0.1)

                  if packet_data is None:
                      self.packet_queue.task_done()
                      continue

                  # Actualizar cabecera con número de paquete actual
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

                  # Calcular fragmentación del paquete
                  header_size = self.FRAGMENT_HEADER_SIZE
                  payload_size = MAX_PAYLOAD_BYTES - header_size
                  total_size = len(full_packet_to_send)
                  total_fragments = max(1, (total_size + payload_size - 1) // payload_size)
                  fragment_id = fragment_id_counter
                  fragment_id_counter += 1

                  # Enviar cada fragmento
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
                  continue  # Timeout normal en get()
              except Exception as e:
                  if not self.shutdown_flag.is_set():
                      logging.error(f"Error in sender_fragmenter_loop's main try: {e}")

      except Exception as e:
          if not self.shutdown_flag.is_set():
              logging.error(f"Unhandled error in sender_fragmenter_loop setup: {e}")
      finally:
          logging.info("Sender/Fragmenter loop finished.")


  def receiver_loop(self):
      """Recibe fragmentos, los reensambla y los coloca en el buffer para reproducción."""
      last_cleanup_time = time.time()
      CLEANUP_INTERVAL = 1.0
      try:
          while not self.shutdown_flag.is_set():
              try:
                  # Recibir fragmento de red
                  fragment, addr = self.sock.recvfrom(MAX_PAYLOAD_BYTES + self.FRAGMENT_HEADER_SIZE + 100)
                  packet = self.reassemble_packet(fragment, addr)
                  if packet:
                      # Añadir paquete completo al buffer
                      self.jitter_buffer.append(packet)
                      # Proporcionar paquete al callback si no hay ninguno en proceso
                      if self.latest_packet is None and self.jitter_buffer:
                          try:
                            self.latest_packet = self.jitter_buffer.popleft()
                          except IndexError:
                            pass  # Buffer podría haberse vaciado entre la comprobación y popleft()

                  # Limpiar periódicamente fragmentos obsoletos
                  current_time = time.time()
                  if current_time - last_cleanup_time > CLEANUP_INTERVAL:
                      self.clean_old_fragments()
                      last_cleanup_time = current_time

              except socket.timeout:
                  continue  # Timeout normal del socket
              except socket.error as e:
                   if self.shutdown_flag.is_set(): break
                   logging.error(f"Socket error in receiver_loop: {e}")
                   time.sleep(0.1)
              except Exception as e:
                  if not self.shutdown_flag.is_set():
                      logging.error(f"Error in receiver_loop: {e}", exc_info=False)
      except Exception as e:
          if not self.shutdown_flag.is_set():
              logging.error(f"Unhandled error in receiver_loop: {e}")
      finally:
          logging.info("Receiver loop finished.")


  def send_audio_video(self, packed_chunk_no_pkt_num):
      """Encola un paquete de audio/video para su envío fragmentado."""
      if self.destination_address and packed_chunk_no_pkt_num is not None:
          try:
              current_packet_number = self.packet_counter
              self.packet_counter += 1
              packet_to_queue = (packed_chunk_no_pkt_num, current_packet_number)

              # Intentar encolar sin bloqueo
              self.packet_queue.put_nowait(packet_to_queue)

          except queue.Full:
              # Si la cola está llena, descartar el paquete más antiguo y reintentar
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
      self.latest_packet = None  # Marcar como consumido

      # Si no hay paquete pendiente, intentar obtener del buffer
      if packet is None and self.jitter_buffer:
          try:
              packet = self.jitter_buffer.popleft()
          except IndexError:
              packet = None

      return packet


  def _record_IO_and_play(self, indata, outdata, frames, time_info, status):
    """Callback principal que procesa audio/video capturado y recibido."""
    if self.shutdown_flag.is_set():
        outdata[:] = self.zero_chunk_audio
        raise sd.CallbackAbort

    if status: logging.warning(f"Audio callback status: {status}")

    try:
        # 1. Validar audio de entrada
        expected_shape = (self.args.frames_per_chunk, NUMBER_OF_CHANNELS)
        if indata is None or not np.isfinite(indata).all() or indata.shape != expected_shape:
            if indata is not None:
                 logging.warning(f"Invalid audio input shape/content: {indata.shape if indata is not None else 'None'}. Using silence.")
            audio_chunk_to_send = self.zero_chunk_audio.copy()
        else:
            audio_chunk_to_send = indata

        # 2. Obtener frame de video más reciente (no bloqueante)
        current_frame_to_send_or_display = None
        try:
            current_frame_to_send_or_display = self.video_frame_queue.get_nowait()
        except queue.Empty:
            pass
        except Exception as e:
            logging.warning(f"Error getting frame from video queue: {e}")

        # 3. Modo red (con destino remoto configurado)
        if self.destination_address:
            # Empaquetar y enviar audio/video
            packed_chunk = self.pack_audio_video(audio_chunk_to_send, current_frame_to_send_or_display)
            self.send_audio_video(packed_chunk)

            # Recibir y procesar datos remotos
            received_packet = self.receive_audio_video()
            audio_chunk_to_play = self.zero_chunk_audio
            received_video_frame = None

            if received_packet:
                unpacked_audio, unpacked_video = self.unpack_audio_video(received_packet)
                if unpacked_audio is not None and unpacked_audio.shape == expected_shape:
                    audio_chunk_to_play = unpacked_audio
                if unpacked_video is not None:
                    received_video_frame = unpacked_video

            # Reproducir audio recibido
            outdata[:] = audio_chunk_to_play

            # Actualizar frame de video si se recibió uno nuevo
            if received_video_frame is not None:
                 self.current_video_frame = received_video_frame
                 self.new_frame_ready.set()  # Notificar al hilo de display

        # 4. Modo local (sin conexión remota)
        else:
            # Reproducir audio local directamente
            outdata[:] = audio_chunk_to_send

            # Actualizar frame de video local para mostrar
            if current_frame_to_send_or_display is not None:
                self.current_video_frame = current_frame_to_send_or_display
                self.new_frame_ready.set()  # Notificar al hilo de display

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

  def _video_capture_loop(self):
      """
      Hilo dedicado para capturar frames de video continuamente y ponerlos en una cola.
      """
      logging.info("Video capture thread started.")
      capture_interval = 1.0 / VIDEO_FPS

      while not self.shutdown_flag.is_set():
          start_time = time.monotonic()

          if self.cap and self.cap.isOpened():
              try:
                  ret, frame = self.cap.read()
                  if ret and frame is not None:
                      # Validar dimensiones del frame antes de encolar
                      if frame.shape == (VIDEO_HEIGHT, VIDEO_WIDTH, 3):
                          try:
                              # Intentar encolar sin bloquear; si está llena, descartar el frame más antiguo
                              self.video_frame_queue.put_nowait(frame)
                          except queue.Full:
                              try:
                                  self.video_frame_queue.get_nowait()  # Descartar frame antiguo
                                  self.video_frame_queue.put_nowait(frame)
                                  logging.debug("Video queue full, discarded oldest frame and added newest.")
                              except (queue.Empty, queue.Full) as e:
                                  logging.warning(f"Video queue issue during overflow handling: {type(e).__name__}")
                              except Exception as e_inner:
                                  logging.error(f"Error managing full video queue: {e_inner}")
                          except Exception as e_put:
                              logging.error(f"Error putting frame into video queue: {e_put}")
                      else:
                          logging.warning(f"Captured frame has incorrect shape: {frame.shape}. Expected {(VIDEO_HEIGHT, VIDEO_WIDTH, 3)}. Discarding.")
                  elif not ret:
                      logging.warning("cap.read() returned False. Camera issue?")
                      time.sleep(0.5)

              except cv2.error as e:
                  if not self.shutdown_flag.is_set(): logging.error(f"OpenCV error during capture: {e}")
                  time.sleep(1.0)
              except Exception as e:
                  if not self.shutdown_flag.is_set(): logging.error(f"Unexpected error in video capture loop: {e}")
                  time.sleep(1.0)
          else:
              logging.debug("Video capture device not ready or closed.")
              time.sleep(0.5)

          # Mantener aproximadamente el FPS deseado
          elapsed_time = time.monotonic() - start_time
          sleep_time = max(0, capture_interval - elapsed_time)
          if sleep_time > 0:
             time.sleep(sleep_time)

      logging.info("Video capture thread finished.")


  def video_display_loop(self):
      """Muestra frames de video en ventana OpenCV."""
      try:
          cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
          cv2.resizeWindow('Video', VIDEO_WIDTH, VIDEO_HEIGHT)

          while not self.shutdown_flag.is_set():
              # Esperar nuevo frame o timeout
              got_frame = self.new_frame_ready.wait(timeout=0.1)
              if got_frame:
                  # Obtener copia del frame actual para reducir race conditions
                  frame_to_show = self.current_video_frame.copy()

                  # Validar frame antes de mostrarlo
                  if frame_to_show is not None and frame_to_show.shape == (VIDEO_HEIGHT, VIDEO_WIDTH, 3):
                      cv2.imshow('Video', frame_to_show)
                  else:
                      cv2.imshow('Video', self.zero_chunk_video)

                  self.new_frame_ready.clear()

              # Procesar teclas y eventos de ventana
              key = cv2.waitKey(1) & 0xFF
              if key == ord('q'):
                  print("'q' pressed in video window, initiating shutdown...")
                  self.shutdown_flag.set()
                  break
              elif cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
                  print("Video window closed by user, initiating shutdown...")
                  self.shutdown_flag.set()
                  break

      except cv2.error as e:
           if not self.shutdown_flag.is_set() and "could not be found" not in str(e) and "NULL window" not in str(e):
               logging.error(f"OpenCV error in video display: {e}")
      except Exception as e:
           if not self.shutdown_flag.is_set():
               logging.error(f"Error in video display loop: {e}")
      finally:
          logging.info("Video display loop finished.")
          try: cv2.destroyWindow('Video'); cv2.waitKey(1)
          except: pass


  def setup_video_capture(self):
      """Inicializa la cámara y configura parámetros de captura."""
      try:
          self.cap = cv2.VideoCapture(0)  # Usar cámara predeterminada
          if not self.cap.isOpened():
              logging.error("Failed to open video capture device (index 0)")
              self.cap = None
              return False

          # Configurar resolución y FPS
          self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
          self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
          self.cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
          # Reducir buffer para menor latencia
          self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

          # Tiempo para estabilización de la cámara
          time.sleep(0.5)

          # Validar que la cámara funciona
          ret, frame = self.cap.read()
          if ret and frame is not None:
              # Obtener configuración real aplicada por la cámara
              actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
              actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
              actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
              logging.info(f"Video capture started. Requested: {VIDEO_WIDTH}x{VIDEO_HEIGHT}@{VIDEO_FPS}fps. Actual: {actual_w}x{actual_h}@{actual_fps:.2f}fps")
              if actual_w != VIDEO_WIDTH or actual_h != VIDEO_HEIGHT:
                  logging.warning(f"Camera resolution {actual_w}x{actual_h} differs from requested {VIDEO_WIDTH}x{VIDEO_HEIGHT}. Using actual.")
          else:
              logging.error("Failed to read initial frame from camera after setup.")
              self.cap.release()
              self.cap = None
              return False

          return True
      except Exception as e:
          logging.error(f"Exception during video capture setup: {e}")
          if self.cap:
            try: self.cap.release()
            except: pass
          self.cap = None
          return False


  def mic_stream(self, callback_function):
      """Inicializa y gestiona los streams de audio/video y la red."""
      self.audio_stream_instance = None
      stream_started = False
      try:
          # 1. Inicializar captura de video
          if not self.setup_video_capture():
              parser.exit(1, "Error: Failed to initialize video capture.")

          # 2. Iniciar hilo dedicado a la captura de video
          self.video_capture_thread = threading.Thread(target=self._video_capture_loop, name="VideoCaptureThread")
          self.video_capture_thread.daemon = True
          self.video_capture_thread.start()

          # 3. Iniciar hilo de visualización de video
          self.display_thread = threading.Thread(target=self.video_display_loop, name="VideoDisplayThread")
          self.display_thread.daemon = True
          self.display_thread.start()

          # 4. Iniciar hilos de red si hay destino configurado
          if self.destination_address:
              self.sender_fragmenter_thread = threading.Thread(target=self.sender_fragmenter_loop, name="SenderFragmenterThread")
              self.sender_fragmenter_thread.daemon = True
              self.sender_fragmenter_thread.start()

              self.receiver_thread = threading.Thread(target=self.receiver_loop, name="ReceiverThread")
              self.receiver_thread.daemon = True
              self.receiver_thread.start()

          # 5. Verificar configuración de dispositivos de audio
          try:
               sd.check_input_settings(device=self.args.input_device, channels=NUMBER_OF_CHANNELS, dtype=np.int16, samplerate=self.args.frames_per_second)
               sd.check_output_settings(device=self.args.output_device, channels=NUMBER_OF_CHANNELS, dtype=np.int16, samplerate=self.args.frames_per_second)
          except (ValueError, sd.PortAudioError) as e:
               logging.error(f"Invalid audio device settings: {e}")
               try: print("\nAvailable audio devices:\n", sd.query_devices())
               except Exception as e_dev: print(f"Could not query audio devices: {e_dev}")
               parser.exit(1, f"Error with audio device settings: {e}")

          # 6. Crear e iniciar stream de audio
          self.audio_stream_instance = sd.Stream(
              device=(self.args.input_device, self.args.output_device),
              dtype=np.int16,
              samplerate=self.args.frames_per_second,
              blocksize=self.args.frames_per_chunk,
              channels=NUMBER_OF_CHANNELS,
              callback=callback_function,
              finished_callback=self.stream_finished_callback
          )

          # 7. Bucle principal de la aplicación
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
              print("Press Ctrl+C or 'q'/'close' the video window to quit.")

              # Mantener ejecución y monitorear hilos
              while not self.shutdown_flag.is_set():
                    # Verificar estado de hilos esenciales
                    essential_threads = [
                        (self.video_capture_thread, "Video Capture"),
                        (self.display_thread, "Video Display")
                    ]
                    if self.destination_address:
                        essential_threads.extend([
                           (self.sender_fragmenter_thread, "Sender"),
                           (self.receiver_thread, "Receiver")
                        ])

                    all_running = True
                    for thread, name in essential_threads:
                        if thread and not thread.is_alive():
                            logging.error(f"Essential thread '{name}' terminated unexpectedly.")
                            all_running = False
                            break
                    if not all_running:
                        self.shutdown_flag.set()
                        break

                    # Verificar estado del stream de audio
                    if not self.audio_stream_instance.active:
                         logging.error("Audio stream became inactive unexpectedly.")
                         self.shutdown_flag.set()
                         break

                    time.sleep(0.2)

      except sd.PortAudioError as e:
          logging.error(f"PortAudio error during stream setup or execution: {e}")
          try: print("\nAvailable audio devices:\n", sd.query_devices())
          except Exception as e_dev: print(f"Could not query audio devices: {e_dev}")
          parser.exit(1, f"\nError with audio stream: {e}")
      except Exception as e:
          logging.error(f"Error in mic_stream setup or main loop: {e}", exc_info=True)
          self.shutdown_flag.set()
      finally:
          logging.info("Exiting mic_stream function.")
          self.shutdown_flag.set()
          if not stream_started and self.audio_stream_instance:
              # Cerrar stream si falló antes del bloque 'with'
              try:
                  if not self.audio_stream_instance.closed:
                      self.audio_stream_instance.close()
              except Exception as e_close:
                   logging.warning(f"Error closing audio stream instance after early failure: {e_close}")


  def stream_finished_callback(self):
      """Callback ejecutado cuando finaliza el stream de audio."""
      if not self.shutdown_flag.is_set():
          logging.info("Audio stream finished callback triggered unexpectedly.")


  def file_stream(self, callback_function):
      """Reproducción desde archivo (no implementada con video)."""
      logging.error("File playback not implemented with video")
      print("Error: File playback is not supported in this version.")
      parser.exit(1, "File playback is not supported.")


  def run(self):
      """Método principal que inicia la aplicación."""
      try:
          if self.args.list_devices:
              print("Available audio devices:")
              print(sd.query_devices())
              parser.exit(0)

          print("Starting video/audio intercom...")
          self.audio_stream(self._record_IO_and_play)

      except KeyboardInterrupt:
          print("\nInterrupted by user (Ctrl+C)")
          self.shutdown_flag.set()
      except SystemExit:
          # Permitir salidas controladas
          raise
      except Exception as e:
          logging.error(f"Unhandled exception in run: {e}", exc_info=True)
          self.shutdown_flag.set()


  def shutdown(self):
      """Realiza un apagado ordenado liberando todos los recursos."""
      if self.shutdown_flag.is_set():
        return

      print("\nShutting down gracefully...")
      self.shutdown_flag.set()

      # 1. Detener stream de audio
      print("  - Stopping audio stream...")
      if self.audio_stream_instance:
          try:
              if self.audio_stream_instance.active:
                  self.audio_stream_instance.stop()
              if not self.audio_stream_instance.closed:
                   self.audio_stream_instance.close()
              logging.debug("Audio stream stopped and closed.")
          except sd.PortAudioError as e:
               if "Stream is not active" not in str(e) and "Invalid stream" not in str(e):
                   logging.warning(f"  - Ignored PortAudioError stopping stream: {e}")
          except AttributeError:
               logging.warning("  - Audio stream instance was None during shutdown.")
          except Exception as e:
              logging.error(f"  - Error stopping/closing audio stream: {e}")
          finally:
               self.audio_stream_instance = None

      # 2. Esperar a que terminen los hilos
      print("  - Waiting for worker threads to terminate...")
      threads_to_join = [
          (self.video_capture_thread, "VideoCaptureThread"),
          (self.receiver_thread, "ReceiverThread"),
          (self.sender_fragmenter_thread, "SenderFragmenterThread"),
          (self.display_thread, "VideoDisplayThread")
      ]
      termination_timeout = 2.0

      for thread, name in threads_to_join:
          if thread and thread.is_alive():
              logging.debug(f"  - Joining {name}...")
              try:
                  thread.join(timeout=termination_timeout)
                  if thread.is_alive():
                      logging.warning(f"  - Thread {name} did not terminate cleanly after {termination_timeout}s.")
                  else:
                      logging.debug(f"  - Thread {name} terminated.")
              except RuntimeError:
                  logging.warning(f"  - Could not join thread {name} (possibly not started).")
              except Exception as e:
                  logging.error(f"  - Error joining thread {name}: {e}")

      # 3. Liberar recursos de video
      print("  - Releasing video capture device...")
      if self.cap:
          try:
              self.cap.release()
              logging.debug("Video capture released.")
          except Exception as e:
              logging.error(f"  - Error releasing video capture: {e}")
          finally:
              self.cap = None

      # 4. Cerrar ventanas OpenCV
      print("  - Closing OpenCV windows...")
      try:
          cv2.destroyAllWindows()
          logging.debug("OpenCV windows destroyed.")
      except Exception as e:
          logging.warning(f"  - Non-critical error closing OpenCV windows: {e}")

      # 5. Cerrar socket de red
      print("  - Closing network socket...")
      if self.sock:
          try:
              self.sock.close()
              logging.debug("Network socket closed.")
          except Exception as e:
              logging.error(f"  - Error closing socket: {e}")
          finally:
               self.sock = None

      # 6. Limpiar colas
      print("  - Clearing queues...")
      queues_to_clear = [self.packet_queue, self.video_frame_queue]
      for q in queues_to_clear:
          if q:
              while not q.empty():
                  try:
                      q.get_nowait()
                  except queue.Empty:
                      break
                  except Exception as e:
                      logging.warning(f"Error clearing queue item: {e}")
                      break
      logging.debug("Queues cleared.")

      print("Shutdown complete.")


# Referencia global para el manejador de señales
intercom = None

def signal_handler(sig, frame):
  """Maneja señales Ctrl+C y SIGTERM para cierre ordenado."""
  global intercom
  signal_name = signal.Signals(sig).name
  print(f"\nSignal {signal_name} ({sig}) received.")
  if intercom:
      if not intercom.shutdown_flag.is_set():
           print("Initiating shutdown sequence from signal handler...")
           intercom.shutdown_flag.set()
      else:
           print("Shutdown already in progress (signal received again).")
  else:
       print("Intercom instance not available. Forcing exit.")
       sys.exit(1)

if __name__ == "__main__":
  # 1. Crear instancia de VideoAudioIntercom
  final_exit_code = 0
  try:
       intercom = VideoAudioIntercom(args)
  except Exception as e_init:
       logging.critical(f"Failed to initialize VideoAudioIntercom: {e_init}", exc_info=True)
       sys.exit(1)

  # 2. Registrar manejadores de señales
  try:
      signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
      signal.signal(signal.SIGTERM, signal_handler) # kill/system shutdown
  except ValueError:
      logging.warning("Could not set signal handlers. Running without custom signal handling.")

  # 3. Ejecutar aplicación con manejo completo de errores
  try:
      intercom.run()
  except SystemExit as e:
      msg = f"Exiting application (code: {e.code})" if str(e.code) != '0' else "Exiting application cleanly."
      print(msg)
      final_exit_code = e.code if isinstance(e.code, int) else 0
  except KeyboardInterrupt:
       print("\nKeyboardInterrupt caught in main block (fallback).")
       if intercom and not intercom.shutdown_flag.is_set():
           intercom.shutdown_flag.set()
       final_exit_code = 130
  except Exception as e:
      logging.critical(f"Critical unhandled exception in main execution: {e}", exc_info=True)
      final_exit_code = 1
      if intercom and not intercom.shutdown_flag.is_set():
          intercom.shutdown_flag.set()
  finally:
      if intercom:
          intercom.shutdown()

      print("Minimal with video Intercom finished!")
      sys.exit(final_exit_code)

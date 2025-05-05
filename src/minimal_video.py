#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""
Minimal_Video: Extiende de minimal.py para agregar transmisión/visualización de video sin
compresión/encodificación, usando raw data. Incluye opción verbose (--show_stats, --show_samples y --show_spectrum).
- Se transmite video full‐duplex vía UDP sin usar colas, es decir, se envía el frame directamente.
- El flag --show_video habilita la visualización.

Utiliza un socket UDP para transmisión y fragmenta los frames.
Header (big-endian): FrameID(I), TotalFrags(H), FragIdx(H), Width(H), Height(H)

Nuevos parámetros:
--video_payload_size : Tamaño deseado (bytes) payload video/fragmento UDP (defecto 32000).
--width              : Ancho del video (defecto 320).
--height             : Alto del video (defecto 180).
--fps                : Frames por segundo video (defecto 30).
--show_video         : Habilita la visualización del video (desactivado por defecto).
--video_port         : Puerto para transmitir/recibir video (defecto 4445).
"""

import cv2
import socket
import struct
import threading
import time
import math
import numpy as np
import select
import argparse
import psutil
import logging
import minimal


spinner = minimal.spinning_cursor()


def int_or_str(text):
 try:
     return int(text)
 except ValueError:
     return text


# Utilizamos el parser de minimal y le agregamos los argumentos extra
if not hasattr(minimal, 'parser'):
 minimal.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser = minimal.parser


parser.add_argument("-v", "--video_payload_size", type=int, default=32000,
                 help="Tamaño deseado (bytes) payload video/fragmento UDP (defecto 32000).")
parser.add_argument("-w", "--width", type=int, default=320, help="Ancho video (defecto 320)")
parser.add_argument("-g", "--height", type=int, default=180, help="Alto video (defecto 180)")
parser.add_argument("-z", "--fps", type=int, default=30, help="Frames por segundo video (defecto 30)")
parser.add_argument("--show_video", action="store_true", default=False,
                 help="Habilita la visualización del video (desactivado por defecto).")
parser.add_argument("--video_port", type=int, default=4445,
                 help="Puerto para transmitir/recibir video (defecto 4445).")


args = None


class Minimal_Video(minimal.Minimal):
 def __init__(self):
      global args
      # Aseguramos que se parseen los argumentos si aún no se han hecho
      if args is None:
          args = minimal.parser.parse_args()
      minimal.args = args

      # Llama al constructor de la clase base
      super().__init__()

      # Configuración del socket de vídeo
      self.video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
      self.video_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      # Aumentamos el buffer para mejorar rendimiento
      self.video_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8388608)
      self.video_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8388608)
      try:
          self.video_sock.bind(("0.0.0.0", args.video_port))
      except OSError as e:
          print(f"Error bind socket video: {e}")
          raise
      self.video_addr = (args.destination_address, args.video_port)
      self.video_sock.setblocking(False)

      # Estructuras necesarias para recepción y reconstrucción de frames de vídeo
      self.recv_frames = {}
      self.recv_frames_lock = threading.Lock()
      self.latest_received_frame = None
      self.latest_received_frame_lock = threading.Lock()

      # Configuración para fragmentación de los frames
      self._header_format = "!IHHHH"
      try:
          max_udp_payload = getattr(self, 'MAX_PAYLOAD_BYTES', 32768)
      except AttributeError:
          max_udp_payload = 32768
      self.header_size = 12
      self.max_payload_possible = max_udp_payload - self.header_size
      self.effective_video_payload_size = max(1, min(args.video_payload_size, self.max_payload_possible))
      if self.effective_video_payload_size != args.video_payload_size:
          print(f"Aviso: --video_payload_size ajustado a {self.effective_video_payload_size} bytes.")


      # Si --show_video NO se pasa, la cámara no se inicializa y la aplicación
      # se configurará solo para la recepción de vídeo.
      self.cap = None
      self.capture_enabled = False
      self.width = 0
      self.height = 0
      self.fps = 0
      self.latest_captured_frame = None
      self.latest_captured_frame_lock = threading.Lock()
      self.frame_id_counter = 0  # Para numerar los frames enviados
      
      # Variables para optimización
      self.next_cleanup_time = time.time() + 1.0
      self.last_show_time = 0


      if args.show_video:
          print("Flag --show_video detectado. Intentando inicializar la cámara...")
          try:
              self.cap = cv2.VideoCapture(0)
              if not self.cap.isOpened():
                  raise IOError("No se pudo abrir la cámara.")
              # Establecer dimensiones deseadas
              if args.width > 0:
                  self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
              if args.height > 0:
                  self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
              # Configuración adicional para mejorar el rendimiento
              self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Tamaño mínimo de buffer
              # Leer dimensiones reales
              self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
              self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
              self.fps = args.fps  # Usamos el fps indicado (para el bucle de captura)
              print(f"Cámara inicializada: {self.width}x{self.height} @ {self.fps} FPS")
              print(f"Payload/frag UDP: {self.effective_video_payload_size} bytes")
              self.capture_enabled = True
          except Exception as e:
              print(f"Error al inicializar la cámara: {e}. Captura y envío deshabilitados.")
              if self.cap:
                  self.cap.release()
              self.cap = None
              self.capture_enabled = False
              self.width = 0
              self.height = 0
              self.fps = 0
      else:
          print("Flag --show_video no detectado. La cámara no se inicializará; solo se habilitará la recepción de vídeo.")

      self.running = True

 def receive_video(self):
     # Intentamos recibir múltiples paquetes en un ciclo para mejorar la eficiencia
     packets_processed = 0
     max_packets_per_cycle = 20  # Limitamos para no bloquear demasiado tiempo
     
     try:
         rlist, _, _ = select.select([self.video_sock], [], [], 0.001)  # Reducimos el timeout
         if not rlist:
             return
         
         while packets_processed < max_packets_per_cycle:
             try:
                 packet, addr = self.video_sock.recvfrom(getattr(self, 'MAX_PAYLOAD_BYTES', 32768))
                 packets_processed += 1
                 
                 if len(packet) < self.header_size:
                     continue
                 
                 header = packet[:self.header_size]
                 payload = packet[self.header_size:]
                 
                 try:
                     frame_id, total_frags, frag_idx, remote_width, remote_height = struct.unpack(self._header_format, header)
                 except struct.error:
                     continue
                 
                 with self.recv_frames_lock:
                     if frame_id not in self.recv_frames:
                         if total_frags <= 0 or total_frags > 5000:
                             continue
                         self.recv_frames[frame_id] = {"fragments": [None] * total_frags,
                                                       "received_count": 0,
                                                       "total": total_frags,
                                                       "timestamp": time.time(),
                                                       "width": remote_width,
                                                       "height": remote_height}
                     
                     entry = self.recv_frames.get(frame_id)
                     if not entry or entry["total"] != total_frags or frag_idx >= entry["total"] or entry["fragments"][frag_idx] is not None:
                         continue
                     
                     entry["fragments"][frag_idx] = payload
                     entry["received_count"] += 1
                     
                     if entry["received_count"] == entry["total"]:
                         if None not in entry["fragments"]:
                             frame_data = b"".join(entry["fragments"])
                             expected_size = entry["width"] * entry["height"] * 3
                             if len(frame_data) == expected_size:
                                 try:
                                     frame = np.frombuffer(frame_data, dtype=np.uint8)
                                     frame = frame.reshape((entry["height"], entry["width"], 3))
                                     # Evitamos la copia si no es necesaria
                                     with self.latest_received_frame_lock:
                                         self.latest_received_frame = frame
                                 except Exception:
                                     pass
                         del self.recv_frames[frame_id]
                 
                 # Actualizar contadores en modo verbose
                 if hasattr(self, 'video_received_bytes_count'):
                     self.video_received_bytes_count += len(packet)
                     self.video_received_messages_count += 1
                     
             except socket.error:
                 # Socket vacío, salimos del bucle
                 break
             except Exception:
                 break
     except:
         pass

 def clean_old_frames(self):
     if not self.recv_frames:
         return
     with self.recv_frames_lock:
         now = time.time()
         timeout = 0.5  # Reducimos el timeout para frames incompletos
         remove_ids = []
         for fid, data in self.recv_frames.items():
             if now - data.get("timestamp", now) > timeout:
                 fragments = data["fragments"]
                 total = data["total"]
                 expected_size = data["width"] * data["height"] * 3
                 
                 # Procesamiento más eficiente de fragmentos faltantes
                 has_missing = False
                 for i in range(total):
                     if fragments[i] is None:
                         has_missing = True
                         if i < total - 1:
                             fragments[i] = bytes(self.effective_video_payload_size)  # cadena de ceros
                         else:
                             last_frag_size = expected_size - self.effective_video_payload_size * (total - 1)
                             fragments[i] = bytes(last_frag_size)  # cadena de ceros
                 
                 # Solo reconstruimos si hay fragmentos faltantes
                 if has_missing:
                     frame_data = b"".join(fragments)
                     if len(frame_data) == expected_size:
                         try:
                             frame = np.frombuffer(frame_data, dtype=np.uint8)
                             frame = frame.reshape((data["height"], data["width"], 3))
                             with self.latest_received_frame_lock:
                                 self.latest_received_frame = frame
                         except:
                             pass
                 remove_ids.append(fid)
                 
         # Eliminar frames procesados
         for fid in remove_ids:
             try:
                 del self.recv_frames[fid]
             except KeyError:
                 pass

 def video_loop(self):
     """
     Combina captura, envío, recepción y visualización de video en un solo hilo.
     Sigue el patrón: capturar -> fragmentar -> enviar -> recibir -> componer -> mostrar
     """
     period = 1.0 / max(1, self.fps) if self.capture_enabled else 0.033
     next_time = time.perf_counter()
     window_title = "Video"
     show_interval = 1.0 / 30.0  # Limitar mostrar a 30 FPS máximo
     
     # Prebuildear estructuras de datos para reducir tiempo de procesamiento
     _send_buffer = bytearray(self.max_payload_possible + self.header_size)
     _framing_data_stored = {}
     
     while self.running:
         current_time = time.perf_counter()
         
         # Procesar recepción con mayor prioridad y frecuencia
         self.receive_video()
         
         # Controlar la frecuencia de captura
         if current_time >= next_time:
             # 1. Capturar frame
             frame_to_send = None
             if self.capture_enabled and self.cap:
                 ret, frame = self.cap.read()
                 if ret:
                     if (frame.shape[1], frame.shape[0]) != (self.width, self.height):
                         try:
                             frame = cv2.resize(frame, (self.width, self.height), 
                                              interpolation=cv2.INTER_NEAREST)  # INTER_NEAREST es más rápido
                         except cv2.error:
                             pass
                     else:
                         # No es necesario redimensionar, usamos el frame directamente
                         frame_to_send = frame
                         # Almacenamos sin copiar para ahorrar memoria
                         self.latest_captured_frame = frame
             
             # 2. Fragmentar y enviar
             if frame_to_send is not None:
                 data = frame_to_send.tobytes()  # Convertimos a bytes solo una vez
                 total_len = len(data)
                 if total_len > 0:
                     # Calculamos información de fragmentación una vez
                     total_frags = math.ceil(total_len / self.effective_video_payload_size)
                     if total_frags > 0:
                         # Preparamos el header base (solo cambia frag_idx)
                         base_header = struct.pack("!IHH", 
                                                  self.frame_id_counter,
                                                  total_frags, 
                                                  0)  # frag_idx se actualizará
                         dim_header = struct.pack("!HH", self.width, self.height)
                         
                         # Envío en bloque con menos interrupciones
                         for frag_idx in range(total_frags):
                             start = frag_idx * self.effective_video_payload_size
                             end = min(start + self.effective_video_payload_size, total_len)
                             payload = data[start:end]
                             
                             # Construimos el header para este fragmento
                             header = base_header[:6] + struct.pack("!H", frag_idx) + dim_header
                             
                             # Construimos el paquete completo
                             packet = header + payload
                             
                             try:
                                 # Enviar sin copias adicionales
                                 self.video_sock.sendto(packet, self.video_addr)
                                 # Actualizamos contadores si estamos en modo verbose
                                 if hasattr(self, 'video_sent_bytes_count'):
                                     self.video_sent_bytes_count += len(packet)
                                     self.video_sent_messages_count += 1
                             except:
                                 pass
                         
                         self.frame_id_counter += 1
             
             # Actualizamos el tiempo para la siguiente captura
             next_time = current_time + period
         
         # 3. Procesamiento periódico de frames antiguos
         now = time.time()
         if now >= self.next_cleanup_time:
             self.clean_old_frames()
             self.next_cleanup_time = now + 0.5  # Cada 0.5 segundos es suficiente
         
         # 4. Mostrar frame (limitado por frecuencia para no sobrecargar)
         if args.show_video and now - self.last_show_time >= show_interval:
             self.last_show_time = now
             frame_to_display = None
             
             # Primero intentamos mostrar el frame recibido
             with self.latest_received_frame_lock:
                 if self.latest_received_frame is not None:
                     frame_to_display = self.latest_received_frame  # Usamos sin copiar
             
             # Si no hay frame recibido, usamos el frame capturado
             if frame_to_display is None and self.capture_enabled:
                 with self.latest_captured_frame_lock:
                     if self.latest_captured_frame is not None:
                         frame_to_display = self.latest_captured_frame  # Usamos sin copiar
             
             if frame_to_display is not None:
                 try:
                     cv2.imshow(window_title, frame_to_display)
                 except:
                     pass
             
             key = cv2.waitKey(1)
             if key & 0xFF == ord('q'):
                 print("Tecla 'q' presionada, deteniendo...")
                 self.running = False
         
         # 5. Controlamos el uso de CPU para no saturar
         sleep_time = 0.001
         time.sleep(sleep_time)

 def run(self):
     print("Iniciando video con bucle unificado...")
     
     # Solo necesitamos un hilo para todas las operaciones de video
     t_unified = threading.Thread(target=self.video_loop, daemon=True, name="UnifiedVideoThread")
     t_unified.start()
     
     try:
         super().run() 
     except KeyboardInterrupt:
         print("Interrupción por teclado detectada.")
     finally:
         print("Deteniendo aplicación de video...")
         self.running = False
         if self.cap and self.cap.isOpened():
             self.cap.release()
         cv2.destroyAllWindows()
         if self.video_sock:
             self.video_sock.close()
         print("Aplicación de video detenida.")


class Minimal_Video__verbose(Minimal_Video, minimal.Minimal__verbose):
 def __init__(self):
     super().__init__()
     try:
         minimal.Minimal__verbose.__init__(self)
         print(f"Verbose Mode: stats cycle = {self.seconds_per_cycle}s")
     except AttributeError:
         print("Error: No se pudo inicializar minimal.Minimal__verbose. Las estadísticas no funcionarán.")
     # Inicializamos los contadores de video para la versión verbose
     self.video_sent_bytes_count = 0
     self.video_sent_messages_count = 0
     self.video_received_bytes_count = 0
     self.video_received_messages_count = 0

 def moving_average(self, average, new_sample, number_of_samples):
     return average + (new_sample - average) / number_of_samples

 def loop_cycle_feedback(self):
     header1 = f"{'':8s} {'audio':>16s} {'video':>16s} {'audio':>16s} {'video':>16s} {'Global':>8s}"
     header2 = f"{'cycle':8s} {'sent':>8s} {'recv':>8s} {'sent':>8s} {'recv':>8s} {'KBPS':>8s} {'KBPS':>8s} {'KBPS':>8s} {'KBPS':>8s} {'%CPU':>4s} {'%CPU':>4s}"
     print(header1)
     print(header2)
     print("=" * 80)
     cycle = 1
     self.old_time = time.time()
     self.old_CPU_time = psutil.Process().cpu_times()[0]
     # Inicializamos las variables de promedio,
     # en caso de que no se hayan inicializado en __init__
     self.average_CPU_usage = 0
     self.average_sent_KBPS = 0
     self.average_received_KBPS = 0


     while self.running:
         now = time.time()
         elapsed = max(now - self.old_time, 0.001)
         elapsed_CPU_time = psutil.Process().cpu_times()[0] - self.old_CPU_time
         self.CPU_usage = 100 * elapsed_CPU_time / elapsed
         self.global_CPU_usage = psutil.cpu_percent(interval=None)
      
         audio_sent_kbps = int(self.sent_bytes_count * 8 / 1000 / elapsed)
         audio_recv_kbps = int(self.received_bytes_count * 8 / 1000 / elapsed)
         video_sent_kbps = int(self.video_sent_bytes_count * 8 / 1000 / elapsed)
         video_recv_kbps = int(self.video_received_bytes_count * 8 / 1000 / elapsed)
      
         # Actualizamos los promedios usando la función moving_average:
         self.average_CPU_usage = self.moving_average(self.average_CPU_usage, self.CPU_usage, cycle)
         self.average_sent_KBPS = self.moving_average(self.average_sent_KBPS, video_sent_kbps, cycle)
         self.average_received_KBPS = self.moving_average(self.average_received_KBPS, video_recv_kbps, cycle)
      
         print(f"{cycle:8d} {self.sent_messages_count:8d} {self.received_messages_count:8d} "
             f"{self.video_sent_messages_count:8d} {self.video_received_messages_count:8d} "
             f"{audio_sent_kbps:8d} {audio_recv_kbps:8d} {video_sent_kbps:8d} {video_recv_kbps:8d} "
             f"{int(self.CPU_usage):4d} {int(self.global_CPU_usage):4d}")
      
         # Reiniciamos los contadores para el siguiente ciclo:
         self.sent_bytes_count = 0
         self.received_bytes_count = 0
         self.sent_messages_count = 0
         self.received_messages_count = 0
         self.video_sent_bytes_count = 0
         self.video_received_bytes_count = 0
         self.video_sent_messages_count = 0
         self.video_received_messages_count = 0
      
         cycle += 1
         self.old_time = now
         self.old_CPU_time = psutil.Process().cpu_times()[0]
         time.sleep(1)

 def print_final_averages(self):
     print("\n" * 4)
     print(f"CPU usage average = {self.average_CPU_usage:.2f} %")
     print(f"Payload sent average = {self.average_sent_KBPS:.2f} kilo bits per second")
     print(f"Payload received average = {self.average_received_KBPS:.2f} kilo bits per second")

 def run(self):
     if not hasattr(self, 'loop_cycle_feedback'):
         print("Advertencia: El bucle de feedback de estadísticas no está disponible. Ejecutando sin estadísticas.")
         super().run()
         return
     cycle_feedback_thread = threading.Thread(target=self.loop_cycle_feedback, daemon=True, name="FeedbackThread")
     self.print_header()
     cycle_feedback_thread.start()
     super().run()


if __name__ == "__main__":
 try:
     import argcomplete
     argcomplete.autocomplete(minimal.parser)
 except Exception:
     pass
 args = minimal.parser.parse_args()
 if not hasattr(args, 'destination_address') or not args.destination_address:
     args.destination_address = "localhost"

 verbose_enabled = (getattr(args, 'show_stats', False) or
                    getattr(args, 'show_samples', False) or
                    getattr(args, 'show_spectrum', False))
 verbose_class_exists = hasattr(minimal, 'Minimal__verbose')

 if verbose_enabled and verbose_class_exists:
     print("Iniciando en modo Verbose...")
     intercom_app = Minimal_Video__verbose()
 elif verbose_enabled and not verbose_class_exists:
     print("Advertencia: Modo verbose activado pero minimal.Minimal__verbose no encontrado. Ejecutando sin estadísticas.")
     intercom_app = Minimal_Video()
 else:
     intercom_app = Minimal_Video()

 try:
     intercom_app.run()
 except KeyboardInterrupt:
     pass
 except Exception as e:
     print(f"\nError inesperado: {e}")
     import traceback
     traceback.print_exc()
 finally:
     if hasattr(intercom_app, 'print_final_averages') and callable(intercom_app.print_final_averages):
         # Se espera un breve retardo para que se acumulen los últimos datos
         time.sleep(0.2)
         intercom_app.print_final_averages()
     print("Programa terminado.")
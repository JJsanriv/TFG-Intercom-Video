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
       if args is None:
            args = minimal.parser.parse_args()
       minimal.args = args
       # No se realiza diferenciación entre local y remoto; se usa la dirección indicada
       super().__init__()
       # Se crea el socket para video (UDP)
       self.video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
       self.video_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
       try:
           self.video_sock.bind(("0.0.0.0", args.video_port))
       except OSError as e:
           print(f"Error bind socket video: {e}")
           raise
       self.video_addr = (args.destination_address, args.video_port)
       self.video_sock.setblocking(False)
       self.recv_frames = {}
       self.recv_frames_lock = threading.Lock()
       self.latest_received_frame = None
       self.latest_received_frame_lock = threading.Lock()
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
       # Se elimina la cola y el hilo exclusivo para envío
       self.frame_id_counter = 0
       try:
           self.cap = cv2.VideoCapture(0)
           if not self.cap.isOpened():
               raise IOError("No se pudo abrir la cámara.")
           if args.width > 0:
               self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
           if args.height > 0:
               self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
           self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
           self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
           self.fps = args.fps
           print(f"Cámara: {self.width}x{self.height} @ {self.fps} FPS")
           print(f"Payload/frag UDP: {self.effective_video_payload_size} bytes")
           self.capture_enabled = True
           self.latest_captured_frame = None
           self.latest_captured_frame_lock = threading.Lock()
       except Exception as e:
           print(f"Error cámara: {e}. Captura deshabilitada.")
           self.cap = None
           self.capture_enabled = False
           self.width = 0
           self.height = 0
           self.fps = 0
       self.running = True


   def capture_and_send_video_loop(self):
       """
       Captura el frame, lo fragmenta y envía cada fragmento inmediatamente.
       Se hace de forma directa sin usar colas.
       """
       if not self.capture_enabled:
           return
       period = 1.0 / max(1, self.fps)
       next_capture_time = time.perf_counter()
       target_dim = (self.width, self.height)
       while self.running:
           capture_start_time = time.perf_counter()
           sleep_time = next_capture_time - capture_start_time
           time.sleep(sleep_time if sleep_time > 0.001 else 0.001)
           ret, frame = self.cap.read()
           if not ret:
               next_capture_time = time.perf_counter() + period
               continue
           next_capture_time += period
           if (frame.shape[1], frame.shape[0]) != target_dim:
               try:
                   frame = cv2.resize(frame, target_dim, interpolation=cv2.INTER_LINEAR)
               except cv2.error:
                   continue
           with self.latest_captured_frame_lock:
               self.latest_captured_frame = frame.copy()
           # Fragmentación y envío inmediato
           data = frame.tobytes()
           total_len = len(data)
           if total_len == 0:
               continue
           total_frags = math.ceil(total_len / self.effective_video_payload_size)
           if total_frags == 0:
               continue
           frag_idx = 0
           for start in range(0, total_len, self.effective_video_payload_size):
               if not self.running:
                   break
               end = min(start + self.effective_video_payload_size, total_len)
               payload = data[start:end]
               header = struct.pack(self._header_format, self.frame_id_counter,
                                    total_frags, frag_idx, self.width, self.height)
               packet = header + payload
               try:
                   self.video_sock.sendto(packet, self.video_addr)
               except socket.error:
                   time.sleep(0.001)
                   pass
               except Exception as e:
                   print(f"Error send frag {frag_idx}: {e}")
                   pass
               # Se insertó un retardo mínimo para evitar saturar el socket
               time.sleep(0.0001)
               frag_idx += 1
           if self.running:
               self.frame_id_counter += 1
           time.sleep(0.001)


   def receive_video(self):
       try:
           rlist, _, _ = select.select([self.video_sock], [], [], 0.005)
           if not rlist:
               return
           packet, addr = self.video_sock.recvfrom(getattr(self, 'MAX_PAYLOAD_BYTES', 32768))
           if len(packet) < self.header_size:
               return
           header = packet[:self.header_size]
           payload = packet[self.header_size:]
           try:
               frame_id, total_frags, frag_idx, remote_width, remote_height = struct.unpack(self._header_format, header)
           except struct.error:
               return
           with self.recv_frames_lock:
               if frame_id not in self.recv_frames:
                   if total_frags <= 0 or total_frags > 5000:
                       print(f"Advertencia: total_frags inválido ({total_frags}) de {addr}")
                       return
                   self.recv_frames[frame_id] = {"fragments": [None] * total_frags,
                                                  "received_count": 0,
                                                  "total": total_frags,
                                                  "timestamp": time.time(),
                                                  "width": remote_width,
                                                  "height": remote_height}
               entry = self.recv_frames.get(frame_id)
               if not entry or entry["total"] != total_frags or frag_idx >= entry["total"] or entry["fragments"][frag_idx] is not None:
                   return
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
                               with self.latest_received_frame_lock:
                                   self.latest_received_frame = frame
                           except Exception as e:
                               print(f"Error procesando frame completo: {e}")
                   del self.recv_frames[frame_id]
       except socket.error:
           pass
       except Exception as e:
           print(f"Error inesperado en receive_video: {e}")


   def receive_video_loop(self):
       while self.running:
           self.receive_video()
           time.sleep(0.001)
           if time.time() % 1.0 < 0.02:
               self.clean_old_frames()


   def clean_old_frames(self):
       if not self.recv_frames:
           return
       with self.recv_frames_lock:
           now = time.time()
           timeout = 1.5  # tiempo en segundos para considerar un frame incompleto como expirado
           remove_ids = []
           for fid, data in self.recv_frames.items():
               if now - data.get("timestamp", now) > timeout:
                   fragments = data["fragments"]
                   total = data["total"]
                   # El tamaño esperado del frame en bytes
                   expected_size = data["width"] * data["height"] * 3
                   # Rellenamos cada fragmento faltante con ceros
                   for i in range(total):
                       if fragments[i] is None:
                           if i < total - 1:
                               frag_len = self.effective_video_payload_size
                           else:
                               frag_len = expected_size - self.effective_video_payload_size * (total - 1)
                           fragments[i] = bytes(frag_len)  # cadena de ceros
                   # Reconstruimos el frame completo
                   frame_data = b"".join(fragments)
                   if len(frame_data) == expected_size:
                       try:
                           frame = np.frombuffer(frame_data, dtype=np.uint8)
                           frame = frame.reshape((data["height"], data["width"], 3))
                           with self.latest_received_frame_lock:
                               self.latest_received_frame = frame
                       except Exception as e:
                           print(f"Error procesando frame en clean_old_frames: {e}")
                   remove_ids.append(fid)
           for fid in remove_ids:
               try:
                   del self.recv_frames[fid]
               except KeyError:
                   pass




   def display_video_loop(self):
       last_display_time = time.time()
       display_fps_target = 30
       min_display_interval = 1.0 / display_fps_target
       window_title = "Video"
       while self.running:
           now = time.time()
           wait_time = min_display_interval - (now - last_display_time)
           time.sleep(wait_time if wait_time > 0.001 else 0.001)
           frame_to_show = None
           # Se intenta primero mostrar el frame recibido
           with self.latest_received_frame_lock:
               if self.latest_received_frame is not None:
                   frame_to_show = self.latest_received_frame.copy()
           # Si no hay frame recibido, se usa el frame capturado
           if frame_to_show is None and self.capture_enabled:
               with self.latest_captured_frame_lock:
                   if self.latest_captured_frame is not None:
                       frame_to_show = self.latest_captured_frame.copy()
           if frame_to_show is not None:
               try:
                   cv2.imshow(window_title, frame_to_show)
                   last_display_time = time.time()
               except cv2.error:
                   pass
           key = cv2.waitKey(1)
           if key & 0xFF == ord('q'):
               print("Tecla 'q' presionada, deteniendo...")
               self.running = False
               break


       try:
           cv2.destroyWindow(window_title)
       except cv2.error:
           pass


   def run(self):
       print("Iniciando video sin cola (envío directo)...")
       threads = []
       if self.capture_enabled:
           t_capture = threading.Thread(target=self.capture_and_send_video_loop, daemon=True, name="CaptureAndSendThread")
           threads.append(t_capture)
       if args.show_video:
           t_display = threading.Thread(target=self.display_video_loop, daemon=True, name="DisplayThread")
           threads.append(t_display)
       t_recv = threading.Thread(target=self.receive_video_loop, daemon=True, name="ReceiveThread")
       threads.append(t_recv)
       for t in threads:
           t.start()
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


   # Se sobreescribe el método de envío para incorporar estadísticas
   def capture_and_send_video_loop(self):
       if not self.capture_enabled:
           return
       period = 1.0 / max(1, self.fps)
       next_capture_time = time.perf_counter()
       target_dim = (self.width, self.height)
       while self.running:
           capture_start_time = time.time()
           sleep_time = next_capture_time - capture_start_time
           time.sleep(sleep_time if sleep_time > 0.001 else 0.001)
           ret, frame = self.cap.read()
           if not ret:
               next_capture_time = time.perf_counter() + period
               continue
           next_capture_time += period
           if (frame.shape[1], frame.shape[0]) != target_dim:
               try:
                   frame = cv2.resize(frame, target_dim, interpolation=cv2.INTER_LINEAR)
               except cv2.error:
                   continue
           with self.latest_captured_frame_lock:
               self.latest_captured_frame = frame.copy()
           data = frame.tobytes()
           total_len = len(data)
           if total_len == 0:
               continue
           total_frags = math.ceil(total_len / self.effective_video_payload_size)
           if total_frags == 0:
               continue
           frag_idx = 0
           for start in range(0, total_len, self.effective_video_payload_size):
               if not self.running:
                   break
               end = min(start + self.effective_video_payload_size, total_len)
               payload = data[start:end]
               header = struct.pack(self._header_format, self.frame_id_counter,
                                    total_frags, frag_idx, self.width, self.height)
               packet = header + payload
               try:
                   self.video_sock.sendto(packet, self.video_addr)
                   self.video_sent_bytes_count += len(packet)
                   self.video_sent_messages_count += 1
               except socket.error:
                   time.sleep(0.001)
                   pass
               except Exception as e:
                   print(f"Error send frag {frag_idx}: {e}")
                   pass
               time.sleep(0.0001)
               frag_idx += 1
           if self.running:
               self.frame_id_counter += 1
           # Aquí se elimina el mensaje "DEBUG: Frame…"
           time.sleep(0.001)


   # Se sobreescribe también el método de recepción para actualizar contadores
   def receive_video(self):
       try:
           rlist, _, _ = select.select([self.video_sock], [], [], 0.005)
           if not rlist:
               return
           packet, addr = self.video_sock.recvfrom(getattr(self, 'MAX_PAYLOAD_BYTES', 32768))
           self.video_received_bytes_count += len(packet)
           self.video_received_messages_count += 1
           if len(packet) < self.header_size:
               return
           header = packet[:self.header_size]
           payload = packet[self.header_size:]
           try:
               frame_id, total_frags, frag_idx, remote_width, remote_height = struct.unpack(self._header_format, header)
           except struct.error:
               return
           with self.recv_frames_lock:
               if frame_id not in self.recv_frames:
                   if total_frags <= 0 or total_frags > 5000:
                       print(f"Advertencia: total_frags inválido ({total_frags}) de {addr}")
                       return
                   self.recv_frames[frame_id] = {"fragments": [None] * total_frags,
                                                  "received_count": 0,
                                                  "total": total_frags,
                                                  "timestamp": time.time(),
                                                  "width": remote_width,
                                                  "height": remote_height}
               entry = self.recv_frames.get(frame_id)
               if not entry or entry["total"] != total_frags or frag_idx >= entry["total"] or entry["fragments"][frag_idx] is not None:
                   return
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
                               with self.latest_received_frame_lock:
                                   self.latest_received_frame = frame
                           except Exception as e:
                               print(f"Error procesando frame completo: {e}")
                   del self.recv_frames[frame_id]
       except socket.error:
           pass
       except Exception as e:
           print(f"Error inesperado en receive_video: {e}")


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








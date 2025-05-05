#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""
Minimal_Video: Extiende de minimal.py para agregar transmisión/visualización de video sin
compresión/encodificación, usando raw data. Incluye opción verbose (--show_stats, --show_samples y --show_spectrum).
- Se transmite video full‐duplex vía UDP sin usar colas, es decir, se envía el frame directamente.
- El flag --show_video habilita la visualización.

Utiliza un socket UDP para transmisión y fragmenta los frames.
Header (big-endian): FragIdx(H) - Solo se transmite la posición del fragmento

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
        self.latest_received_frame = None
        self.latest_received_frame_lock = threading.Lock()
      
        # Variables para el nuevo protocolo simplificado
        self.current_frame_fragments = None
        self.fragments_received = 0

        # Configuración para fragmentación simplificada de los frames
        self._header_format = "!H"  # Solo FragIdx (unsigned short)
        self.header_size = 2  # Ahora solo 2 bytes para FragIdx
        self.effective_video_payload_size = args.video_payload_size
        self.max_payload_possible = self.effective_video_payload_size - self.header_size
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
      
        # Variables para el protocolo simplificado
        self.expected_frame_size = 0  # Se calculará cuando se inicialice la cámara
        self.total_frags = 0  # Se calculará cuando se inicialice la cámara
      
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
              
                # Cálculos para el nuevo protocolo simplificado
                self.expected_frame_size = self.width * self.height * 3  # RGB
                self.total_frags = math.ceil(self.expected_frame_size / self.effective_video_payload_size)
                self.current_frame_fragments = [None] * self.total_frags
              
                print(f"Cámara inicializada: {self.width}x{self.height} @ {self.fps} FPS")
                print(f"Payload/frag UDP: {self.effective_video_payload_size} bytes")
                print(f"Frame esperado: {self.expected_frame_size} bytes, {self.total_frags} fragmentos")
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
            # Para la recepción, configuramos los tamaños basados en los argumentos
            self.width = args.width
            self.height = args.height
            self.expected_frame_size = self.width * self.height * 3  # RGB
            self.total_frags = math.ceil(self.expected_frame_size / self.effective_video_payload_size)
            self.current_frame_fragments = [None] * self.total_frags

        self.running = True

    def receive_video(self):
        """
        Versión simplificada que solo utiliza el índice de fragmento.
        """
        packets_processed = 0
        max_packets_per_cycle = 20
     
        try:
            # Se incrementa un poco el timeout para evitar busy waiting extremo
            rlist, _, _ = select.select([self.video_sock], [], [], 0.005)
            if not rlist:
                return
         
            while packets_processed < max_packets_per_cycle:
                try:
                    packet, addr = self.video_sock.recvfrom(self.effective_video_payload_size + self.header_size)
                    packets_processed += 1
                 
                    if len(packet) < self.header_size:
                        continue
                 
                    header = packet[:self.header_size]
                    payload = packet[self.header_size:]
                 
                    try:
                        frag_idx, = struct.unpack(self._header_format, header)
                    except struct.error:
                        continue
                 
                    # Validar índice de fragmento
                    if 0 <= frag_idx < self.total_frags and self.current_frame_fragments[frag_idx] is None:
                        self.current_frame_fragments[frag_idx] = payload
                        self.fragments_received += 1
                     
                        # Si recibimos todos los fragmentos, reconstruimos el frame
                        if self.fragments_received == self.total_frags:
                            frame_data = b"".join(self.current_frame_fragments)
                            if len(frame_data) == self.expected_frame_size:
                                try:
                                    frame = np.frombuffer(frame_data, dtype=np.uint8)
                                    frame = frame.reshape((self.height, self.width, 3))
                                    with self.latest_received_frame_lock:
                                        self.latest_received_frame = frame
                                except Exception:
                                    pass
                         
                            # Reiniciamos para el próximo frame
                            self.current_frame_fragments = [None] * self.total_frags
                            self.fragments_received = 0
                 
                    # Actualizar contadores en modo verbose
                    if hasattr(self, 'video_received_bytes_count'):
                        self.video_received_bytes_count += len(packet)
                        self.video_received_messages_count += 1
                     
                except socket.error:
                    # Socket vacío, salimos del bucle
                    break
                except Exception as e:
                    print(f"Error en recepción: {e}")
                    break
        except Exception as e:
            print(f"Excepción en receive_video: {e}")

        def check_incomplete_frame(self):
            """
            Si el frame está incompleto, simplemente rellena latest_received_frame con ceros,
            """
            if self.fragments_received > 0 and self.fragments_received < self.total_frags:
                # Frame incompleto: rellena con ceros.
                blank_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                with self.latest_received_frame_lock:
                    self.latest_received_frame = blank_frame
                # Reinicia para el próximo frame
                self.current_frame_fragments = [None] * self.total_frags
                self.fragments_received = 0

    def video_loop(self):
        window_title = "Video"
        target_period = 1.0 / self.fps if self.fps > 0 else 1.0 / 30.0

        while self.running:
            cycle_start = time.monotonic()

            # 1. Procesar recepción de video
            self.receive_video()

            # 2. Capturar y enviar frame, si corresponde
            if self.capture_enabled:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

                self.latest_captured_frame = frame
                data = frame.tobytes()

                for frag_idx in range(self.total_frags):
                    start = frag_idx * self.effective_video_payload_size
                    end = min(start + self.effective_video_payload_size, len(data))
                    payload = data[start:end]
                    header = struct.pack(self._header_format, frag_idx)
                    packet = header + payload
                    sent = False
                    while not sent and self.running:
                        try:
                            self.video_sock.sendto(packet, self.video_addr)
                            sent = True
                        except BlockingIOError:
                            time.sleep(0.002)

            # 3. Mostrar frame
            if args.show_video:
                frame_to_display = None
                with self.latest_received_frame_lock:
                    if self.latest_received_frame is not None:
                        frame_to_display = self.latest_received_frame
                if frame_to_display is None and self.capture_enabled:
                    frame_to_display = self.latest_captured_frame

                if frame_to_display is not None:
                    try:
                        cv2.imshow(window_title, frame_to_display)
                    except Exception as e:
                        print(f"Error al mostrar frame: {e}")
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    print("Tecla 'q' presionada, deteniendo...")
                    self.running = False

            # 4. Procesamiento periódico de frames incompletos
            if time.monotonic() >= self.next_cleanup_time:
                self.check_incomplete_frame()
                self.next_cleanup_time = time.monotonic() + 0.5

            # 5. Sincronización de ciclo
            elapsed = time.monotonic() - cycle_start
            to_sleep = max(0, target_period - elapsed)
            if to_sleep > 0:
                time.sleep(to_sleep)

    def run(self):
        print("Iniciando video con bucle unificado y protocolo simplificado...")
     
        # Solo se utiliza un hilo para todas las operaciones de video
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
        # Inicializamos contadores para la versión verbose
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
        # Inicializamos promedios
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
         
            # Actualización de promedios
            self.average_CPU_usage = self.moving_average(self.average_CPU_usage, self.CPU_usage, cycle)
            self.average_sent_KBPS = self.moving_average(self.average_sent_KBPS, video_sent_kbps, cycle)
            self.average_received_KBPS = self.moving_average(self.average_received_KBPS, video_recv_kbps, cycle)
         
            print(f"{cycle:8d} {self.sent_messages_count:8d} {self.received_messages_count:8d} "
                  f"{self.video_sent_messages_count:8d} {self.video_received_messages_count:8d} "
                  f"{audio_sent_kbps:8d} {audio_recv_kbps:8d} {video_sent_kbps:8d} {video_recv_kbps:8d} "
                  f"{int(self.CPU_usage):4d} {int(self.global_CPU_usage):4d}")
         
            # Reiniciar contadores
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

    def video_loop(self):
        window_title = "Video"
        target_period = 1.0 / self.fps if self.fps > 0 else 1.0 / 30.0

        while self.running:
            cycle_start = time.monotonic()

            # 1. Procesar recepción de video
            self.receive_video()

            # 2. Capturar y enviar frame, si corresponde
            if self.capture_enabled:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

                self.latest_captured_frame = frame
                data = frame.tobytes()

                for frag_idx in range(self.total_frags):
                    start = frag_idx * self.effective_video_payload_size
                    end = min(start + self.effective_video_payload_size, len(data))
                    payload = data[start:end]
                    header = struct.pack(self._header_format, frag_idx)
                    packet = header + payload
                    sent = False
                    while not sent and self.running:
                        try:
                            self.video_sock.sendto(packet, self.video_addr)
                            # ACTUALIZA CONTADORES VERBOSE
                            self.video_sent_bytes_count += len(packet)
                            self.video_sent_messages_count += 1
                            sent = True
                        except BlockingIOError:
                            time.sleep(0.002)

            # 3. Mostrar frame
            if args.show_video:
                frame_to_display = None
                with self.latest_received_frame_lock:
                    if self.latest_received_frame is not None:
                        frame_to_display = self.latest_received_frame
                if frame_to_display is None and self.capture_enabled:
                    frame_to_display = self.latest_captured_frame

                if frame_to_display is not None:
                    try:
                        cv2.imshow(window_title, frame_to_display)
                    except Exception as e:
                        print(f"Error al mostrar frame: {e}")
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    print("Tecla 'q' presionada, deteniendo...")
                    self.running = False

            # 4. Procesamiento periódico de frames incompletos
            if time.monotonic() >= self.next_cleanup_time:
                self.check_incomplete_frame()
                self.next_cleanup_time = time.monotonic() + 0.5

            # 5. Sincronización de ciclo
            elapsed = time.monotonic() - cycle_start
            to_sleep = max(0, target_period - elapsed)
            if to_sleep > 0:
                time.sleep(to_sleep)

    def run(self):
        if not hasattr(self, 'loop_cycle_feedback'):
            print("Advertencia: El bucle de feedback de estadísticas no está disponible. Ejecutando sin estadísticas.")
            super().run()
            return
        cycle_feedback_thread = threading.Thread(target=self.loop_cycle_feedback, daemon=True, name="FeedbackThread")
        self.print_header()
        cycle_feedback_thread.start()
        print("Iniciando video con bucle unificado y protocolo simplificado (verbose)...")
        # Solo la versión verbose usa su propio video_loop
        t_unified = threading.Thread(target=self.video_loop, daemon=True, name="UnifiedVideoThread")
        t_unified.start()
        try:
            super(Minimal_Video, self).run()
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
            time.sleep(0.2)
            intercom_app.print_final_averages()
        print("Programa terminado.")

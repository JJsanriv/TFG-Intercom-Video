#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""
Minimal_Video: Extiende de minimal.py para agregar transmisión/visualización de video sin
compresión/encodificación, usando raw data. Incluye opción verbose (--show_stats, --show_samples y --show_spectrum).
- Se transmite video full‐duplex vía UDP sin usar colas, es decir, se envía el frame directamente.
- El flag --show_video habilita la visualización y la transmisión de video.
- Sin --show_video, se comporta exactamente como minimal.py (solo audio).

Utiliza un socket UDP para transmisión y fragmenta los frames.
Header (big-endian): FragIdx(H) - Solo se transmite la posición del fragmento

Nuevos parámetros:
--video_payload_size : Tamaño deseado (bytes) payload video/fragmento UDP (defecto 1400).
--width              : Ancho del video (defecto 320).
--height             : Alto del video (defecto 180).
--fps                : Frames por segundo video (defecto 30).
--show_video         : Habilita la visualización y transmisión del video (desactivado por defecto).
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


parser.add_argument("-v", "--video_payload_size", type=int, default=1400,
                    help="Tamaño deseado (bytes) payload video/fragmento UDP (defecto 1400).")
parser.add_argument("-w", "--width", type=int, default=320, help="Ancho video (defecto 320)")
parser.add_argument("-g", "--height", type=int, default=240, help="Alto video (defecto 240)")
parser.add_argument("-z", "--fps", type=int, default=30, help="Frames por segundo video (defecto 30)")
parser.add_argument("--show_video", action="store_true", default=False,
                    help="Habilita la visualización y transmisión del video (desactivado por defecto).")
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
        
        # Si no se activa el video, terminamos la inicialización aquí
        if not args.show_video:
            return
            
        # Configuración del socket de vídeo
        self.video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.video_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Aumentamos el buffer para mejorar rendimiento
        self.video_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8388608)
        self.video_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8388608)
        
        self.video_sock.setblocking(False) # No bloqueante
        try:
            self.video_sock.bind(("0.0.0.0", args.video_port))
        except OSError as e:
            print(f"Error bind socket video: {e}")
            raise
        self.video_addr = (args.destination_address, args.video_port)

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

        # Inicialización cámara
        self.cap = None
        self.width = 0
        self.height = 0
        self.fps = 0
        self.latest_captured_frame = None
      
        # Variables para el protocolo simplificado
        self.expected_frame_size = 0  # Se calculará cuando se inicialice la cámara
        self.total_frags = 0  # Se calculará cuando se inicialice la cámara

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
            if args.fps > 0:
                self.cap.set(cv2.CAP_PROP_FPS, args.fps)
            
            # Configuración adicional para mejorar el rendimiento
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Tamaño mínimo de buffer
            # Leer dimensiones reales
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) 
          
            # Cálculos para el nuevo protocolo simplificado
            self.expected_frame_size = self.width * self.height * 3  # RGB
            self.total_frags = math.ceil(self.expected_frame_size / self.effective_video_payload_size)
            
          
            print(f"Cámara inicializada: {self.width}x{self.height} @ {self.fps} FPS")
            print(f"Payload/frag UDP: {self.effective_video_payload_size} bytes")
            print(f"Frame esperado: {self.expected_frame_size} bytes, {self.total_frags} fragmentos")
            
            # Precalcular rangos de fragmentos y headers para mejorar rendimiento
            self.fragment_ranges = []
            self.fragment_headers = []
            for frag_idx in range(self.total_frags):
                start = frag_idx * self.effective_video_payload_size
                end = min(start + self.effective_video_payload_size, self.expected_frame_size)
                self.fragment_ranges.append((start, end))
                self.fragment_headers.append(struct.pack(self._header_format, frag_idx))
                
            # Inicializar frames
            self.remote_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self.received_remote_frame = False

            # Inicializa el buffer del frame a ceros (Precalculamos el tamaño)
            if not hasattr(self, 'temp_frame_buffer') or self.temp_frame_buffer.shape != (self.height, self.width, 3):
                self.temp_frame_buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
        except Exception as e:
            print(f"Error al inicializar la cámara: {e}. Deshabilitando video.")
            if self.cap:
                self.cap.release()
            self.cap = None

        self.running = True

    """
    def receive_video(self):

        # Inicializa el frame a negro antes de recibir fragmentos nuevos
        self.temp_frame_buffer.fill(0)

        # Usa select para no bloquear nunca
        rlist, _, _ = select.select([self.video_sock], [], [], 0)
        if not rlist:
            # No hay datos, seguimos
            self.remote_frame = self.temp_frame_buffer.copy()
            return

        while True:
            try:
                packet, addr = self.video_sock.recvfrom(self.effective_video_payload_size + self.header_size)
            except BlockingIOError:
                break  # No hay más datos disponibles
            except Exception as e:
                print("Error recibiendo paquete UDP de vídeo:", e)
                break

            # Procesa el paquete: extrae el fragmento y lo copia al buffer
            header = packet[:self.header_size]
            payload = packet[self.header_size:]
            try:
                frag_idx, = struct.unpack(self._header_format, header)
            except struct.error:
                continue

            if 0 <= frag_idx < self.total_frags:
                start = frag_idx * self.effective_video_payload_size
                end = min(start + len(payload), self.expected_frame_size)
                flat_frame = self.temp_frame_buffer.reshape(-1)
                flat_frame[start:end] = np.frombuffer(payload, dtype=np.uint8, count=(end-start))

        # Copia el frame temporal para mostrarlo
        self.remote_frame = self.temp_frame_buffer.copy()
    """

    def video_loop(self):
        
        while self.running:
                       
            # 1. Capturar frame
            _, frame = self.cap.read()
            data = frame.tobytes()

            # 2. Entrelazar envío y recepción de fragmentos
            self.temp_frame_buffer.fill(0)
            
            for frag_idx in range(self.total_frags):
                # ENVÍO: Enviar un fragmento
                start, end = self.fragment_ranges[frag_idx]
                payload = data[start:end]
                packet = self.fragment_headers[frag_idx] + payload
                
                try:
                    self.video_sock.sendto(packet, self.video_addr)
                except BlockingIOError:
                    # Si el buffer está lleno, esperamos un poco y continuamos
                    time.sleep(0.001)
                    continue
                
                # RECEPCIÓN: Intentar recibir un fragmento (sin bloquear)
                try:
                    rlist, _, _ = select.select([self.video_sock], [], [], 0)
                    if rlist:
                        packet, addr = self.video_sock.recvfrom(self.effective_video_payload_size + self.header_size)
                        
                        # Procesa el paquete recibido
                        header = packet[:self.header_size]
                        payload = packet[self.header_size:]
                        
                        try:
                            recv_frag_idx, = struct.unpack(self._header_format, header)
                            
                            start = recv_frag_idx * self.effective_video_payload_size
                            end = min(start + len(payload), self.expected_frame_size)
                            flat_frame = self.temp_frame_buffer.reshape(-1)
                            flat_frame[start:end] = np.frombuffer(payload, dtype=np.uint8, count=(end-start))
                        except struct.error:
                            pass
                except BlockingIOError:
                    pass
                except Exception:
                    pass
            
            # 3. Actualizar y mostrar el frame remoto
            self.remote_frame = self.temp_frame_buffer.copy()
            cv2.imshow("Video", self.remote_frame)
            cv2.waitKey(1)

    def run(self):
        # Si el video no está habilitado, comportarse como minimal.py
        if not args.show_video or self.cap is None:
            print("Video desactivado. Ejecutando solo la parte de audio.")
            super().run()
            return
        
        if self.cap is not None:
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
                if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
                    self.cap.release()
                cv2.destroyAllWindows()
                if hasattr(self, 'video_sock') and self.video_sock:
                    self.video_sock.close()
                print("Aplicación de video detenida.")


class Minimal_Video__verbose(Minimal_Video, minimal.Minimal__verbose):
    def __init__(self):
        super().__init__()
        
        # Si el video no está habilitado, no inicializar contadores de video
        if not args.show_video or not hasattr(self, 'cap') or self.cap is None:
            return
            
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
        
        # --- Implementación para control de tiempo ---
        self.total_number_of_sent_frames = 0
        self.frame_time = 1.0 / self.fps  # Segundos por frame
        
        # Si se especifica tiempo de lectura, calcular el tiempo de finalización
        self.end_time = None
        if args.reading_time:
            # En lugar de contar frames, establecemos un tiempo de finalización absoluto
            self.end_time = time.time() + float(args.reading_time)
            print(f"Programa terminará automáticamente después de {args.reading_time} segundos")
            print(f"Tiempo de finalización programado: {time.strftime('%H:%M:%S', time.localtime(self.end_time))}")
            
            # Configurar un evento para señalizar el final
            self.time_event = threading.Event()

    def loop_cycle_feedback(self):
        # Si el video no está habilitado, usar la versión de la clase padre
        if not args.show_video or not hasattr(self, 'cap') or self.cap is None:
            if hasattr(minimal.Minimal__verbose, 'loop_cycle_feedback'):
                return super(Minimal_Video, self).loop_cycle_feedback()
            return
            
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

        start_time = time.time()  # Para mostrar tiempo transcurrido
        
        # Ciclo principal mientras estemos dentro del tiempo límite o sin límite
        while self.running:
            now = time.time()
            
            # Verificar si hemos alcanzado el tiempo límite
            if self.end_time and now >= self.end_time:
                print(f"\nLímite de tiempo alcanzado: {args.reading_time} segundos")
                self.time_event.set()  # Señalizar que hemos terminado
                break
                
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
            
            # Mostrar información de tiempo si hay límite
            time_info = ""
            if self.end_time:
                elapsed_total = now - start_time
                remaining = max(0, self.end_time - now)
                progress = min(100, 100 * elapsed_total / args.reading_time)
                time_info = f" | {elapsed_total:.1f}s/{args.reading_time}s ({progress:.0f}%)"
         
            print(f"{cycle:8d} {self.sent_messages_count:8d} {self.received_messages_count:8d} "
                  f"{self.video_sent_messages_count:8d} {self.video_received_messages_count:8d} "
                  f"{audio_sent_kbps:8d} {audio_recv_kbps:8d} {video_sent_kbps:8d} {video_recv_kbps:8d} "
                  f"{int(self.CPU_usage):4d} {int(self.global_CPU_usage):4d}{time_info}")
         
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
            
        # Señalizar que hemos terminado (para video_loop)
        self.running = False

    def video_loop(self):
        while self.running:
            # 1. Capturar frame (la cámara ya controla el FPS por hardware)
            _, frame = self.cap.read()
            data = frame.tobytes()

            # 2. Entrelazar envío y recepción de fragmentos
                        
            fragments_received_this_cycle = 0
            
            for frag_idx in range(self.total_frags):
                # ENVÍO: Enviar un fragmento
                start, end = self.fragment_ranges[frag_idx]
                payload = data[start:end]
                packet = self.fragment_headers[frag_idx] + payload
                
                try:
                    self.video_sock.sendto(packet, self.video_addr)
                    # Actualiza contadores de envío
                    self.video_sent_bytes_count += len(packet)
                    self.video_sent_messages_count += 1
                except BlockingIOError:
                    # Si el buffer está lleno, esperamos un poco y continuamos
                    time.sleep(0.001)
                    continue
                
                # RECEPCIÓN: Intentar recibir un fragmento (sin bloquear)
                try:
                    rlist, _, _ = select.select([self.video_sock], [], [], 0)
                    if rlist:
                        packet, addr = self.video_sock.recvfrom(self.effective_video_payload_size + self.header_size)
                        
                        # Actualiza contadores de recepción
                        self.video_received_bytes_count += len(packet)
                        self.video_received_messages_count += 1
                        
                        # Procesa el paquete recibido
                        header = packet[:self.header_size]
                        payload = packet[self.header_size:]
                        
                        try:
                            recv_frag_idx, = struct.unpack(self._header_format, header)
                            
                            start = recv_frag_idx * self.effective_video_payload_size
                            end = min(start + len(payload), self.expected_frame_size)
                            flat_frame = self.temp_frame_buffer.reshape(-1)
                            flat_frame[start:end] = np.frombuffer(payload, dtype=np.uint8, count=(end-start))
                            fragments_received_this_cycle += 1
                        except struct.error:
                            pass
                except BlockingIOError:
                    pass
                except Exception as e:
                    # Solo registrar errores en modo verbose si hubiera un flag específico
                    pass
            
            # 3. Actualizar y mostrar el frame remoto
            self.remote_frame = self.temp_frame_buffer.copy()
            
            # Opcional: registrar estadísticas de recepción para debug
            self.fragments_received_in_last_cycle = fragments_received_this_cycle
            
            cv2.imshow("Video", self.remote_frame)
            cv2.waitKey(1)
            
    def run(self):
        # Si no hay video, usar el comportamiento de Minimal__verbose
        if not args.show_video or not hasattr(self, 'cap') or self.cap is None:
            print("Video desactivado. Ejecutando solo la parte de audio en modo verbose.")
            if hasattr(minimal, 'Minimal__verbose'):
                super(Minimal_Video, self).run()
            else:
                super().run()
            return
            
        if not hasattr(self, 'loop_cycle_feedback'):
            print("Advertencia: El bucle de feedback de estadísticas no está disponible. Ejecutando sin estadísticas.")
            super().run()
            return
        
        if self.cap is not None:
            cycle_feedback_thread = threading.Thread(target=self.loop_cycle_feedback, daemon=True, name="FeedbackThread")
            self.print_header()
            cycle_feedback_thread.start()
            print("Iniciando video con bucle unificado y protocolo simplificado (verbose)...")
            
            t_unified = threading.Thread(target=self.video_loop, daemon=True, name="UnifiedVideoThread")
            t_unified.start()
            
            try:
                with self.stream(self._handler):
                    if self.end_time:
                        # Esperar hasta que se alcance el tiempo límite
                        self.time_event.wait(timeout=args.reading_time + 0.5)  # Pequeño margen
                    else:
                        # Comportamiento normal esperando entrada de usuario
                        input("Presiona Enter para terminar\n")
                    
            except KeyboardInterrupt:
                print("Interrupción por teclado detectada.")
            finally:
                print("Deteniendo aplicación de video...")
                self.running = False
                if self.cap and self.cap.isOpened():
                    self.cap.release()
                cv2.destroyAllWindows()
                if hasattr(self, 'video_sock') and self.video_sock:
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
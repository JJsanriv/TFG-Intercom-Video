#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""
Minimal_Video_FPS: Extiende Minimal_Video para controlar los FPS (frames por segundo) de 
la transmisión de video. Actúa como un regulador para evitar sobrecarga de CPU y red.

El control de FPS se implementa esperando el tiempo necesario entre frames para mantener
la tasa solicitada. Si el procesamiento toma demasiado tiempo, no se espera.

Hereda todos los parámetros de Minimal_Video y usa --fps para determinar la tasa objetivo.
"""

import cv2
import time
import minimal_video
import struct
import numpy as np
import select
import psutil

class Minimal_Video_FPS(minimal_video.Minimal_Video):
    def __init__(self):
        super().__init__()
        self.set_fps()

    def set_fps(self):
        """Configura el FPS objetivo desde los argumentos"""
        self.fps_target = None
        if hasattr(minimal_video, 'args'):
            args = minimal_video.args
            if hasattr(args, 'fps') and args.fps > 0:
                self.fps_target = args.fps
                print(f"[Minimal_Video_FPS] FPS objetivo para control de bucle: {self.fps_target}")

    def video_loop(self):
        """Método principal para la captura y transmisión de video con control de FPS"""
        try:
            while self.running:
                loop_start = time.time()
                
                # Capturar frame
                ret, frame = self.cap.read()
                
                # Convertir a bytes para transmisión
                data = frame.tobytes()

                # Procesar fragmentos y enviarlos
                self._process_fragments(data)
                
                # Mostrar el frame
                cv2.imshow("Video", self.remote_frame)
                cv2.waitKey(1)
                
                # Control de FPS
                self._control_framerate(loop_start)
                
        except Exception as e:
            print(f"Error en video_loop: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_fragments(self, frame_data):
        """Procesa y envía los fragmentos del frame, y recibe fragmentos entrantes"""
        for frag_idx in range(self.total_frags):
            # ENVÍO
            start, end = self.fragment_ranges[frag_idx]
            payload = frame_data[start:end]
            packet = self.fragment_headers[frag_idx] + payload

            try:
                self.video_sock.sendto(packet, self.video_addr)
            except Exception:
                # Ignorar errores de envío
                pass

            # RECEPCIÓN (espera hasta 1 ms)
            try:
                rlist, _, _ = select.select([self.video_sock], [], [], 0.001)
                if rlist:
                    packet, addr = self.video_sock.recvfrom(self.effective_video_payload_size + self.header_size)
                    self._process_received_packet(packet)
            except Exception:
                # Ignorar errores de recepción
                pass
                
    def _process_received_packet(self, packet):
        """Procesa un paquete recibido y actualiza el frame remoto"""
        header = packet[:self.header_size]
        payload = packet[self.header_size:]
        
        try:
            recv_frag_idx, = struct.unpack(self._header_format, header)
            start = recv_frag_idx * self.effective_video_payload_size
            end = min(start + len(payload), self.expected_frame_size)
            
            # Actualizar el frame remoto con el fragmento recibido
            flat_frame = self.remote_frame.reshape(-1)
            flat_frame[start:end] = np.frombuffer(payload, dtype=np.uint8, count=(end - start))
        except Exception:
            # Ignorar errores de procesamiento
            pass
    
    def _control_framerate(self, start_time):
        """Controla la tasa de frames esperando el tiempo necesario"""
        if self.fps_target:
            elapsed = time.time() - start_time
            frame_time = 1.0 / self.fps_target
            delay = frame_time - elapsed
            
            if delay > 0:
                time.sleep(delay)


class Minimal_Video_FPS_Verbose(Minimal_Video_FPS, minimal_video.Minimal_Video__verbose):
    _printed_init_info = False  # Flag de clase para evitar prints duplicados
    _printed_final_averages = False  # Flag de clase para evitar prints duplicados finales

    def __init__(self):
        
        super().__init__()

        # Solo mostrar el mensaje de inicialización una vez
        if not Minimal_Video_FPS_Verbose._printed_init_info:
            print("[Minimal_Video_FPS_Verbose] Modo verbose con estadísticas de FPS inicializado")
            print("\n=== Estadísticas de FPS se mostrarán en línea separada ===\n")
            Minimal_Video_FPS_Verbose._printed_init_info = True

        # Variables para estadísticas de FPS
        self._fps_real = 0
        self._frame_times = []
        self._max_frame_history = 30  # Para calcular FPS promedio
        self._last_frame_time = time.time()
        self._fps_stats_cycle = 0

    def _process_fragments(self, frame_data):
        """Versión que cuenta estadísticas pero delega la funcionalidad principal a la clase base"""
        super()._process_fragments(frame_data)
        # Solo para que minimal_video.Minimal_Video__verbose tenga la información que necesita
        if hasattr(self, 'video_sent_bytes_count'):
            self.video_sent_bytes_count += len(frame_data)
            self.video_sent_messages_count += 1

    def _control_framerate(self, start_time):
        """Versión que registra estadísticas de FPS y delega el control a la clase base"""
        now = time.time()
        frame_duration = now - self._last_frame_time
        self._last_frame_time = now
        self._frame_times.append(frame_duration)
        if len(self._frame_times) > self._max_frame_history:
            self._frame_times.pop(0)
        if self._frame_times:
            avg_frame_time = sum(self._frame_times) / len(self._frame_times)
            self._fps_real = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        super()._control_framerate(start_time)
        self._fps_stats_cycle += 1



    def print_final_averages(self):
        """Muestra estadísticas finales, solo una vez."""
        if Minimal_Video_FPS_Verbose._printed_final_averages:
            return
        Minimal_Video_FPS_Verbose._printed_final_averages = True

        # Llama solo a la parte relevante de la base si lo necesitas
        if hasattr(minimal_video.Minimal_Video__verbose, 'print_final_averages'):
            minimal_video.Minimal_Video__verbose.print_final_averages(self)

        # Añadir estadísticas de FPS
        if self._frame_times:
            avg_frame_time = sum(self._frame_times) / len(self._frame_times)
            fps_real_avg = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            print("\n=== Estadísticas de FPS ===")
            print(f"FPS objetivo:     {self.fps_target:.1f}")
            print(f"FPS real promedio: {fps_real_avg:.1f}")
            print(f"Eficiencia FPS:    {(fps_real_avg/self.fps_target*100 if self.fps_target else 0):.1f}%")
            print("==========================")

if __name__ == "__main__":
    try:
        import argcomplete
        argcomplete.autocomplete(minimal_video.parser)
    except ImportError:
        pass
    
    args = minimal_video.parser.parse_args()
    minimal_video.args = args
    
    # Comprobar si el modo verbose está activado
    verbose_enabled = (getattr(args, 'show_stats', False) or
                       getattr(args, 'show_samples', False) or
                       getattr(args, 'show_spectrum', False))
    verbose_class_exists = hasattr(minimal_video, 'Minimal_Video__verbose')
    
    if verbose_enabled and verbose_class_exists:
        print("Iniciando en modo Verbose FPS...")
        intercom_app = Minimal_Video_FPS_Verbose()
    else:
        print("Iniciando en modo FPS estándar...")
        intercom_app = Minimal_Video_FPS()
    
    try:
        intercom_app.run()
    except KeyboardInterrupt:
        print("\nInterrupción por teclado detectada.")
    except Exception as e:
        print(f"\nError inesperado: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if hasattr(intercom_app, 'print_final_averages') and callable(intercom_app.print_final_averages):
            time.sleep(0.2)
            intercom_app.print_final_averages()
        print("Programa terminado.")
#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""
Minimal_Video_FPS: Extiende Minimal_Video para controlar los FPS (frames por segundo) de 
la transmisión de video. Actúa como un regulador para evitar sobrecarga de CPU y red.

El control de FPS se implementa esperando el tiempo necesario entre frames para mantener
la tasa solicitada. Si el procesamiento toma demasiado tiempo, no se espera.

Hereda todos los parámetros de Minimal_Video y usa --fps para determinar la tasa objetivo.
"""

import time
import minimal_video
import struct
import numpy as np
import select
import cv2

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
    
    def _control_framerate(self, start_time):
        """Controla la tasa de frames esperando el tiempo necesario"""
        if self.fps_target:
            elapsed = time.time() - start_time
            frame_time = 1.0 / self.fps_target
            delay = frame_time - elapsed
            
            if delay > 0:
                time.sleep(delay)
    
    def video_loop(self):
        """Versión modificada que añade control de FPS"""
        try:
            while self.running:
                loop_start = time.time()
                
                # Llamar a la implementación base pero no directamente al método completo
                # porque no podríamos insertar el control de FPS al final de cada iteración
                ret, frame = self.cap.read()
                
                data = frame.tobytes()
                
                # Usar la misma lógica de procesamiento de fragmentos que la clase base
                for frag_idx in range(self.total_frags):
                    # ENVÍO
                    start, end = self.fragment_ranges[frag_idx]
                    payload = data[start:end]
                    packet = self.fragment_headers[frag_idx] + payload

                    try:
                        self.video_sock.sendto(packet, self.video_addr)
                    except Exception:
                        pass

                    # RECEPCIÓN (espera hasta 1 ms)
                    rlist, _, _ = select.select([self.video_sock], [], [], 0.001)
                    if rlist:
                        packet, addr = self.video_sock.recvfrom(self.effective_video_payload_size + self.header_size)
                            
                        # Procesar el paquete como en la clase base
                        header = packet[:self.header_size]
                        payload = packet[self.header_size:]
                            
                        recv_frag_idx, = struct.unpack(self._header_format, header)
                        start = recv_frag_idx * self.effective_video_payload_size
                        end = min(start + len(payload), self.expected_frame_size)
                                
                        flat_frame = self.remote_frame.reshape(-1)
                        flat_frame[start:end] = np.frombuffer(payload, dtype=np.uint8, count=(end - start))
                            
                # Mostrar el frame como en la clase base
                
                cv2.imshow("Video", self.remote_frame)
                cv2.waitKey(1)
                
                # Esta es la única diferencia real: control de framerate
                self._control_framerate(loop_start)
                
        except Exception as e:
            pass


class Minimal_Video_FPS_Verbose(Minimal_Video_FPS, minimal_video.Minimal_Video__verbose):
    """Versión verbose que añade estadísticas de FPS"""
    
    def __init__(self):
        # Inicializar variables específicas de FPS antes de llamar a los constructores base
        self._fps_real = 0
        self._frame_times = []
        self._max_frame_history = 30
        self._last_frame_time = time.time()
        
        # Llamar a los constructores base en el orden correcto
        Minimal_Video_FPS.__init__(self)
        minimal_video.Minimal_Video__verbose.__init__(self)
        
        # Mostrar mensaje de inicialización
        print("[Minimal_Video_FPS_Verbose] Modo verbose con estadísticas de FPS inicializado")
    
    def _control_framerate(self, start_time):
        """Registra estadísticas de FPS antes de aplicar el control de framerate"""
        # Calcular FPS real
        now = time.time()
        frame_duration = now - self._last_frame_time
        self._last_frame_time = now
        
        # Actualizar historial de tiempos entre frames
        self._frame_times.append(frame_duration)
        if len(self._frame_times) > self._max_frame_history:
            self._frame_times.pop(0)
        
        # Calcular FPS promedio actual
        if self._frame_times:
            avg_frame_time = sum(self._frame_times) / len(self._frame_times)
            self._fps_real = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        # Llamar al control de framerate de la clase base
        Minimal_Video_FPS._control_framerate(self, start_time)
    
    def video_loop(self):
        """Utiliza el video_loop de la clase base Minimal_Video_FPS"""
        # No hay necesidad de reimplementar este método, ya que estamos heredando
        # correctamente de Minimal_Video_FPS y no necesitamos cambiar su comportamiento
        Minimal_Video_FPS.video_loop(self)
        
    def print_final_averages(self):
        """Añade estadísticas de FPS después de las estadísticas base"""
        # Llamar primero a la implementación de la clase base para mostrar sus estadísticas
        if hasattr(minimal_video.Minimal_Video__verbose, 'print_final_averages'):
            minimal_video.Minimal_Video__verbose.print_final_averages(self)
        
        # Añadir estadísticas específicas de FPS
        if hasattr(self, '_frame_times') and self._frame_times:
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
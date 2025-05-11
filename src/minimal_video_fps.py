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
                data = self.capture_image()
                for frag_idx in range(self.total_frags):
                    self.send_video_fragment(frag_idx, data)
                    self.receive_video_fragment()
                self.show_video()
                self._control_framerate(loop_start)
        except Exception as e:
            print(f"[Minimal_Video_FPS] Error en el bucle de video: {e}")
            pass

class Minimal_Video_FPS_Verbose(Minimal_Video_FPS, minimal_video.Minimal_Video__verbose):
    """Versión verbose que añade estadísticas de FPS"""
    def __init__(self):
        self._fps_real = 0
        self._frame_times = []
        self._max_frame_history = 30
        self._last_frame_time = time.time()
        Minimal_Video_FPS.__init__(self)
        minimal_video.Minimal_Video__verbose.__init__(self)
        print("[Minimal_Video_FPS_Verbose] Modo verbose con estadísticas de FPS inicializado")
    
    def _control_framerate(self, start_time):
        now = time.time()
        frame_duration = now - self._last_frame_time
        self._last_frame_time = now

        self._frame_times.append(frame_duration)
        if len(self._frame_times) > self._max_frame_history:
            self._frame_times.pop(0)
        if self._frame_times:
            avg_frame_time = sum(self._frame_times) / len(self._frame_times)
            self._fps_real = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        Minimal_Video_FPS._control_framerate(self, start_time)

    def video_loop(self):
        Minimal_Video_FPS.video_loop(self)
        
    def print_final_averages(self):
        if hasattr(minimal_video.Minimal_Video__verbose, 'print_final_averages'):
            minimal_video.Minimal_Video__verbose.print_final_averages(self)
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
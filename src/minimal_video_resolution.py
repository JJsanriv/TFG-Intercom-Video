#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""
Minimal_Video_Resolution: Extends Minimal_Video_FPS to allow selecting specific video resolutions compatible with the camera. 
If the requested resolution is not available, it selects the closest higher one and then resizes the image to the desired resolution.

It inherits all parameters from Minimal_Video_FPS and adds:
    --camera_device: Camera device (e.g., /dev/video0)
"""

import cv2
import minimal_video_fps
import minimal_video
import numpy as np
import time
import struct
import subprocess
import re
import os

minimal_video.parser.add_argument('--camera_device', type=str, default='/dev/video0', help='Dispositivo de cámara (ej: /dev/video0)')


class Minimal_Video_Resolution(minimal_video_fps.Minimal_Video_FPS):
    def __init__(self):
        super().__init__()
        self.set_resolution()
        if self.resize_needed:
            self.update_fragment_parameters()

    def set_resolution(self):
        self.target_width = 0
        self.target_height = 0
        self.resize_needed = False

        args = minimal_video.args
        if hasattr(args, 'width') and hasattr(args, 'height') and args.width > 0 and args.height > 0:
            self.target_width = args.width
            self.target_height = args.height

            self.find_closest_resolution()

            print(f"[Minimal_Video_Resolution] Resolución objetivo: {self.target_width}x{self.target_height}")
            print(f"[Minimal_Video_Resolution] Resolución captura: {self.capture_width}x{self.capture_height}")
            if self.resize_needed:
                print("[Minimal_Video_Resolution] Se aplicará reescalado de imagen")

    def get_supported_resolutions(self):
        supported_resolutions = []
        camera_device = getattr(minimal_video.args, 'camera_device', '/dev/video0')
        
        try:
            try:
                subprocess.run(['which', 'v4l2-ctl'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("[Minimal_Video_Resolution] v4l2-ctl no encontrado, usando resoluciones estándar")
                return self.get_standard_resolutions()
                
            if not os.path.exists(camera_device):
                print(f"[Minimal_Video_Resolution] Dispositivo {camera_device} no encontrado, usando resoluciones estándar")
                return self.get_standard_resolutions()
            
            
            cmd = ['v4l2-ctl', f'--device={camera_device}', '--list-formats-ext']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print("[Minimal_Video_Resolution] Error al obtener formatos de cámara, usando resoluciones estándar")
                return self.get_standard_resolutions()
            
            # It analyzes the output of v4l2-ctl to extract supported resolutions
            output = result.stdout
            size_pattern = r'Size: Discrete (\d+)x(\d+)'
            matches = re.findall(size_pattern, output)
            for width_str, height_str in matches:
                width = int(width_str)
                height = int(height_str)
                if (width, height) not in supported_resolutions:  # Avoid duplicates
                    supported_resolutions.append((width, height))
            supported_resolutions.sort(key=lambda res: res[0] * res[1])
            if not supported_resolutions:
                print("[Minimal_Video_Resolution] No se detectaron resoluciones, usando estándar")
                return self.get_standard_resolutions()
            return supported_resolutions
        except Exception as e:
            print(f"[Minimal_Video_Resolution] Error al consultar resoluciones: {e}")
            return self.get_standard_resolutions()

    def get_standard_resolutions(self):
        """Returns a list of standard resolutions if no specific ones are found"""
        standard_resolutions = [
            (320, 240),    # QVGA
            (352, 288),    # CIF
            (640, 360),    # nHD
            (640, 480),    # VGA
            (800, 600),    # SVGA
            (1024, 768),   # XGA
            (1280, 720),   # HD
            (1280, 1024),  # SXGA
            (1366, 768),   # WXGA
            (1600, 900),   # HD+
            (1920, 1080),  # Full HD
            (2560, 1440),  # QHD
            (3840, 2160)   # 4K UHD
        ]
        print("[Minimal_Video_Resolution] Usando lista de resoluciones estándar")
        return standard_resolutions

    def find_closest_resolution(self):
    
        if not hasattr(self, 'cap') or self.cap is None:
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

        self.capture_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.capture_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.resize_needed = (self.capture_width != self.target_width or 
                              self.capture_height != self.target_height)

        # If the requested resolution is not available, find the closest higher one
        if self.capture_width < self.target_width or self.capture_height < self.target_height:
            supported_resolutions = self.get_supported_resolutions()
            closest_resolution = None
            for res_width, res_height in supported_resolutions:
                if res_width >= self.target_width and res_height >= self.target_height:
                    closest_resolution = (res_width, res_height)
                    break
            if closest_resolution:
                res_width, res_height = closest_resolution
                print(f"[Minimal_Video_Resolution] Configurando resolución a {res_width}x{res_height}")
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_height)
                self.capture_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.capture_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.resize_needed = True
            else:
                print("[Minimal_Video_Resolution] No se encontró resolución adecuada")
    
    def update_fragment_parameters(self):
        if not self.resize_needed:
            return

        default_frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        self.expected_frame_size = default_frame.nbytes
        self.total_frags = (self.expected_frame_size + self.effective_video_payload_size - 1) // self.effective_video_payload_size
        self.remote_frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        self.fragment_ranges = []
        for i in range(self.total_frags):
            start = i * self.effective_video_payload_size
            end = min(start + self.effective_video_payload_size, self.expected_frame_size)
            self.fragment_ranges.append((start, end))
        self.fragment_headers = [struct.pack(self._header_format, i) for i in range(self.total_frags)]
        print(f"[Minimal_Video_Resolution] Actualizado fragmentación para {self.total_frags} fragmentos")
    
    def process_frame(self, frame):
        if self.resize_needed and hasattr(self, 'target_width') and self.target_width > 0:
            interpolation = cv2.INTER_AREA if (frame.shape[1] > self.target_width) else cv2.INTER_LINEAR
            resized_frame = cv2.resize(frame, (self.target_width, self.target_height), interpolation=interpolation)
            return resized_frame
        return frame

    def capture_image(self):
        _, frame = self.cap.read()
        frame = self.process_frame(frame)
        return frame.tobytes()

    def video_loop(self):
        try:
            while self.running:
                loop_start = time.time()
                data = self.capture_image()
                for frag_idx in range(self.total_frags):
                    self.send_video_fragment(frag_idx, data)
                    self.receive_video_fragment()
                self.show_video()
                self.control_framerate(loop_start)
        except Exception as e:
            pass

class Minimal_Video_Resolution_Verbose(Minimal_Video_Resolution, minimal_video_fps.Minimal_Video_FPS_Verbose):
    
    def __init__(self):
        self._resize_times = []
        self._max_resize_history = 30
        self._supported_resolutions = []
        Minimal_Video_Resolution.__init__(self)
        self._total_resize_time = 0
        self._resize_count = 0
        self._supported_resolutions = self.get_supported_resolutions()
        print("[Minimal_Video_Resolution_Verbose] Modo verbose con estadísticas de resolución inicializado")

    def process_frame(self, frame):
        
        if self.resize_needed:
            start_time = time.time()
            resized_frame = super().process_frame(frame)
            resize_time = time.time() - start_time
            self._resize_times.append(resize_time)
            if len(self._resize_times) > self._max_resize_history:
                self._resize_times.pop(0)
            self._total_resize_time += resize_time
            self._resize_count += 1
            return resized_frame
        return frame

    def capture_image(self):
        _, frame = self.cap.read()
        frame = self.process_frame(frame)
        return frame.tobytes()

    def video_loop(self):
        try:
            while self.running:
                loop_start = time.time()
                data = self.capture_image()
                fragments_received_this_cycle = 0

                for frag_idx in range(self.total_frags):
                    sent_len = self.send_video_fragment(frag_idx, data)
                    self.video_sent_bytes_count += sent_len
                    self.video_sent_messages_count += 1

                    recv_idx, recv_len = self.receive_video_fragment()
                    if recv_len:
                        self.video_received_bytes_count += recv_len
                        self.video_received_messages_count += 1
                        fragments_received_this_cycle += 1

                self._fragments_received_this_cycle = fragments_received_this_cycle
                self.show_video()
                self.control_framerate(loop_start)
        except Exception as e:
            print(f"[Minimal_Video_Resolution_Verbose] Error en el bucle de video: {e}")
            pass

    def print_final_averages(self):
        super().print_final_averages()
        if self.resize_needed and self._resize_count > 0:
            avg_resize_time = self._total_resize_time / self._resize_count
            print("\n=== Estadísticas de Resolución ===")
            print(f"Resolución objetivo: {self.target_width}x{self.target_height}")
            print(f"Resolución real: {self.capture_width}x{self.capture_height}")
            print(f"Tiempo promedio de reescalado: {avg_resize_time*1000:.2f} ms")
            print(f"Impacto en rendimiento:        {avg_resize_time/((1.0/self.fps_target) if self.fps_target else 1)*100:.1f}%")
            print("=================================")
        print("\n=== Resoluciones compatibles de la cámara ===")
        if self._supported_resolutions:
            for idx, (width, height) in enumerate(self._supported_resolutions, 1):
                resolution_str = f"{width}x{height}"
                if width == self.capture_width and height == self.capture_height:
                    resolution_str += " * SELECCIONADA"
                print(f"  {idx}. {resolution_str}")
        else:
            print("  No se detectaron resoluciones compatibles específicas.")
        camera_device = getattr(minimal_video.args, 'camera_device', '/dev/video0')
        print(f"\nDispositivo de cámara: {camera_device}")
        print("==========================================")

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
        print("Iniciando en modo Verbose con control de resolución...")
        intercom_app = Minimal_Video_Resolution_Verbose()
    else:
        print("Iniciando en modo estándar con control de resolución...")
        intercom_app = Minimal_Video_Resolution()

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

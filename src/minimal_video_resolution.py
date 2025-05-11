#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""
Minimal_Video_Resolution: Extiende Minimal_Video_FPS para permitir seleccionar resoluciones 
de video específicas. Si la resolución solicitada no está disponible, selecciona la más cercana
por encima y luego reescala la imagen al tamaño deseado.

Hereda todos los parámetros de Minimal_Video_FPS y añade:
--width: Ancho deseado en píxeles
--height: Alto deseado en píxeles
--camera_device: Dispositivo de cámara (ej: /dev/video0)
"""

import cv2
import argparse
import minimal_video_fps
import minimal_video
import numpy as np
import time
import struct
import subprocess
import re
import os
import logging
import select

# Añadir argumento para especificar el dispositivo de cámara
if not hasattr(minimal_video.parser, 'camera_device_added'):
    minimal_video.parser.add_argument('--camera_device', type=str, default='/dev/video0',
                                      help='Dispositivo de cámara (ej: /dev/video0)')
    minimal_video.parser.camera_device_added = True


class Minimal_Video_Resolution(minimal_video_fps.Minimal_Video_FPS):
    def __init__(self):
        # Inicializar la clase base primero
        super().__init__()
        # Configurar la resolución después
        self.set_resolution()
        # Reinicializar parámetros de fragmentación si cambia la resolución
        if self.resize_needed:
            self._update_fragment_parameters()
        
    def set_resolution(self):
        """Configura la resolución de video deseada y la resolución real a usar"""
        self.target_width = 0
        self.target_height = 0
        self.resize_needed = False
        
        args = minimal_video.args
        if hasattr(args, 'width') and hasattr(args, 'height') and args.width > 0 and args.height > 0:
            self.target_width = args.width
            self.target_height = args.height
            
            # Obtener resoluciones disponibles
            self._find_closest_resolution()
            
            print(f"[Minimal_Video_Resolution] Resolución objetivo: {self.target_width}x{self.target_height}")
            print(f"[Minimal_Video_Resolution] Resolución captura: {self.capture_width}x{self.capture_height}")
            if self.resize_needed:
                print("[Minimal_Video_Resolution] Se aplicará reescalado de imagen")
    
    def _get_supported_resolutions(self):
        """
        Obtiene las resoluciones soportadas por la cámara usando v4l2-ctl
        
        Returns:
            list: Lista de tuplas (ancho, alto) ordenadas de menor a mayor
        """
        supported_resolutions = []
        camera_device = getattr(minimal_video.args, 'camera_device', '/dev/video0')
        
        try:
            # Comprobar si v4l2-ctl está disponible
            try:
                subprocess.run(['which', 'v4l2-ctl'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("[Minimal_Video_Resolution] v4l2-ctl no encontrado, usando resoluciones estándar")
                return self._get_standard_resolutions()
                
            # Comprobar si el dispositivo existe
            if not os.path.exists(camera_device):
                print(f"[Minimal_Video_Resolution] Dispositivo {camera_device} no encontrado, usando resoluciones estándar")
                return self._get_standard_resolutions()
            
            # Ejecutar el comando v4l2-ctl para obtener las resoluciones soportadas
            cmd = ['v4l2-ctl', f'--device={camera_device}', '--list-formats-ext']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print("[Minimal_Video_Resolution] Error al obtener formatos de cámara, usando resoluciones estándar")
                return self._get_standard_resolutions()
            
            # Analizar la salida para extraer las resoluciones
            output = result.stdout
            # Buscar todas las líneas que contienen "Size: Discrete"
            size_pattern = r'Size: Discrete (\d+)x(\d+)'
            matches = re.findall(size_pattern, output)
            
            # Convertir los resultados a tuplas de enteros
            for width_str, height_str in matches:
                width = int(width_str)
                height = int(height_str)
                if (width, height) not in supported_resolutions:  # Evitar duplicados
                    supported_resolutions.append((width, height))
            
            # Ordenar las resoluciones de menor a mayor área
            supported_resolutions.sort(key=lambda res: res[0] * res[1])
            
            if not supported_resolutions:
                print("[Minimal_Video_Resolution] No se detectaron resoluciones, usando estándar")
                return self._get_standard_resolutions()
            
            return supported_resolutions
        
        except Exception as e:
            print(f"[Minimal_Video_Resolution] Error al consultar resoluciones: {e}")
            return self._get_standard_resolutions()
    
    def _get_standard_resolutions(self):
        """Devuelve lista de resoluciones estándar como fallback"""
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
    
    def _find_closest_resolution(self):
        """Encuentra la resolución disponible más cercana por encima de la solicitada"""
        if not hasattr(self, 'cap') or self.cap is None:
            return
            
        # Método simple: intentar establecer la resolución deseada directamente
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        
        # Verificar qué resolución se estableció realmente
        self.capture_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.capture_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Determinar si necesitamos reescalar
        self.resize_needed = (self.capture_width != self.target_width or 
                             self.capture_height != self.target_height)
        
        # Si la resolución actual es menor que la solicitada, buscar la siguiente más grande
        if self.capture_width < self.target_width or self.capture_height < self.target_height:
            # Obtener resoluciones soportadas por la cámara
            supported_resolutions = self._get_supported_resolutions()
            
            # Encontrar la resolución soportada más cercana por encima
            closest_resolution = None
            for res_width, res_height in supported_resolutions:
                if res_width >= self.target_width and res_height >= self.target_height:
                    closest_resolution = (res_width, res_height)
                    break
            
            # Si encontramos una resolución adecuada, configurar la cámara
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
    
    def _update_fragment_parameters(self):
        """Actualiza los parámetros de fragmentación según la resolución objetivo"""
        if not self.resize_needed:
            return
            
        # Crear un frame de prueba con las dimensiones objetivo para calcular el tamaño
        dummy_frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        
        # Actualizar el tamaño esperado de un frame completo
        self.expected_frame_size = dummy_frame.nbytes
        
        # Recalcular el número de fragmentos necesarios
        self.total_frags = (self.expected_frame_size + self.effective_video_payload_size - 1) // self.effective_video_payload_size
        
        # Recrear la matriz para el frame remoto con las dimensiones correctas
        self.remote_frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        
        # Recalcular los rangos de fragmentos
        self.fragment_ranges = []
        for i in range(self.total_frags):
            start = i * self.effective_video_payload_size
            end = min(start + self.effective_video_payload_size, self.expected_frame_size)
            self.fragment_ranges.append((start, end))
        
        # Recalcular las cabeceras de fragmentos
        self.fragment_headers = [struct.pack(self._header_format, i) for i in range(self.total_frags)]
        
        print(f"[Minimal_Video_Resolution] Actualizado fragmentación para {self.total_frags} fragmentos")
    
    def _process_frame(self, frame):
        """Procesa el frame aplicando reescalado si es necesario"""
        if self.resize_needed and hasattr(self, 'target_width') and self.target_width > 0:
            # Usar interpolación INTER_AREA para reducir y INTER_LINEAR para agrandar
            interpolation = cv2.INTER_AREA if (frame.shape[1] > self.target_width) else cv2.INTER_LINEAR
            resized_frame = cv2.resize(frame, (self.target_width, self.target_height), 
                                      interpolation=interpolation)
            return resized_frame
        return frame
    
    def video_loop(self):
        """Versión modificada que añade reescalado de resolución"""
        try:
            while self.running:
                loop_start = time.time()
                
                # Capturar frame
                ret, frame = self.cap.read()
                if not ret:
                    continue
                    
                # Aplicar reescalado si es necesario (única diferencia esencial)
                frame = self._process_frame(frame)
                
                # Convertir a bytes para enviar
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
                            
                # Mostrar el frame
                cv2.imshow("Video", self.remote_frame)
                cv2.waitKey(1)
                
                # Control de framerate (reutilizado de la clase base)
                self._control_framerate(loop_start)
                
        except Exception as e:
            pass


class Minimal_Video_Resolution_Verbose(Minimal_Video_Resolution, minimal_video_fps.Minimal_Video_FPS_Verbose):
    """Versión verbose que añade estadísticas de resolución"""
    
    def __init__(self):
        # Inicializar variables específicas antes de llamar a los constructores base
        self._resize_times = []
        self._max_resize_history = 30
        
        # Guardar resoluciones detectadas para estadísticas
        self._supported_resolutions = []
        
        # Llamar a los constructores base en el orden correcto
        Minimal_Video_Resolution.__init__(self)
        
        # Inicializar variables de estadísticas adicionales
        self._total_resize_time = 0
        self._resize_count = 0
        
        # Obtener las resoluciones soportadas para mostrarlas después
        self._supported_resolutions = self._get_supported_resolutions()
        
        print("[Minimal_Video_Resolution_Verbose] Modo verbose con estadísticas de resolución inicializado")
    
    def _process_frame(self, frame):
        """Versión con medición de tiempo del proceso de reescalado"""
        if self.resize_needed:
            start_time = time.time()
            resized_frame = super()._process_frame(frame)
            resize_time = time.time() - start_time
            
            # Actualizar estadísticas
            self._resize_times.append(resize_time)
            if len(self._resize_times) > self._max_resize_history:
                self._resize_times.pop(0)
            
            self._total_resize_time += resize_time
            self._resize_count += 1
            
            return resized_frame
        return frame
    
    def print_final_averages(self):
        """Añade estadísticas de resolución a las estadísticas base"""
        # Llamar primero a la implementación de la clase base
        super().print_final_averages()
        
        # Añadir estadísticas específicas de resolución
        if self.resize_needed and self._resize_count > 0:
            avg_resize_time = self._total_resize_time / self._resize_count
            
            print("\n=== Estadísticas de Resolución ===")
            print(f"Resolución objetivo: {self.target_width}x{self.target_height}")
            print(f"Resolución real: {self.capture_width}x{self.capture_height}")
            print(f"Tiempo promedio de reescalado: {avg_resize_time*1000:.2f} ms")
            print(f"Impacto en rendimiento:        {avg_resize_time/((1.0/self.fps_target) if self.fps_target else 1)*100:.1f}%")
            print("=================================")
        
        # Mostrar resoluciones compatibles detectadas
        print("\n=== Resoluciones compatibles de la cámara ===")
        if self._supported_resolutions:
            for idx, (width, height) in enumerate(self._supported_resolutions, 1):
                resolution_str = f"{width}x{height}"
                # Marcar la resolución actual con un asterisco
                if width == self.capture_width and height == self.capture_height:
                    resolution_str += " * SELECCIONADA"
                print(f"  {idx}. {resolution_str}")
        else:
            print("  No se detectaron resoluciones compatibles específicas.")
        
        # Mostrar dispositivo de cámara utilizado
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
    
    # Comprobar si el modo verbose está activado
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
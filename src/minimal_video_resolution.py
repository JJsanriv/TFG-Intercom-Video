import cv2
import minimal_video_fps
import numpy as np
import math
import struct
import time

class Minimal_Video_Resolution(minimal_video_fps.Minimal_Video_FPS):
    def __init__(self):
        # Detectar resoluciones disponibles antes de inicializar la clase base
        self.detect_supported_resolutions()
        super().__init__()
        self.configure_resolution()

    def detect_supported_resolutions(self):
        """Detecta dinámicamente las resoluciones y FPS soportados por la cámara"""
        print("Detectando resoluciones compatibles con su cámara...")
        
        temp_cap = cv2.VideoCapture(0)
        if not temp_cap.isOpened():
            print("Error: No se pudo abrir la cámara para detección de resoluciones")
            self.supported_resolutions = [(320, 240), (640, 480), (800, 600)]
            self.resolution_fps_map = {
                (320, 240): [30, 25, 20, 15, 10],
                (640, 480): [30, 25, 20, 15, 10],
                (800, 600): [30, 25, 20, 15]
            }
            return
        
        test_resolutions = [
            (160, 120), (320, 240), (640, 480), 
            (800, 600), (1024, 768), (1280, 720), 
            (1920, 1080), (1920, 1200), (2560, 1440)
        ]
        test_fps = [10, 15, 20, 25, 30, 60, 90, 120]
        
        self.supported_resolutions = []
        self.resolution_fps_map = {}
        
        for width, height in test_resolutions:
            temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if (actual_width, actual_height) not in self.supported_resolutions and actual_width > 0 and actual_height > 0:
                self.supported_resolutions.append((actual_width, actual_height))
                self.resolution_fps_map[(actual_width, actual_height)] = []
                
                for fps in test_fps:
                    temp_cap.set(cv2.CAP_PROP_FPS, fps)
                    actual_fps = int(temp_cap.get(cv2.CAP_PROP_FPS))
                    if actual_fps > 0 and actual_fps not in self.resolution_fps_map[(actual_width, actual_height)]:
                        self.resolution_fps_map[(actual_width, actual_height)].append(actual_fps)
        
        temp_cap.release()
        self.supported_resolutions.sort(key=lambda res: res[0] * res[1])
        for res in self.resolution_fps_map:
            self.resolution_fps_map[res].sort(reverse=True)
        
        self.display_supported_modes()

    def display_supported_modes(self):
        """Muestra al usuario todas las resoluciones y FPS disponibles"""
        print("\n===== RESOLUCIONES Y FPS COMPATIBLES CON SU CÁMARA =====")
        if not self.supported_resolutions:
            print("No se detectaron resoluciones compatibles. Usando valores por defecto.")
            return
        print("\nResoluciones detectadas:")
        for i, (width, height) in enumerate(self.supported_resolutions, 1):
            fps_list = self.resolution_fps_map.get((width, height), [])
            fps_str = ", ".join(map(str, fps_list)) if fps_list else "FPS desconocido"
            print(f"  {i}. {width}x{height} - FPS compatibles: {fps_str}")

    def configure_resolution(self):
        """Configura la resolución óptima basada en los argumentos y lo que soporta la cámara"""
        if not hasattr(self, 'cap') or self.cap is None:
            return
            
        args = minimal_video_fps.minimal_video.args
        requested_width = args.width
        requested_height = args.height
        requested_fps = args.fps if hasattr(args, 'fps') else 30
        
        closest_resolution = self.find_closest_resolution(requested_width, requested_height)
        actual_width, actual_height = closest_resolution
        
        compatible_fps = self.find_compatible_fps(actual_width, actual_height, requested_fps)
        exact_res = (actual_width == requested_width) and (actual_height == requested_height)
        exact_fps = (compatible_fps == requested_fps)
        
        if exact_res and exact_fps:
            print(f"\n✅ Resolución y FPS seleccionados ({requested_width}x{requested_height} @ {requested_fps} FPS) son compatibles.\n")
        else:
            print("\n⚠️  La resolución y/o FPS seleccionados NO son compatibles con la cámara.\n")
            if not exact_res:
                print(f"   - Resolución solicitada: {requested_width}x{requested_height}  ⟶  Usando la más cercana: {actual_width}x{actual_height}")
            if not exact_fps:
                print(f"   - FPS solicitado: {requested_fps}  ⟶  Usando el FPS compatible más cercano: {compatible_fps}")
            print(f"   (Si quieres forzar otros valores, consulta la tabla de arriba)\n")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, actual_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, actual_height)
        self.cap.set(cv2.CAP_PROP_FPS, compatible_fps)
        time.sleep(0.5)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps_target = self.fps
        print(f"Configuración final: {self.width}x{self.height} @ {self.fps} FPS")
        self.expected_frame_size = self.width * self.height * 3
        self.total_frags = math.ceil(self.expected_frame_size / self.effective_video_payload_size)
        self.update_fragment_info()

    def find_closest_resolution(self, requested_width, requested_height):
        """Encuentra la resolución soportada más cercana a la solicitada"""
        if not self.supported_resolutions:
            return (requested_width, requested_height)
        min_distance = float('inf')
        closest = None
        for width, height in self.supported_resolutions:
            distance = ((width - requested_width)**2 + (height - requested_height)**2)**0.5
            if distance < min_distance:
                min_distance = distance
                closest = (width, height)
        return closest
    
    def find_compatible_fps(self, width, height, requested_fps):
        """Encuentra el FPS compatible más cercano para la resolución dada"""
        if (width, height) not in self.resolution_fps_map or not self.resolution_fps_map[(width, height)]:
            return requested_fps
        compatible_fps = self.resolution_fps_map[(width, height)]
        closest_fps = min(compatible_fps, key=lambda fps: abs(fps - requested_fps))
        return closest_fps
                
    def update_fragment_info(self):
        """Actualiza información de fragmentación basada en la nueva resolución"""
        self.fragment_ranges = []
        self.fragment_headers = []
        for frag_idx in range(self.total_frags):
            start = frag_idx * self.effective_video_payload_size
            end = min(start + self.effective_video_payload_size, self.expected_frame_size)
            self.fragment_ranges.append((start, end))
            self.fragment_headers.append(struct.pack(self._header_format, frag_idx))
        self.remote_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if hasattr(self, 'temp_frame_buffer'):
            self.temp_frame_buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)

class Minimal_Video_Resolution_Verbose(Minimal_Video_Resolution, minimal_video_fps.Minimal_Video_FPS_Verbose):
    def __init__(self):
        # Inicializar variables para estadísticas antes de llamar al constructor de la clase base
        self.requested_width = None
        self.requested_height = None
        self.requested_fps = None
        self.real_width = None
        self.real_height = None
        self.real_fps = None
        
        # Llamar al constructor de la clase base
        super().__init__()
        
    def configure_resolution(self):
        """Versión verbose que recopila estadísticas adicionales"""
        # Guardar los valores solicitados antes de configurar
        args = minimal_video_fps.minimal_video.args
        self.requested_width = args.width
        self.requested_height = args.height
        self.requested_fps = args.fps if hasattr(args, 'fps') else 30
        
        # Llamar al método de la clase base para hacer la configuración
        super().configure_resolution()
        
        # Ahora guardar las estadísticas adicionales que solo necesita la versión verbose
        self.real_width = self.width
        self.real_height = self.height
        self.real_fps = self.fps
        

    def print_final_averages(self):
        """Versión mejorada del método para mostrar estadísticas finales"""
        # Llamar al método de la clase base
        super().print_final_averages()
        
        # Solo añadir estadísticas de resolución si tenemos los datos
        if hasattr(self, 'requested_width') and hasattr(self, 'real_width'):
            print("\n=== Estadísticas de Resolución ===")
            print(f"Resolución solicitada: {self.requested_width}x{self.requested_height}")
            print(f"Resolución real utilizada: {self.real_width}x{self.real_height}")
            print("=================================")

if __name__ == "__main__":
    try:
        import argcomplete
        argcomplete.autocomplete(minimal_video_fps.minimal_video.parser)
    except ImportError:
        pass
    
    args = minimal_video_fps.minimal_video.parser.parse_args()
    minimal_video_fps.minimal_video.args = args
    verbose_enabled = (
        getattr(args, 'show_stats', False) or
        getattr(args, 'show_samples', False) or
        getattr(args, 'show_spectrum', False)
    )
    if verbose_enabled:
        print("Iniciando en modo Verbose con detección automática de resolución y FPS...")
        intercom_app = Minimal_Video_Resolution_Verbose()
    else:
        print("Iniciando con detección automática de resolución y FPS...")
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
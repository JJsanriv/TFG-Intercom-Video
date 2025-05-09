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
        if args is None:
            args = minimal.parser.parse_args()
        minimal.args = args

        super().__init__()

        if not args.show_video:
            return

        self.video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.video_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.video_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8388608)
        self.video_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8388608)

        self.video_sock.setblocking(False)
        try:
            self.video_sock.bind(("0.0.0.0", args.video_port))
        except OSError as e:
            print(f"Error bind socket video: {e}")
            raise
        self.video_addr = (args.destination_address, args.video_port)

        self.current_frame_fragments = None
        self.fragments_received = 0

        self._header_format = "!H"
        self.header_size = 2
        self.effective_video_payload_size = args.video_payload_size
        self.max_payload_possible = self.effective_video_payload_size - self.header_size
        self.effective_video_payload_size = max(1, min(args.video_payload_size, self.max_payload_possible))
        if self.effective_video_payload_size != args.video_payload_size:
            print(f"Aviso: --video_payload_size ajustado a {self.effective_video_payload_size} bytes.")

        self.cap = None
        self.width = 0
        self.height = 0
        self.fps = 0
        self.latest_captured_frame = None

        self.expected_frame_size = 0
        self.total_frags = 0

        print("Flag --show_video detectado. Intentando inicializar la cámara...")
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise IOError("No se pudo abrir la cámara.")
            if args.width > 0:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            if args.height > 0:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            if args.fps > 0:
                self.cap.set(cv2.CAP_PROP_FPS, args.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            self.expected_frame_size = self.width * self.height * 3
            self.total_frags = math.ceil(self.expected_frame_size / self.effective_video_payload_size)

            print(f"Cámara inicializada: {self.width}x{self.height} @ {self.fps} FPS")
            print(f"Payload/frag UDP: {self.effective_video_payload_size} bytes")
            print(f"Frame esperado: {self.expected_frame_size} bytes, {self.total_frags} fragmentos")

            self.fragment_ranges = []
            self.fragment_headers = []
            for frag_idx in range(self.total_frags):
                start = frag_idx * self.effective_video_payload_size
                end = min(start + self.effective_video_payload_size, self.expected_frame_size)
                self.fragment_ranges.append((start, end))
                self.fragment_headers.append(struct.pack(self._header_format, frag_idx))

            self.remote_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self.received_remote_frame = False

            if not hasattr(self, 'temp_frame_buffer') or self.temp_frame_buffer.shape != (self.height, self.width, 3):
                self.temp_frame_buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # fragments_received_this_cycle ya NO está en minimal, sólo en verbose

        except Exception as e:
            print(f"Error al inicializar la cámara: {e}. Deshabilitando video.")
            if self.cap:
                self.cap.release()
            self.cap = None

        self.running = True

    def video_loop(self):
        while self.running:
            _, frame = self.cap.read()
            data = frame.tobytes()

            for frag_idx in range(self.total_frags):
                # ENVÍO
                start, end = self.fragment_ranges[frag_idx]
                payload = data[start:end]
                packet = self.fragment_headers[frag_idx] + payload

                try:
                    self.video_sock.sendto(packet, self.video_addr)
                except BlockingIOError:
                    # No hacemos continue, permitimos seguir a la recepción
                    pass

                # RECEPCIÓN (espera hasta 1 ms)
                try:
                    rlist, _, _ = select.select([self.video_sock], [], [], 0.001)
                    if rlist:
                        packet, addr = self.video_sock.recvfrom(self.effective_video_payload_size + self.header_size)
                        header = packet[:self.header_size]
                        payload = packet[self.header_size:]
                        try:
                            recv_frag_idx, = struct.unpack(self._header_format, header)
                            start = recv_frag_idx * self.effective_video_payload_size
                            end = min(start + len(payload), self.expected_frame_size)
                            flat_frame = self.remote_frame.reshape(-1)
                            flat_frame[start:end] = np.frombuffer(payload, dtype=np.uint8, count=(end - start))
                        except struct.error:
                            pass
                except BlockingIOError:
                    pass
                except Exception:
                    pass

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
        
            t_unified = threading.Thread(target=self.video_loop, daemon=True, name="UnifiedVideoThread")
            t_unified.start()
        
            try:
                super().run() 
            except KeyboardInterrupt:
                print("Interrupción por teclado detectada.")
            finally:
                print("Deteniendo aplicación de video...")
                self.running = False
                if hasattr(self, 'cap') and self.cap.isOpened():
                    self.cap.release()
                cv2.destroyAllWindows()
                if hasattr(self, 'video_sock') and self.video_sock:
                    self.video_sock.close()
                print("Aplicación de video detenida.")

class Minimal_Video__verbose(Minimal_Video, minimal.Minimal__verbose):
    def __init__(self):
        super().__init__()

        if not args.show_video or not hasattr(self, 'cap') or self.cap is None:
            return

        try:
            minimal.Minimal__verbose.__init__(self)
            print(f"Verbose Mode: stats cycle = {self.seconds_per_cycle}s")
        except AttributeError:
            print("Error: No se pudo inicializar minimal.Minimal__verbose. Las estadísticas no funcionarán.")

        self.video_sent_bytes_count = 0
        self.video_sent_messages_count = 0
        self.video_received_bytes_count = 0
        self.video_received_messages_count = 0

        # Acumuladores para promedios globales
        self._total_audio_sent_bytes = 0
        self._total_audio_received_bytes = 0
        self._total_video_sent_bytes = 0
        self._total_video_received_bytes = 0
        self._stats_start_time = time.time()

        # Para estadísticas de fragmentos recibidos por ciclo (solo en verbose)
        self._fragments_received_this_cycle = 0
        self._fragments_received_history = []

        self.total_number_of_sent_frames = 0
        self.frame_time = 1.0 / self.fps

        self.end_time = None
        if args.reading_time:
            self.end_time = time.time() + float(args.reading_time)
            print(f"Programa terminará automáticamente después de {args.reading_time} segundos")
            print(f"Tiempo de finalización programado: {time.strftime('%H:%M:%S', time.localtime(self.end_time))}")
            self.time_event = threading.Event()

    def print_header(self):
        header1 = (
            f"{'':8s}"
            " | " + f"{'AUDIO (msg)':^13s}"
            " | " + f"{'VIDEO (msg)':^13s}"
            " | " + f"{'AUDIO (kbps)':^15s}"
            " | " + f"{'VIDEO (kbps)':^15s}"
            " |     " + f"{'FRAGS':^9s}"
            "       |     " + f"{'CPU (%)':^8s}"
        )
        header2 = (
            f"{'Cycle':>8s}"
            " | " + f"{'Sent':>5s} {'Recv':>5s}"
            "   | " + f"{'Sent':>5s} {'Recv':>5s}"
            "   | " + f"{'Sent':>6s} {'Recv':>6s}"
            "   | " + f"{'Sent':>6s} {'Recv':>6s}"
            "   | " + f"{'Recv/cycle ':>5s} {'Total':>3s}"
            "   | " + f"{'Program':>4s} {'System':>4s}"
        )
        print(header1)
        print(header2)
        print("=" * (8 + 3 + 13 + 3 + 13 + 3 + 15 + 3 + 15 + 3 + 9 + 3 + 8 + 17))

    def loop_cycle_feedback(self):
        if not args.show_video or not hasattr(self, 'cap') or self.cap is None:
            if hasattr(minimal.Minimal__verbose, 'loop_cycle_feedback'):
                return super(Minimal_Video, self).loop_cycle_feedback()
            return

        cycle = 1
        self.old_time = time.time()
        self.old_CPU_time = psutil.Process().cpu_times()[0]

        start_time = self._stats_start_time

        while self.running:
            now = time.time()
            if self.end_time and now >= self.end_time:
                print(f"\nLímite de tiempo alcanzado: {args.reading_time} segundos")
                self.time_event.set()
                break

            elapsed = max(now - self.old_time, 0.001)
            elapsed_CPU_time = psutil.Process().cpu_times()[0] - self.old_CPU_time
            self.CPU_usage = 100 * elapsed_CPU_time / elapsed
            self.global_CPU_usage = psutil.cpu_percent(interval=None)

            audio_sent_kbps = int(self.sent_bytes_count * 8 / 1000 / elapsed)
            audio_recv_kbps = int(self.received_bytes_count * 8 / 1000 / elapsed)
            video_sent_kbps = int(self.video_sent_bytes_count * 8 / 1000 / elapsed)
            video_recv_kbps = int(self.video_received_bytes_count * 8 / 1000 / elapsed)

            self._total_audio_sent_bytes += self.sent_bytes_count
            self._total_audio_received_bytes += self.received_bytes_count
            self._total_video_sent_bytes += self.video_sent_bytes_count
            self._total_video_received_bytes += self.video_received_bytes_count

            # Guarda histórico de fragmentos recibidos por ciclo
            self._fragments_received_history.append(self._fragments_received_this_cycle)

            time_info = ""
            if self.end_time:
                elapsed_total = now - start_time
                progress = min(100, 100 * elapsed_total / args.reading_time)
                time_info = f" | {elapsed_total:.1f}s/{args.reading_time}s ({progress:.0f}%)"

            print(
                f"{cycle:>8d} |"
                f"{self.sent_messages_count:>5d} {self.received_messages_count:>5d}    |"
                f"{self.video_sent_messages_count:>5d} {self.video_received_messages_count:>5d}    |"
                f"{audio_sent_kbps:>6d} {audio_recv_kbps:>6d}    |"
                f"{video_sent_kbps:>6d} {video_recv_kbps:>6d}    |"
                f"{self._fragments_received_this_cycle:>5d} {self.total_frags:>3d}            |"
                f"{int(self.CPU_usage):>4d} {int(self.global_CPU_usage):>6d}"
                f"{time_info}"
            )

            # Reiniciar contadores de uso
            self.sent_bytes_count = 0
            self.received_bytes_count = 0
            self.sent_messages_count = 0
            self.received_messages_count = 0
            self.video_sent_bytes_count = 0
            self.video_received_bytes_count = 0
            self.video_sent_messages_count = 0
            self.video_received_messages_count = 0
            self._fragments_received_this_cycle = 0

            cycle += 1
            self.old_time = now
            self.old_CPU_time = psutil.Process().cpu_times()[0]
            time.sleep(1)

        self.running = False

    def print_final_averages(self):
        total_time = time.time() - self._stats_start_time
        if total_time < 0.1:
            print("Duración demasiado corta para calcular promedios de ancho de banda.")
            return

        audio_sent_kbps = self._total_audio_sent_bytes * 8 / 1000 / total_time
        audio_received_kbps = self._total_audio_received_bytes * 8 / 1000 / total_time
        video_sent_kbps = self._total_video_sent_bytes * 8 / 1000 / total_time
        video_received_kbps = self._total_video_received_bytes * 8 / 1000 / total_time

        avg_frags = (
            sum(self._fragments_received_history) / len(self._fragments_received_history)
            if self._fragments_received_history else 0
        )

        print("\n=== Estadísticas globales de ancho de banda ===")
        print(f"Audio enviado:    {audio_sent_kbps:.2f} kbps")
        print(f"Audio recibido:   {audio_received_kbps:.2f} kbps")
        print(f"Video enviado:    {video_sent_kbps:.2f} kbps")
        print(f"Video recibido:   {video_received_kbps:.2f} kbps")
        print(f"Avg frags/ciclo:  {avg_frags:.1f} / {self.total_frags}")
        print(f"Tiempo total:     {total_time:.1f} s")
        print("=======================================================")

    def video_loop(self):
        while self.running:
            _, frame = self.cap.read()
            data = frame.tobytes()
            fragments_received_this_cycle = 0

            for frag_idx in range(self.total_frags):
                # ENVÍO
                start, end = self.fragment_ranges[frag_idx]
                payload = data[start:end]
                packet = self.fragment_headers[frag_idx] + payload

                try:
                    self.video_sock.sendto(packet, self.video_addr)
                    self.video_sent_bytes_count += len(packet)
                    self.video_sent_messages_count += 1
                except BlockingIOError:
                    # No hacemos continue, permitimos seguir a la recepción
                    pass

                # RECEPCIÓN (espera hasta 1 ms)
                try:
                    rlist, _, _ = select.select([self.video_sock], [], [], 0.001)
                    if rlist:
                        packet, addr = self.video_sock.recvfrom(self.effective_video_payload_size + self.header_size)
                        self.video_received_bytes_count += len(packet)
                        self.video_received_messages_count += 1
                        header = packet[:self.header_size]
                        payload = packet[self.header_size:]
                        try:
                            recv_frag_idx, = struct.unpack(self._header_format, header)
                            start = recv_frag_idx * self.effective_video_payload_size
                            end = min(start + len(payload), self.expected_frame_size)
                            flat_frame = self.remote_frame.reshape(-1)
                            flat_frame[start:end] = np.frombuffer(payload, dtype=np.uint8, count=(end-start))
                            fragments_received_this_cycle += 1
                        except struct.error:
                            pass
                except BlockingIOError:
                    pass
                except Exception:
                    pass

            self._fragments_received_this_cycle = fragments_received_this_cycle
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
            self.print_header()
            cycle_feedback_thread = threading.Thread(target=self.loop_cycle_feedback, daemon=True, name="FeedbackThread")
            cycle_feedback_thread.start()
            print("Iniciando video con bucle unificado y protocolo simplificado (verbose)...")

            t_unified = threading.Thread(target=self.video_loop, daemon=True, name="UnifiedVideoThread")
            t_unified.start()

            try:
                with self.stream(self._handler):
                    if self.end_time:
                        self.time_event.wait(timeout=args.reading_time + 0.5)
                    else:
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
import cv2
import minimal_video
import time
import numpy as np
import struct

class Minimal_Video_FPS(minimal_video.Minimal_Video):
    def __init__(self):
        super().__init__()
        self.set_fps()

    def set_fps(self):
        # Lee el FPS objetivo de los argumentos y guarda como atributo
        self.fps_target = None
        if hasattr(minimal_video, 'args'):
            args = minimal_video.args
            if hasattr(args, 'fps') and args.fps > 0:
                self.fps_target = args.fps
                print(f"[Minimal_Video_FPS] FPS objetivo para control de bucle: {self.fps_target}")

    def video_loop(self):
        
        while self.running:
            loop_start = time.time()
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
                    pass

                # RECEPCIÓN (espera hasta 1 ms)
                try:
                    import select
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

            # CONTROL DE FPS AQUÍ:
            if self.fps_target:
                elapsed = time.time() - loop_start
                delay = (1.0 / self.fps_target) - elapsed
                if delay > 0:
                    time.sleep(delay)

class Minimal_Video_FPS_Verbose(minimal_video.Minimal_Video__verbose, Minimal_Video_FPS):
    def __init__(self):
        Minimal_Video_FPS.__init__(self)
        minimal_video.Minimal_Video__verbose.__init__(self)


    def video_loop(self):
        
        while self.running:
            loop_start = time.time()
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
                    pass

                # RECEPCIÓN (espera hasta 1 ms)
                try:
                    import select
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

            # CONTROL DE FPS: espera lo necesario para respetar el FPS objetivo
            fps_target = getattr(self, "fps_target", None)
            if fps_target:
                elapsed = time.time() - loop_start
                delay = (1.0 / fps_target) - elapsed
                if delay > 0:
                    time.sleep(delay)

if __name__ == "__main__":
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
        intercom_app = Minimal_Video_FPS()

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
            import time
            time.sleep(0.2)
            intercom_app.print_final_averages()
        print("Programa terminado.")
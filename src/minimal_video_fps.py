import cv2
import minimal_video

class Minimal_Video_FPS(minimal_video.Minimal_Video):
    def __init__(self):
        super().__init__()
        # Solo setea FPS si hay cámara y el argumento se proporcionó
        if hasattr(self, 'cap') and self.cap is not None:
            try:
                fps_to_set = getattr(self, 'fps', 0)
                # Si el usuario pasó --fps y es >0, lo aplicamos
                if hasattr(minimal_video, 'args'):
                    args = minimal_video.args
                    if hasattr(args, 'fps') and args.fps > 0:
                        self.cap.set(cv2.CAP_PROP_FPS, args.fps)
                        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                        print(f"[Minimal_Video_FPS] FPS seteados a: {self.fps}")
            except Exception as e:
                print(f"[Minimal_Video_FPS] Error seteando FPS: {e}")

if __name__ == "__main__":
    import minimal_video
    # Usa el mismo parser/args que minimal_video
    args = minimal_video.parser.parse_args()
    minimal_video.args = args
    # El resto igual que en minimal_video
    verbose_enabled = (getattr(args, 'show_stats', False) or
                       getattr(args, 'show_samples', False) or
                       getattr(args, 'show_spectrum', False))
    verbose_class_exists = hasattr(minimal_video, 'Minimal__verbose')

    if verbose_enabled and verbose_class_exists:
        print("Iniciando en modo Verbose FPS...")
        class Minimal_Video_FPS_Verbose(Minimal_Video_FPS, minimal_video.Minimal__verbose):
            def __init__(self):
                Minimal_Video_FPS.__init__(self)
                minimal_video.Minimal__verbose.__init__(self)
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
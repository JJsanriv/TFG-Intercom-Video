#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""
Minimal_Video_FPS: Extends Minimal_Video to control the FPS (frames per second) of the video transmission. 
It adjusts the FPS compatible with the current resolution of the camera and acts as a regulator to avoid CPU and network overload.

FPS control is implemented by waiting the necessary time between frames to maintain the requested rate. 
If processing takes too long, no waiting occurs.

It inherits all parameters from Minimal_Video so you can use --fps to determine the target rate.
"""

import time
import minimal_video
import numpy as np


class Minimal_Video_FPS(minimal_video.Minimal_Video):
    def __init__(self):
        super().__init__()
        self.set_fps()

    def set_fps(self):
        self.fps_target = None
        if hasattr(minimal_video, 'args'):
            args = minimal_video.args
            if hasattr(args, 'fps') and args.fps > 0:
                self.fps_target = args.fps
                print(f"[Minimal_Video_FPS] Target FPS for loop control: {self.fps_target}")
    
    def control_framerate(self, start_time):

        if self.fps_target:
            elapsed = time.time() - start_time
            frame_time = 1.0 / self.fps_target
            delay = frame_time - elapsed 
            
            if delay > 0:
                time.sleep(delay)
    
    def video_loop(self):
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
            print(f"[Minimal_Video_FPS] Error in video loop: {e}")
            pass

class Minimal_Video_FPS_Verbose(Minimal_Video_FPS, minimal_video.Minimal_Video__verbose):

    def __init__(self):
        self.fps_real = 0
        self.frame_times = [] # List to store frame times
        self.max_frame_history = 30 # Number of frames to average
        self.last_frame_time = time.time() 
        super().__init__()
        print("[Minimal_Video_FPS_Verbose] Verbose mode with FPS statistics initialized")
    
    def control_framerate(self, start_time):
        now = time.time()
        frame_duration = now - self.last_frame_time
        self.last_frame_time = now

        self.frame_times.append(frame_duration)
        if len(self.frame_times) > self.max_frame_history: # Limit the history size
            self.frame_times.pop(0) # Remove the oldest frame time
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times) # Average frame time
            self.fps_real = 1.0 / avg_frame_time if avg_frame_time > 0 else 0 # Calculate FPS
        Minimal_Video_FPS.control_framerate(self, start_time)

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
            print(f"[Minimal_Video_FPS] Error in video loop: {e}")
            pass

    def print_final_averages(self):
        if hasattr(minimal_video.Minimal_Video__verbose, 'print_final_averages'):
            minimal_video.Minimal_Video__verbose.print_final_averages(self)
        if hasattr(self, 'frame_times') and self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            fps_real_avg = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            print("\n=== FPS Statistics ===")
            print(f"Target FPS:       {self.fps_target:.1f}")
            print(f"Average real FPS: {fps_real_avg:.1f}")
            print(f"FPS efficiency:   {(fps_real_avg/self.fps_target*100 if self.fps_target else 0):.1f}%")
            print("======================")

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
        print("Starting in Verbose FPS mode...")
        intercom_app = Minimal_Video_FPS_Verbose()
    else:
        print("Starting in standard FPS mode...")
        intercom_app = Minimal_Video_FPS()

    try:
        intercom_app.run()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if hasattr(intercom_app, 'print_final_averages') and callable(intercom_app.print_final_averages):
            time.sleep(0.2)
            intercom_app.print_final_averages()
        print("Program terminated.")
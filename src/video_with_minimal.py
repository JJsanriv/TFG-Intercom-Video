#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK


import os
import signal
import argparse
import sys
import sounddevice as sd
import numpy as np
import socket
import logging
import soundfile as sf
import cv2
import struct
import time
import traceback


FORMAT = "(%(levelname)s) %(module)s: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)


def spinning_cursor():
   while True:
       for cursor in '|/-\\':
           yield cursor
spinner = spinning_cursor()


def int_or_str(text):
   try:
       return int(text)
   except ValueError:
       return text


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input-device", type=int_or_str, help="Input device ID or substring")
parser.add_argument("-o", "--output-device", type=int_or_str, help="Output device ID or substring")
parser.add_argument("-d", "--list-devices", action="store_true", help="Print the available audio devices and quit")
parser.add_argument("-s", "--frames_per_second", type=float, default=44100, help="sampling rate in frames/second")
parser.add_argument("-c", "--frames_per_chunk", type=int, default=1024, help="Number of frames in a chunk")
parser.add_argument("-l", "--listening_port", type=int, default=4444, help="My listening port")
parser.add_argument("-a", "--destination_address", type=str, default=None, help="Destination (interlocutor's listening) address")
parser.add_argument("-p", "--destination_port", type=int, default=4444, help="Destination (interlocutor's listening) port")
parser.add_argument("-f", "--filename", type=str, help="Use a wav/oga/... file instead of the mic data")
parser.add_argument("-t", "--reading_time", type=int, help="Time reading data (mic or file) (only with effect if --show_stats or --show_data is used)")


args = parser.parse_args()


MAX_PAYLOAD_BYTES = 1500
VIDEO_PORT = 4445
VIDEO_FPS = 10
NUMBER_OF_CHANNELS = 2


class VideoAudioIntercom:
   def __init__(self, args):
       self.args = args
       self.sock_audio = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
       self.sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
       self.listening_endpoint_audio = ("0.0.0.0", self.args.listening_port)
       self.listening_endpoint_video = ("0.0.0.0", VIDEO_PORT)
       self.sock_audio.bind(self.listening_endpoint_audio)
       self.sock_video.bind(self.listening_endpoint_video)
       self.chunk_time = self.args.frames_per_chunk / self.args.frames_per_second
       self.destination_address = self.args.destination_address
       self.destination_port_audio = self.args.destination_port
       self.destination_port_video = VIDEO_PORT
       self.shutdown_flag = False
       self.zero_chunk = np.zeros((self.args.frames_per_chunk, NUMBER_OF_CHANNELS), np.int16)
       self.cap = None
       logging.info(f"chunk_time = {self.chunk_time} seconds")


       if self.args.filename:
           logging.info(f"Using \"{self.args.filename}\" as input")
           self.wavfile = sf.SoundFile(self.args.filename, 'r')
           self.audio_stream = self.file_stream
       else:
           self.audio_stream = self.mic_stream


   def pack_audio(self, audio_chunk):
       return audio_chunk.tobytes()


   def unpack_audio(self, packed_chunk):
       return np.frombuffer(packed_chunk, np.int16).reshape(-1, NUMBER_OF_CHANNELS)


   def send_audio(self, packed_chunk):
       self.sock_audio.sendto(packed_chunk, (self.destination_address, self.destination_port_audio))


   def receive_audio(self):
       audio_chunk_size = self.args.frames_per_chunk * NUMBER_OF_CHANNELS * np.dtype(np.int16).itemsize
       audio_packed, _ = self.sock_audio.recvfrom(audio_chunk_size)
       return audio_packed


   def _record_IO_and_play(self, ADC, DAC, frames, time, status):
      
       if self.shutdown_flag:
           raise sd.CallbackAbort
       try:
           data = ADC.copy()
           packed_chunk = self.pack_audio(data)
           if self.destination_address:
               self.send_audio(packed_chunk)
               try:
                   packed_chunk = self.receive_audio()
                   chunk = self.unpack_audio(packed_chunk)
               except (socket.timeout, BlockingIOError):
                   chunk = self.zero_chunk
                   logging.debug("Playing zero chunk")
           else:
               chunk = data
           DAC[:] = chunk
       except Exception as e:
           logging.error(f"Error in audio processing: {e}")
       if __debug__:
           print(next(spinner), end='\b', flush=True)


   def send_video(self, frame):
       if self.destination_address:
           frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           frame_data = frame.tobytes()
           header = struct.pack('>I', len(frame_data))
           frame_data = header + frame_data
           num_chunks = len(frame_data) // MAX_PAYLOAD_BYTES + 1
           self.sock_video.sendto(struct.pack('>I', num_chunks), (self.destination_address, self.destination_port_video))
           for i in range(num_chunks):
               chunk = frame_data[i * MAX_PAYLOAD_BYTES: (i + 1) * MAX_PAYLOAD_BYTES]
               try:
                   self.sock_video.sendto(chunk, (self.destination_address, self.destination_port_video))
               except Exception as e:
                   logging.error(f"Failed to send video chunk: {e}")


   def receive_video(self):
       frame_chunks = {}
       expected_chunk = 0
       frame_size = 0
       num_chunks, _ = self.sock_video.recvfrom(4)
       num_chunks = struct.unpack('>I', num_chunks)[0]
       while True:
           try:
               chunk, _ = self.sock_video.recvfrom(MAX_PAYLOAD_BYTES)
               frame_chunks[expected_chunk] = chunk
               if expected_chunk == 0:
                   frame_size = struct.unpack('>I', chunk[:4])[0] + 4
               expected_chunk += 1
               if len(frame_chunks) >= num_chunks:
                   break
           except socket.timeout as e:
               logging.debug(f"Video receive timeout: {e}")
               break
           except Exception as e:
               logging.error(f"Failed to receive video chunk: {e}")
               logging.error(traceback.format_exc())
               break
       frame_data = b''
       for i in range(expected_chunk):
           if i in frame_chunks:
               frame_data += frame_chunks[i]
           else:
               frame_data += b'\x00' * MAX_PAYLOAD_BYTES
       if len(frame_data) > 4:
           frame_size = struct.unpack('>I', frame_data[:4])[0]
           frame_data = frame_data[4:]
           if len(frame_data) == frame_size:
               try:
                   frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(240, 320, 3)
                   return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
               except Exception as e:
                   logging.error(f"Failed to process received frame: {e}")
                   return None
       return None


   def mic_stream(self, callback_function):
       try:
               with sd.Stream(device=(self.args.input_device, self.args.output_device),
                              dtype=np.int16,
                              samplerate=self.args.frames_per_second,
                              blocksize=self.args.frames_per_chunk,
                              channels=NUMBER_OF_CHANNELS,
                              callback=callback_function):
                   self.run_video()
       except Exception as e:
           logging.error(f"Error in mic_stream: {e}")


   def file_stream(self):
       try:
           self.run_video()
       except Exception as e:
           logging.error(f"Error in file_stream: {e}")


   def run_video(self):
       if self.cap is None:
           self.cap = cv2.VideoCapture(0)
       if not self.cap.isOpened():
           logging.error("Failed to open video capture")
           return
       if self.args.filename:
           self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
           self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
           self.cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
       else:
           self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
           self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
       try:
           while not self.shutdown_flag:
               ret, frame = self.cap.read()
               if not ret:
                   logging.error("Failed to capture frame")
                   break
               cv2.imshow('Video', frame)
               if self.destination_address:
                   self.send_video(frame)
               if cv2.waitKey(int(1000 / VIDEO_FPS)) & 0xFF == ord('q'):
                   break
               if self.destination_address:
                   received_frame = self.receive_video()
                   if received_frame is not None:
                       cv2.imshow('Received Video', received_frame)
       finally:
           self.cap.release()
           cv2.destroyAllWindows()


   def run(self):
       try:
           if self.args.filename:
               self.file_stream()
           else:
               self.mic_stream(self._record_IO_and_play)
       except Exception as e:
           logging.error(f"Error in run: {e}")


   def shutdown(self):
       self.shutdown_flag = True
       self.sock_audio.close()
       self.sock_video.close()
       if self.cap and self.cap.isOpened():
           self.cap.release()
       cv2.destroyAllWindows()


def shutdown_handler(signum, frame):
   global intercom
   logging.info("Shutting down...")
   intercom.shutdown()
   sys.exit(0)


signal.signal(signal.SIGINT, shutdown_handler)


try:
   import argcomplete  # <tab> completion for argparse.
except ImportError:
   logging.warning("Unable to import argcomplete (optional)")


if __name__ == "__main__":
   parser.description = __doc__
   try:
       argcomplete.autocomplete(parser)
   except Exception:
       logging.warning("argcomplete not working :-/")
   args = parser.parse_known_args()[0]


   if args.list_devices:
       print("Available devices:")
       print(sd.query_devices())
       quit()


   if args.destination_address:
       logging.info(f"Waiting for connection to {args.destination_address}:{args.destination_port}")
       logging.info(f"Waiting for connection to {args.destination_address}:{args.destination_port+1}")
       sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
       sock.settimeout(5)
       try:
           sock.connect((args.destination_address, args.destination_port))
           sock.connect((args.destination_address, args.destination_port+1))
           logging.info("Connection established!")
           sock.close()
       except socket.error as e:
           logging.error(f"Connection failed: {e}")
       finally:
           sock.close()
       time.sleep(5)


   try:
       intercom = VideoAudioIntercom(args)
       intercom.run()
   except KeyboardInterrupt:
       parser.exit("\nSIGINT received")
   finally:
       logging.info("Program terminated")






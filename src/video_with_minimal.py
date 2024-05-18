import signal
import argparse
import sounddevice as sd
import numpy as np
import socket
import time
import psutil
import logging
import soundfile as sf
import cv2
import threading
import sys
from queue import Queue
from threading import Event

FORMAT = "(%(levelname)s) %(module)s: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)

def spinning_cursor():
    ''' https://stackoverflow.com/questions/4995733/how-to-create-a-spinning-command-line-cursor'''
    while True:
        for cursor in '|/-\\':
            yield cursor

spinner = spinning_cursor()

def int_or_str(text):
    '''Helper function for argument parsing.'''
    try:
        return int(text)
    except ValueError:
        return text

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input-device", type=int_or_str, help="Input device ID or substring")
parser.add_argument("-o", "--output-device", type=int_or_str, help="Output device ID or substring")
parser.add_argument("-d", "--list-devices", action="store_true", help="Print the available audio devices and quit")
parser.add_argument("-s", "--frames_per_second", type=float, default=44100, help="sampling rate in frames/second")
parser.add_argument("-c", "--frames_per_chunk", type=int, default=4096, help="Number of frames in a chunk")
parser.add_argument("-l", "--listening_port", type=int, default=4444, help="My listening port")
parser.add_argument("-a", "--destination_address", type=int_or_str, default="localhost", help="Destination (interlocutor's listening) address")
parser.add_argument("-p", "--destination_port", type=int, default=4444, help="Destination (interlocutor's listening-) port")
parser.add_argument("-f", "--filename", type=str, help="Use a wav/oga/... file instead of the mic data")
parser.add_argument("-t", "--reading_time", type=int, help="Time reading data (mic or file) (only with effect if --show_stats or --show_data is used)")
parser.add_argument("--show_stats", action="store_true", help="shows bandwidth, CPU and quality statistics")
parser.add_argument("--show_samples", action="store_true", help="shows samples values")

class Minimal:
    NUMBER_OF_CHANNELS = 2

    def __init__(self, args):
        logging.info(__doc__)
        logging.info(f"NUMBER_OF_CHANNELS = {self.NUMBER_OF_CHANNELS}")
        self.sock_audio = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.listening_endpoint = ("0.0.0.0", args.listening_port)
        self.sock_audio.bind(self.listening_endpoint)
        self.chunk_time = args.frames_per_chunk / args.frames_per_second
        logging.info(f"chunk_time = {self.chunk_time} seconds")
        self.zero_chunk = self.generate_zero_chunk()

        self.video_capture = cv2.VideoCapture(0)
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        if args.filename:
            logging.info(f"Using \"{args.filename}\" as input")
            self.wavfile = sf.SoundFile(args.filename, 'r')
            self._handler = self._read_IO_and_play
            self.stream = self.file_stream
        else:
            self._handler = self._record_IO_and_play
            self.stream = self.mic_stream

    def pack_audio(self, audio_chunk):
        return audio_chunk.tobytes()

    def unpack_audio(self, packed_chunk):
        return np.frombuffer(packed_chunk, np.int16)

    def send_audio(self, packed_chunk):
        self.sock_audio.sendto(packed_chunk, (args.destination_address, args.destination_port))

    def receive_audio(self):
        audio_chunk_size = args.frames_per_chunk * self.NUMBER_OF_CHANNELS * np.dtype(np.int16).itemsize
        audio_packed, _ = self.sock_audio.recvfrom(audio_chunk_size)
        return audio_packed

    def generate_zero_chunk(self):
        return np.zeros((args.frames_per_chunk, self.NUMBER_OF_CHANNELS), np.int16)

    def _record_IO_and_play(self, ADC, DAC, frames, time, status):
        if __debug__:
            data = ADC.copy()
            packed_chunk = self.pack_audio(data)
        else:
            packed_chunk = self.pack_audio(ADC)
        self.send_audio(packed_chunk)

        try:
            packed_chunk = self.receive_audio()
            chunk = self.unpack_audio(packed_chunk)
        except (socket.timeout, BlockingIOError):
            chunk = self.zero_chunk
            logging.debug("playing zero chunk")

        chunk = chunk.reshape((int(len(chunk) / 2), 2))
        DAC[:] = chunk
        if __debug__:
            print(next(spinner), end='\b', flush=True)

    def _read_IO_and_play(self, DAC, frames, time, status):
        chunk = self.read_chunk_from_file()
        packed_chunk = self.pack(chunk)
        self.send(packed_chunk)
        try:
            packed_chunk = self.receive()
            chunk = self.unpack(packed_chunk)
        except (socket.timeout, BlockingIOError, ValueError):
            chunk = self.zero_chunk
            logging.debug("playing zero chunk")
        DAC[:] = chunk
        if __debug__:
            print(next(spinner), end='\b', flush=True)
        return chunk

    def mic_stream(self):
        try:
            with sd.InputStream(channels=self.NUMBER_OF_CHANNELS, dtype='int16', callback=self.audio_callback):
                with sd.OutputStream(samplerate=args.frames_per_second, blocksize=args.frames_per_chunk, dtype=np.int16, channels=self.NUMBER_OF_CHANNELS) as stream:
                    while True:
                        self.event.wait()
                        self.event.clear()
                        packed_audio = self.pack_audio(self.audio_data)
                        self.queue.put(packed_audio)
                        stream.write(self.audio_data)
        except Exception as e:
            print(f"Error in mic_stream: {e}")

    def audio_callback(self, indata, frames, time, status):
        self.audio_data = indata
        self.event.set()

    def video_stream(self):
        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            encoded, buffer = cv2.imencode('.jpg', frame)
            buffer = buffer.tobytes()
            max_packet_size = 65000
            for i in range(0, len(buffer), max_packet_size):
                self.video_socket.sendto(buffer[i:i + max_packet_size], (args.destination_address, args.destination_port + 1))
            cv2.imshow('Video Stream', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def file_stream(self):
        with sd.OutputStream(device=args.output_device, samplerate=args.frames_per_second, blocksize=args.frames_per_chunk, dtype=np.int16, channels=self.NUMBER_OF_CHANNELS, callback=self._handler):
            print("Press 'q' to quit...")
            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(0.1)

    def run(self):
        self.queue = Queue()
        self.event = Event()

        audio_thread = threading.Thread(target=self.mic_stream)
        video_thread = threading.Thread(target=self.video_stream)
        audio_thread.start()
        video_thread.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

        audio_thread.join()
        video_thread.join()

if __name__ == "__main__":
    args = parser.parse_args()

    if args.list_devices:
        logging.info(sd.query_devices())
        exit()

    intercom = Minimal(args)
    intercom.run()

import os
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
parser.add_argument("-c", "--frames_per_chunk", type=int, default=4096, help="Number of frames in a chunk")  # Adjusted chunk size
parser.add_argument("-l", "--listening_port", type=int, default=4444, help="My listening port")
parser.add_argument("-a", "--destination_address", type=int_or_str, default="localhost", help="Destination (interlocutor's listening) address")
parser.add_argument("-p", "--destination_port", type=int, default=4444, help="Destination (interlocutor's listing-) port")
parser.add_argument("-f", "--filename", type=str, help="Use a wav/oga/... file instead of the mic data")
parser.add_argument("-t", "--reading_time", type=int, help="Time reading data (mic or file) (only with effect if --show_stats or --show_data is used)")
parser.add_argument("--show_stats", action="store_true", help="shows bandwith, CPU and quality statistics")
parser.add_argument("--show_samples", action="store_true", help="shows samples values")
parser.add_argument("--video_width", type=int, default=640, help="Video width")
parser.add_argument("--video_height", type=int, default=480, help="Video height")
parser.add_argument("--video_fps", type=float, default=30, help="Video frames per second")
parser.add_argument("--video_chunk_size", type=int, default=1024, help="Size of video chunk in bytes")

class Minimal:
    NUMBER_OF_CHANNELS = 2  # The number of audio channels. Currently, in OSX systems NUMBER_OF_CHANNELS must be 1.

    def __init__(self, args):
        ''' Constructor. Basically initializes the sockets stuff. '''
        logging.info(__doc__)
        logging.info(f"NUMBER_OF_CHANNELS = {self.NUMBER_OF_CHANNELS}")
        self.sock_audio = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.listening_endpoint = ("0.0.0.0", args.listening_port)
        self.sock_audio.bind(self.listening_endpoint)
        self.sock_video.bind(("0.0.0.0", args.listening_port + 1))
        self.chunk_time = args.frames_per_chunk / args.frames_per_second
        logging.info(f"chunk_time = {self.chunk_time} seconds")
        self.zero_chunk = self.generate_zero_chunk()

        if args.filename:
            logging.info(f"Using \"{args.filename}\" as input")
            self.wavfile = sf.SoundFile(args.filename, 'r')
            self._handler = self._read_IO_and_play
            self.stream = self.file_stream
        else:
            self._handler = self._record_IO_and_play
            self.stream = self.mic_stream

    def pack_audio(self, audio_chunk):
        '''Builds a packet's payload with an audio chunk.'''
        return audio_chunk.tobytes()

    def pack_video(self, video_chunk):
        '''Builds a list of packets with video chunks, each packet with a size that fits the system's limit, accounting for IP and UDP headers.'''
        packed_chunks = []
        max_packet_size = 65507  # Máximo tamaño de paquete UDP teniendo en cuenta las cabeceras IP y UDP
        chunk_size = args.video_chunk_size
        height, width, _ = video_chunk.shape
        num_chunks = (height * width * 3) // chunk_size
        remainder = (height * width * 3) % chunk_size
        if remainder > 0:
            num_chunks += 1
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, height * width * 3)  # Asegurarse de que el final no exceda el tamaño del array
            packed_chunks.append(video_chunk[start:end].tobytes())

        # Ajustar el tamaño de los paquetes si superan el límite real
        for i in range(len(packed_chunks)):
            if len(packed_chunks[i]) > max_packet_size:
                packed_chunks[i] = packed_chunks[i][:max_packet_size]
        return packed_chunks

    def unpack_audio(self, packed_chunk):
        '''Unpack an audio packed_chunk.'''
        return np.frombuffer(packed_chunk, np.int16)

    def unpack_video(self, packed_chunk):
        '''Unpack a video packed_chunk.'''
        return np.frombuffer(packed_chunk, np.uint8)

    def send_audio(self, packed_chunk):
        '''Sends an UDP packet with audio payload.'''
        self.sock_audio.sendto(packed_chunk, (args.destination_address, args.destination_port))

    def send_video(self, packed_chunks):
        '''Sends UDP packets with video payload.'''
        for chunk in packed_chunks:
            self.sock_video.sendto(chunk, (args.destination_address, args.destination_port + 1))

    def receive_audio(self):
        '''Receives an UDP packet with audio payload without blocking.'''
        audio_chunk_size = args.frames_per_chunk * self.NUMBER_OF_CHANNELS * np.dtype(np.int16).itemsize
        audio_packed, _ = self.sock_audio.recvfrom(audio_chunk_size)
        return audio_packed

    def receive_video(self):
        '''Receives an UDP packet with video payload without blocking.'''
        video_chunk_size = args.video_chunk_size
        video_packed, _ = self.sock_video.recvfrom(video_chunk_size)
        return video_packed

    def generate_zero_chunk(self):
        '''Generates a chunk with zeros that will be used when an inbound chunk is not available.'''
        return np.zeros((args.frames_per_chunk, self.NUMBER_OF_CHANNELS), np.int16)

    def _record_IO_and_play(self, ADC, DAC, frames, time, status):
        '''Interruption handler that samples a chunk, builds a packet with the
        chunk, sends the packet, receives a packet, unpacks it to get
        a chunk, and plays the chunk.
        '''
        video_capture = cv2.VideoCapture(0)
        ret, video_frame = video_capture.read()  

        if ret:  
            video_frame = cv2.resize(video_frame, (args.video_width, args.video_height))
            video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
            video_chunk = np.asarray(video_frame)
        else:
            video_chunk = np.zeros((args.video_height, args.video_width, 3), dtype=np.uint8)  

        audio_packed = self.pack_audio(ADC)
        video_chunks = self.pack_video(video_chunk)
        for chunk in video_chunks:
            self.send_audio(audio_packed)
            self.send_video([chunk])
        
        try:
            audio_packed = self.receive_audio()
            audio_chunk = self.unpack_audio(audio_packed)

            # Ajustar el tamaño del chunk de audio si es necesario
            expected_audio_chunk_size = args.frames_per_chunk * self.NUMBER_OF_CHANNELS
            if len(audio_chunk) != expected_audio_chunk_size:
                logging.warning(f"Received audio chunk size ({len(audio_chunk)}) does not match expected size ({expected_audio_chunk_size}). Adjusting...")
                audio_chunk = audio_chunk[:expected_audio_chunk_size]
        except (socket.timeout, BlockingIOError):
            audio_chunk = self.zero_chunk
            logging.debug("playing zero chunk")
        
        # Redimensionar el chunk de audio para que coincida con el tamaño esperado
        audio_chunk = audio_chunk.reshape((args.frames_per_chunk, self.NUMBER_OF_CHANNELS))

        # Asignar el chunk de audio a DAC
        DAC[:] = audio_chunk
        
        if __debug__:
            print(next(spinner), end='\b', flush=True)

    def read_chunk_from_file(self):
        chunk = self.wavfile.buffer_read(args.frames_per_chunk, dtype='int16')
        if len(chunk) < args.frames_per_chunk * 4:
            logging.warning("Input exhausted! :-/")
            pid = os.getpid()
            os.kill(pid, signal.SIGINT)
            return self.zero_chunk
        chunk = np.frombuffer(chunk, dtype=np.int16)
        chunk = np.reshape(chunk, (args.frames_per_chunk, self.NUMBER_OF_CHANNELS))
        return chunk
            
    def _read_IO_and_play(self, DAC, frames, time, status):
        '''Similar to _record_IO_and_play, but the recorded chunk is
        obtained from the mic (instead of being provided by the
        interruption handler).
        '''
        if __debug__:
            ADC = self.read_chunk_from_file()
            audio_packed = self.pack_audio(ADC)
            video_packed = self.pack_video(np.zeros((args.video_height, args.video_width, 3), dtype=np.uint8))
            self.send_audio(audio_packed)
            self.send_video(video_packed)
        else:
            audio_packed = self.pack_audio(DAC)
            video_packed = self.pack_video(np.zeros((args.video_height, args.video_width, 3), dtype=np.uint8))
            self.send_audio(audio_packed)
            self.send_video(video_packed)
        
        try:
            audio_packed = self.receive_audio()
            audio_chunk = self.unpack_audio(audio_packed)

            # Ajustar el tamaño del chunk de audio si es necesario
            expected_audio_chunk_size = args.frames_per_chunk * self.NUMBER_OF_CHANNELS
            if len(audio_chunk) != expected_audio_chunk_size:
                logging.warning(f"Received audio chunk size ({len(audio_chunk)}) does not match expected size ({expected_audio_chunk_size}). Adjusting...")
                audio_chunk = audio_chunk[:expected_audio_chunk_size]
        except (socket.timeout, BlockingIOError):
            audio_chunk = self.zero_chunk
            logging.debug("playing zero chunk")
        
        # Redimensionar el chunk de audio para que coincida con el tamaño esperado
        audio_chunk = audio_chunk.reshape((args.frames_per_chunk, self.NUMBER_OF_CHANNELS))

        # Asignar el chunk de audio a DAC
        DAC[:] = audio_chunk
        if __debug__:
            print(next(spinner), end='\b', flush=True)

    def mic_stream(self):
        '''Generates an output stream from the audio card.'''
        with sd.Stream(device=(args.input_device, args.output_device), samplerate=args.frames_per_second, blocksize=args.frames_per_chunk, dtype=np.int16, latency='low', channels=self.NUMBER_OF_CHANNELS, callback=self._handler):
            print("Press 'q' to quit...")
            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(0.1)  # Añadir un pequeño retraso para reducir el uso de la CPU


    def file_stream(self):
        '''Generates an output stream from a file.'''
        with sd.OutputStream(device=args.output_device, samplerate=args.frames_per_second, blocksize=args.frames_per_chunk, dtype=np.int16, channels=self.NUMBER_OF_CHANNELS, callback=self._handler):
            print("Press 'q' to quit...")
            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(0.1)  # Añadir un pequeño retraso para reducir el uso de la CPU

    def run(self):
        '''Starts sending the playing chunks.'''
        if not args.filename:
            self.cap = cv2.VideoCapture(0)
        self.stream()
        if not args.filename:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parser.parse_args()

    if args.list_devices:
        logging.info(sd.query_devices())
        exit()

    intercom = Minimal(args)
    intercom.run()

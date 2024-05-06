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

class Minimal:
    MAX_PAYLOAD_BYTES = 32768  # The maximum UDP packet's payload.
    NUMBER_OF_CHANNELS = 2  # The number of channels. Currently, in OSX systems NUMBER_OF_CHANNELS must be 1.

    def __init__(self, args):
        ''' Constructor. Basically initializes the sockets stuff. '''
        logging.info(__doc__)
        logging.info(f"NUMBER_OF_CHANNELS = {self.NUMBER_OF_CHANNELS}")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.listening_endpoint = ("0.0.0.0", args.listening_port)
        self.sock.bind(self.listening_endpoint)
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

    def pack(self, audio_chunk, video_chunk):
        '''Builds a packet's payloads with a chunk.'''
        audio_packed = audio_chunk.tobytes()
        video_packed = video_chunk.tobytes()
        return audio_packed + video_packed

    def unpack(self, packed_chunk):
        '''Unpack a packed_chunk.'''
        audio_chunk_length = len(packed_chunk) // 2  # Assuming audio and video are of the same length
        audio_chunk = np.frombuffer(packed_chunk[:audio_chunk_length], np.int16)
        video_chunk = np.frombuffer(packed_chunk[audio_chunk_length:], np.int16)
        return audio_chunk, video_chunk
    
    def send(self, packed_chunk):
        '''Sends an UDP packet.'''
        audio_chunk_length = len(packed_chunk) // 2  # Assuming audio and video are of the same length
        audio_packed = packed_chunk[:audio_chunk_length]
        video_packed = packed_chunk[audio_chunk_length:]
        self.sock.sendto(audio_packed, (args.destination_address, args.destination_port))
        self.sock.sendto(video_packed, (args.destination_address, args.destination_port))

    def receive(self):
        '''Receives an UDP packet without blocking.'''
        audio_chunk_size = args.frames_per_chunk * self.NUMBER_OF_CHANNELS * np.dtype(np.int16).itemsize
        audio_packed, _ = self.sock.recvfrom(audio_chunk_size)
        video_packed, _ = self.sock.recvfrom(audio_chunk_size)  # Assuming video chunk size is the same as audio chunk size
        return audio_packed + video_packed

    def generate_zero_chunk(self):
        '''Generates a chunk with zeros that will be used when an inbound chunk is not available.'''
        return np.zeros((args.frames_per_chunk, self.NUMBER_OF_CHANNELS), np.int16)

    def _record_IO_and_play(self, ADC, DAC, frames, time, status):
        '''Interruption handler that samples a chunk, builds a packet with the
        chunk, sends the packet, receives a packet, unpacks it to get
        a chunk, and plays the chunk.
        '''
        if __debug__:
            data = ADC.copy()
            video_chunk = np.zeros_like(data)  # Create a video chunk with zeros
            packed_chunk = self.pack(data, video_chunk)  # Pass both audio and video chunks
        else:
            packed_chunk = self.pack(ADC, np.zeros_like(ADC))  # Assuming video is all zeros
        self.send(packed_chunk)
        try:
            packed_chunk = self.receive()
            audio_chunk, _ = self.unpack(packed_chunk)
            # Reshape audio_chunk to (4096, 2)
            audio_chunk = np.reshape(audio_chunk, (args.frames_per_chunk, self.NUMBER_OF_CHANNELS))
        except (socket.timeout, BlockingIOError):
            audio_chunk = self.zero_chunk
            logging.debug("playing zero chunk")
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
            packed_chunk = self.pack(ADC, np.zeros_like(ADC))  # Assuming video is all zeros
        else:
            packed_chunk = self.pack(DAC, np.zeros_like(DAC))  # Assuming video is all zeros
        self.send(packed_chunk)
        try:
            packed_chunk = self.receive()
            audio_chunk, _ = self.unpack(packed_chunk)
        except (socket.timeout, BlockingIOError):
            audio_chunk = self.zero_chunk
            logging.debug("playing zero chunk")
        DAC[:] = audio_chunk
        if __debug__:
            print(next(spinner), end='\b', flush=True)

    def mic_stream(self):
        '''Generates an output stream from the audio card.'''
        with sd.Stream(device=(args.input_device, args.output_device), samplerate=args.frames_per_second, blocksize=args.frames_per_chunk, dtype=np.int16, latency='low', channels=self.NUMBER_OF_CHANNELS, callback=self._handler):
            input("Press Enter to quit...")
        
    import cv2

    def video_stream(self):
        '''Generates a video stream from the camera.'''
        cap = cv2.VideoCapture(0)  # Open default camera (you might need to adjust the index if you have multiple cameras)
        while True:
            ret, frame = cap.read()  # Read a frame from the camera
            if not ret:
                print("Error: Unable to capture frame")
                break
            # Send the frame to the other end of the communication (you need to implement this)
            # For example, you could convert the frame to bytes and send it using sockets
            # Implement your sending logic here
            cv2.imshow('Video Stream', frame)  # Display the frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Press 'q' to quit
        cap.release()
        cv2.destroyAllWindows()

    def file_stream(self):
        '''Generates an output stream from a file.'''
        with sd.OutputStream(device=args.output_device, samplerate=args.frames_per_second, blocksize=args.frames_per_chunk, dtype=np.int16, channels=self.NUMBER_OF_CHANNELS, callback=self._handler):
            input("Press Enter to quit...")
                
    def run(self):
        '''Starts sending the playing chunks.'''
        self.stream()

def check_cameras_available():
    cam = cv2.VideoCapture(0)
    ret, _ = cam.read()
    while True:
        if ret:
            ret, _ = cam.read()
            if not ret:
                print("Error: Unable to capture frame")
                break
            cv2.imshow('Video Stream', _)  # Muestra el frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Presiona 'q' para salir
    cam.release()
    cv2.destroyAllWindows()
    return ret



if __name__ == "__main__":
    args = parser.parse_args()

    if args.list_devices:
        logging.info(sd.query_devices())
        exit()

    # Verificar si hay cámaras disponibles
    if not check_cameras_available():
        print("No se detectaron cámaras disponibles.")
        exit()

    intercom = Minimal(args)
    intercom.run()

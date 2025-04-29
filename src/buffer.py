#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""Over minimal_video, implements a random access buffer structure for hiding the jitter."""

import argparse
import sounddevice as sd
import numpy as np
import socket
import time
import psutil
import math
import struct
import threading
import minimal_video
import soundfile as sf
import logging
import queue

# --- Usamos el parser ya definido en minimal_video ---
if not hasattr(minimal_video, 'parser'):
    minimal_video.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# --- Añadir argumentos específicos para el buffering ---
minimal_video.parser.add_argument("-b", "--buffering_time", type=int, default=150, help="Miliseconds to buffer")

class Buffering(minimal_video.Minimal_Video):
    CHUNK_NUMBERS = 1 << 15  # Suficiente para la mayoría de los tiempos de buffering

    def __init__(self):
        # Llama al constructor de Minimal_Video (que ya habilita video por defecto)
        super().__init__()
        logging.info(__doc__)
        if minimal_video.args.buffering_time <= 0:
            minimal_video.args.buffering_time = 1  # ms
        logging.info(f"buffering_time = {minimal_video.args.buffering_time} miliseconds")
        self.chunks_to_buffer = int(math.ceil(minimal_video.args.buffering_time / 1000 / self.chunk_time))
        self.zero_chunk = self.generate_zero_chunk()
        self.cells_in_buffer = self.chunks_to_buffer * 2
        self._buffer = [None] * self.cells_in_buffer
        for i in range(self.cells_in_buffer):
            self._buffer[i] = self.zero_chunk
        self.chunk_number = 0
        logging.info(f"chunks_to_buffer = {self.chunks_to_buffer}")

        if minimal_video.args.filename:
            logging.info(f"Using \"{minimal_video.args.filename}\" as input")
            self.wavfile = sf.SoundFile(minimal_video.args.filename, 'r')
            self._handler = self._read_IO_and_play
            self.stream = self.file_stream
        else:
            self._handler = self._record_IO_and_play
            self.stream = self.mic_stream

    def pack(self, chunk_number, chunk):
        """Concatenates a chunk number to the chunk."""
        packed_chunk = struct.pack("!H", chunk_number) + chunk.tobytes()
        return packed_chunk

    def unpack(self, packed_chunk):
        """Splits the packed chunk into a chunk number and a chunk."""
        (chunk_number,) = struct.unpack("!H", packed_chunk[:2])
        chunk = packed_chunk[2:]
        chunk = np.frombuffer(chunk, dtype=np.int16)
        return chunk_number, chunk

    def buffer_chunk(self, chunk_number, chunk):
        self._buffer[chunk_number % self.cells_in_buffer] = chunk

    def unbuffer_next_chunk(self):
        return self._buffer[self.played_chunk_number % self.cells_in_buffer]

    def play_chunk(self, DAC, chunk):
        self.played_chunk_number = (self.played_chunk_number + 1) % self.cells_in_buffer
        chunk = chunk.reshape(minimal_video.args.frames_per_chunk, minimal_video.args.number_of_channels)
        DAC[:] = chunk

    def receive(self):
        packed_chunk, sender = self.sock.recvfrom(self.MAX_PAYLOAD_BYTES)
        return packed_chunk

    def receive_and_buffer(self):
        # Se muestra el spinner, heredado del módulo minimal_video
        print(next(minimal_video.spinner), end='\b', flush=True)
        packed_chunk = self.receive()
        chunk_number, chunk = self.unpack(packed_chunk)
        self.buffer_chunk(chunk_number, chunk)
        return chunk_number

    def _record_IO_and_play(self, ADC, DAC, frames, time_info, status):
        self.chunk_number = (self.chunk_number + 1) % self.CHUNK_NUMBERS
        packed_chunk = self.pack(self.chunk_number, ADC)
        self.send(packed_chunk)
        chunk = self.unbuffer_next_chunk()
        self.play_chunk(DAC, chunk)

    def _read_IO_and_play(self, DAC, frames, time_info, status):
        self.chunk_number = (self.chunk_number + 1) % self.CHUNK_NUMBERS
        read_chunk = self.read_chunk_from_file()
        packed_chunk = self.pack(self.chunk_number, read_chunk)
        self.send(packed_chunk)
        chunk = self.unbuffer_next_chunk()
        self.play_chunk(DAC, chunk)
        return read_chunk

    def run(self):
        logging.info("Press CTRL+c to quit")
        self.played_chunk_number = 0

        # Inicia el hilo de captura de vídeo (si la captura está habilitada)
        if self.capture_enabled:
            t_capture = threading.Thread(target=self.capture_video_loop, daemon=True, name="CaptureThread")
            t_capture.start()

        # Inicia también el hilo de visualización de vídeo
        if self.capture_enabled:
            t_display = threading.Thread(target=self.display_video_loop, daemon=True, name="DisplayThread")
            t_display.start()

        # Luego, abre el stream de audio/asociado al buffering
        with self.stream(self._handler):
            first_received_chunk_number = self.receive_and_buffer()
            logging.debug("first_received_chunk_number = %s", first_received_chunk_number)
            # Posiciona el puntero en el buffer justo en el frame adecuado
            self.played_chunk_number = (first_received_chunk_number - self.chunks_to_buffer) % self.cells_in_buffer

            # Bucle principal de buffering de audio (y de vídeo recibido, si es que corresponde)
            while True:
                self.receive_and_buffer()


class Buffering__verbose(Buffering, minimal_video.Minimal_Video__verbose):
    def __init__(self):
        super().__init__()

    def send(self, packed_chunk):
        Buffering.send(self, packed_chunk)
        self.sent_bytes_count += len(packed_chunk)
        self.sent_messages_count += 1

    def receive(self):
        packed_chunk = Buffering.receive(self)
        self.received_bytes_count += len(packed_chunk)
        self.received_messages_count += 1
        return packed_chunk

    def _record_IO_and_play(self, ADC, DAC, frames, time_info, status):
        if minimal_video.args.show_samples:
            self.show_recorded_chunk(ADC)
        super()._record_IO_and_play(ADC, DAC, frames, time_info, status)
        if minimal_video.args.show_samples:
            self.show_played_chunk(DAC)
        self.recorded_chunk = DAC
        self.played_chunk = ADC

    def _read_IO_and_play(self, DAC, frames, time_info, status):
        read_chunk = super()._read_IO_and_play(DAC, frames, time_info, status)
        if minimal_video.args.show_samples:
            self.show_recorded_chunk(read_chunk)
            self.show_played_chunk(DAC)
        self.recorded_chunk = DAC
        return read_chunk

    def loop_receive_and_buffer(self):
        first_received_chunk_number = self.receive_and_buffer()
        print("first_received_chunk_number =", first_received_chunk_number)
        self.played_chunk_number = (first_received_chunk_number - self.chunks_to_buffer) % self.cells_in_buffer
        if minimal_video.args.show_spectrum:
            while self.total_number_of_sent_chunks < self.chunks_to_send:
                self.receive_and_buffer()
                self.update_display()
        else:
            while self.total_number_of_sent_chunks < self.chunks_to_send:
                self.receive_and_buffer()

    def run(self):
        cycle_feedback_thread = threading.Thread(target=self.loop_cycle_feedback)
        cycle_feedback_thread.daemon = True        
        self.print_running_info()
        super().print_header()
        self.played_chunk_number = 0

         # Inicia el hilo de captura de vídeo (si la captura está habilitada)
        if self.capture_enabled:
            t_capture = threading.Thread(target=self.capture_video_loop, daemon=True, name="CaptureThread")
            t_capture.start()

        # Inicia también el hilo de visualización de vídeo
        if self.capture_enabled:
            t_display = threading.Thread(target=self.display_video_loop, daemon=True, name="DisplayThread")
            t_display.start()


        with self.stream(self._handler):
            cycle_feedback_thread.start()
            self.loop_receive_and_buffer()


try:
    import argcomplete
except ImportError:
    logging.warning("Unable to import argcomplete (optional)")

if __name__ == "__main__":
    minimal_video.parser.description = __doc__
    try:
        argcomplete.autocomplete(minimal_video.parser)
    except Exception:
        logging.warning("argcomplete not working :-/")
    minimal_video.args = minimal_video.parser.parse_known_args()[0]
    
    if minimal_video.args.list_devices:
        print("Available devices:")
        print(sd.query_devices())
        quit()

    if minimal_video.args.show_stats or minimal_video.args.show_samples or minimal_video.args.show_spectrum:
        intercom_app = Buffering__verbose()
    else:
        intercom_app = Buffering()

    try:
        intercom_app.run()
    except KeyboardInterrupt:
        minimal_video.parser.exit("\nSIGINT received")
    finally:
        intercom_app.print_final_averages()

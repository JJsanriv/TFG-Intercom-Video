#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""Over minimal, implements the use of a video stream while minimal runs normally."""

import minimal

minimal.parser.add_argument("-v", "--video", type=str, default=None, help="Video streaming activation")

class MinimalVideo(minimal.Minimal):
    """MinimalVideo class, inherits from Minimal class."""





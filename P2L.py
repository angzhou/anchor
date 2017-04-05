# coding: utf-8
"""
convert image file to L (8bit black and white) mode
"""

import sys
import os.path
from PIL import Image

try:
    img = Image.open(sys.argv[1])
except (FileNotFoundError, OSError, IndexError):
    exit("need a image file")

if img.mode == 'L':
    exit("image is already 8 bit B&W")

img = img.convert(mode='L')
name, ext = os.path.splitext(sys.argv[1])
img.save("%s_0.png" % name)

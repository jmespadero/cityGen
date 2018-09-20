#!/bin/sh
#Use external python3 interpreter, to create debug svg images.
python.exe cityGen2D.py -s 16 -m Temple

#Use blender internal python3 interpreter (wont generate .svg files)
#blender --background --python cityGen2D.py -s 13

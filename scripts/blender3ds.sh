#!/bin/bash
fn=`readlink -f "$1"`
echo Importing $fn
echo "import bpy" > tmp.py
echo "bpy.ops.object.delete()" >> tmp.py
echo "bpy.ops.import_scene.autodesk_3ds(filepath = \"$fn\", axis_forward='-Y', axis_up='Z')" >> tmp.py
blender --python tmp.py
rm -f tmp.py


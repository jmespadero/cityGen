"""
Launcher for the external script "cg-GenGame01.py"
"""
import bpy, os
#Execute an external script
#https://www.blender.org/api/blender_python_api_2_69_release/info_tips_and_tricks.html
cwd = os.path.dirname(bpy.data.filepath)
script = os.path.join(cwd, "cg-GenGame01.py")
print("Exec %s" % script)
exec(compile(open(script).read(), script, 'exec'))



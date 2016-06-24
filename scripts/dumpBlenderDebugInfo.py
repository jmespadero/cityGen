import bpy
from pprint import pprint

#Dump a text file with debug info about resource usage in a blender file
#For blender 2.6 , 2.7,  and above.
            
#Create dictionaries
texturesUsingImage={}
imagesUsedByTexture={}
materialsUsingTexture={}
texturesUsedByMaterial={}
objectsUsingMaterial={}
materialsUsedByObject={}

#Initialize dictionaries 
for img in bpy.data.images :
    texturesUsingImage[img.name] = []

for tx in bpy.data.textures :
    materialsUsingTexture[tx.name] = []
    imagesUsedByTexture[tx.name] = []

for mat in bpy.data.materials:
    texturesUsedByMaterial[mat.name] = []
    objectsUsingMaterial[mat.name] = []

for o in bpy.data.objects :
    materialsUsedByObject[o.name]=[]

#("Texture - Image usage:")
for tx in bpy.data.textures :
    if tx.image:
        texturesUsingImage[tx.image.name].append(tx.name)
        imagesUsedByTexture[tx.name].append(tx.image.name)

#("Materials - Texture usage:")
for mat in bpy.data.materials:
    for tx in mat.texture_slots:
        if tx:
            materialsUsingTexture[tx.name].append(mat.name)
            texturesUsedByMaterial[mat.name].append(tx.name)


#("Objects - Material usage:")
for o in bpy.data.objects:
    for mat in o.material_slots:
        objectsUsingMaterial[mat.name].append(o.name)
        materialsUsedByObject[o.name].append(mat.name)

print("texturesUsingImage")
pprint(texturesUsingImage)

print("imagesUsedByTexture")
pprint(imagesUsedByTexture)

print("materialsUsingTexture")
pprint(materialsUsingTexture)

print("texturesUsedByMaterial")
pprint(texturesUsedByMaterial)

print("objectsUsingMaterial")
pprint(objectsUsingMaterial)

print("materialsUsedByObject")
pprint(materialsUsedByObject)

"""
#Purge Unused images
for img in bpy.data.images:
    if texturesUsingImage[img.name] == [] :
        print('Purge image ', img.name)
        img.user_clear()
"""

print("Size of packed images")
for img in bpy.data.images :
   print(img.name, img.packed_file.size)

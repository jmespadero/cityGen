import bpy

#Remove duplicated materials and textures in blender scenes
#For blender 2.6 , 2.7,  and above.

#Build a list of materials names (except dups)
matNames = []
for mat in bpy.data.materials:
    if not (len(mat.name) > 4 and mat.name[-3:].isdigit() and mat.name[-4] == '.' ):
        matNames.append(mat.name)
        
#Rename materials that are dups of materials no longer in scene
for mat in bpy.data.materials:
    if len(mat.name) > 4 and mat.name[-3:].isdigit() and mat.name[-4] == '.' :
        baseMat = mat.name[:-4]
        if not baseMat in matNames:
            print('Rename material ', mat.name, " to ", baseMat)
            mat.name = baseMat
            matNames.append(mat.name)

#Ask objects to use original materials, instead the dups
for o in bpy.data.objects:
    for mat in o.material_slots:
        name = mat.name
        if len(name) > 4 and name[-3:].isdigit() and name[-4] == '.' :
            base = name[:-4]
            if base in matNames:
                print('Change object', o.name, "material", name, "to", base)
                o.material_slots[name].material = bpy.data.materials[base]
            
#Build a list of texture names (except dups)
txNames = []
for tx in bpy.data.textures:
    name = tx.name
    if not(len(name) > 4 and name[-3:].isdigit() and name[-4] == '.') :
        txNames.append(name)

#Rename texture that are dups of textures no longer in scene
for tx in bpy.data.textures:
    name = tx.name
    if len(name) > 4 and name[-3:].isdigit() and name[-4] == '.' :
        base = name[:-4]
        if not base in txNames:
            print('Rename texture', name, ' to ', base)
            tx.name = base
            txNames.append(base)

#Ask materials to use original textures, instead the dups
for mat in bpy.data.materials:
    for tx in mat.texture_slots:
        if tx:
            name = tx.name
            if len(name) > 4 and name[-3:].isdigit() and name[-4] == '.' :
                base = name[:-4]
                if base in txNames:
                    print('Change material', mat.name, "texture", name, "to", base)
                    mat.texture_slots[name].texture = bpy.data.textures[base]

#Build a list of image names (except dups)
imgNames = []
for img in bpy.data.images:
    name = img.name
    if not(len(name) > 4 and name[-3:].isdigit() and name[-4] == '.') :
        imgNames.append(name)

#Rename images that are dups of images no longer in scene
for img in bpy.data.images:
    name = img.name
    if len(name) > 4 and name[-3:].isdigit() and name[-4] == '.' :
        base = name[:-4]
        if not base in imgNames:
            print('Rename image ', name, ' to ', base)
            img.name = base
            imgNames.append(img.name)

#Ask textures to use original images, instead the dups
for tx in bpy.data.textures:
    if tx.image:
        name = tx.image.name
        if len(name) > 4 and name[-3:].isdigit() and name[-4] == '.' :
            base = name[:-4]
            if base in imgNames:
                print('Change texture ', tx.name, " image ", name, " to ", base)
                tx.image = bpy.data.images[base]

##BEWARE OF THIS!!!

#Clear textures ending in numbers.
for tx in bpy.data.textures:
    name = tx.name
    if len(name) > 4 and name[-3:].isdigit() and name[-4] == '.' :
        print('Purge texture', tx.name)
        tx.user_clear()

#Clear images ending in numbers.
for img in bpy.data.images:
    if len(img.name) > 4 and img.name[-3:].isdigit() and img.name[-4] == '.' :
        print('Purge image ', img.name)
        img.user_clear()

"""
Game generator from project citygen
Reads a .json file generated with cityGen2D.py, and build a 3D model of the city

Copyright 2014 Jose M. Espadero <josemiguel.espadero@urjc.es>
Copyright 2014 Juan Ramos <juanillo07@gmail.com>

Run option 1:
blender --background --python cityGen3D.py

Run option 2:
Open blender and type this in the python console:
  exec(compile(open("cityGen3D.py").read(), "cityGen3D.py", 'exec'))

"""

"""
TODO:
  * Add cg-temple to cities.
  * Build random houses. Create non-rectangular houses for corners.
  * Procedural generation of starred night sky.
  * Add roads outside of the city (at least one in near the gate)
DONE:
  * When importing a blend file (cg-library, etc...) append its "readme.txt" to
    a "readme.txt" in the output. This will honor all the CC-By resources.
  * Fix armatures when set position to an armatured object (see initPos)
"""
import bpy
import math, json, random, os, sys
from math import sqrt, acos, sin, cos
from pprint import pprint
from mathutils import Vector
from datetime import datetime
from random import uniform

#Set default values for args
args={
'cleanLayer0' : True,       # Clean all objects in layer 0
'createGlobalLight' : True,         # Add new light to scene
'inputFilename' : 'city.data.json',  # Set a filename to read 2D city map data
'inputFilenameAI' : 'city.AI.json',   # Set a filename to read AI data
'inputLibraries' : 'cg-library.blend',  # Set a filename for assets (houses, wall, etc...) library.
'inputHouses' : ["House7", "House3","House4","House5","House6"],
'inputPlayer' : 'cg-playerBoy.blend',   # Set a filename for player system.
'createDefenseWall' : True,  # Create exterior boundary of the city
'createGround' : True,       # Create ground boundary of the city
'createStreets' : True,      # Create streets of the city
'createLeaves' : True,       # Create leaves on the streets
'numMonsters' : 4,
'outputCityFilename' : 'outputcity.blend', #Output file with just the city
'outputTourFilename' : 'outputtour.blend', #Output file with complete game
'outputGameFilename' : 'outputgame.blend', #Output file with complete game
}
               
#################################################################
# Functions to create a new cityMap scene (does need run inside blender)

def duplicateObject(sourceObj, objName="copy", select=False, scene=bpy.context.scene):
    """Duplicate a object in the scene.
    sourceObj -- the blender obj to be copied
    objName   -- the name of the new object
    scene     -- the blender scene where the new object will be linked
    """

    # Create new mesh
    #mesh = bpy.data.meshes.new(objName)
    #ob_new = bpy.data.objects.new(objName, mesh)
    #ob_new.data = sourceObj.data.copy()
    #ob_new.scale = sourceObj.scale
    ob_new = sourceObj.copy()

    # Link new object to the given scene and select it
    ob_new.name = objName
    scene.objects.link(ob_new)
    ob_new.select = select

    return ob_new


def duplicateAlongSegment(pt1, pt2, objName, gapSize, force=False):
    """Duplicate an object several times along a path
    pt1 -- First extreme of the path
    pt2 -- Second extreme of the path
    objName -- the name of blender obj to be copied
    gapSize -- Desired space between objects. Will be adjusted to fit path
    """

    #Compute the orientation of segment pt1-pt2
    dx = pt1[0]-pt2[0]
    dy = pt1[1]-pt2[1]

    # return if pt1 == pt2
    if dx == 0 and dy == 0:
        return []

    # Compute the angle with the Y-axis
    ang = acos(dy/sqrt((dx**2)+(dy**2)))
    if dx > 0:
        ang = -ang

    # Get the size of the replicated object in the Y dimension
    ob = bpy.data.objects[objName]
    objSize = (ob.bound_box[7][1]-ob.bound_box[0][1])*ob.scale[1]
    totalSize = objSize+gapSize

    # Compute the direction of the segment
    pathVec = Vector(pt2)-Vector(pt1)
    pathLen = pathVec.length
    pathVec.normalize()

    if (objSize > pathLen):
        return []

    #if gapSize is not zero, change the gap to one that adjust the object
    #Compute the num of (obj+gap) segments in the interval (pt1-pt2)
    if gapSize != 0:
        numObj = round(pathLen/totalSize)
        step = pathLen/numObj
        stepVec = pathVec * step
        iniPoint = Vector(pt1)+(stepVec * 0.5)
    else:
        numObj = math.floor(pathLen/objSize)
        step = objSize
        stepVec = pathVec * step
        delta = pathLen-step*numObj #xke? (delta es el espacio que falta para completar una fila)
        iniPoint = Vector(pt1)+(stepVec*0.5) #se multiplicaba esto por delta, xke?
        

    #Duplicate the object along the path, numObj times
    iniPoint.resize_3d()
    stepVec.resize_3d()
    objList=[]
    if force:
        numObj = numObj - 1
    for i in range(numObj):
        loc = iniPoint + stepVec * i
        g1 = duplicateObject(ob, "_%s" % (objName))
        g1.rotation_euler = (0, 0, ang)
        g1.location = loc
        objList.append(g1)
    if force:
            loc = Vector(pt2) - stepVec * 0.5
            g1 = duplicateObject(ob, "_%s" % (objName))
            g1.rotation_euler = (0, 0, ang)
            g1.location = loc
            objList.append(g1)

    return objList

 
def knapsack_unbounded_dp(items, C, maxofequalhouse):
    NAME, SIZE, VALUE = range(3)
    # order by max value per item size
    C=int(C*10)
    #print(C)
    #items = sorted(items, key=lambda items: (items[1]), reverse=True)
 
    # Sack keeps track of max value so far as well as the count of each item in the sack
    sack = [(0, [0 for i in items]) for i in range(0, C+1)]   # value, [item counts]
 
    for i,item in enumerate(items):
        name, size, value = item
        for c in range(size, C+1):
            sackwithout = sack[c-size]  # previous max sack to try adding this item to
            trial = sackwithout[0] + value
            used = sackwithout[1][i]
            if sack[c][0] < trial:
                # old max sack with this added item is better
                sackaux=sack[c]
                sack[c] = (trial, sackwithout[1][:])
                if i!= len(items)-1:
                    if sack [c][1][i]<maxofequalhouse:
                        sack[c][1][i] +=1   # use one more
                    else:
                        sack[c]=sackaux 
                        break
                else:
                    sack[c][1][i] +=1
        else:
            continue               
                       
                    
 
    value, bagged = sack[C]
    numbagged = sum(bagged)
    size = sum(items[i][1]*n for i,n in enumerate(bagged))
    # convert to (iten, count) pairs) in name order
    bagged = sorted((items[i][NAME], n) for i,n in enumerate(bagged) if n)
    
    
    return value, size, numbagged, bagged

def knapsack_unbounded_dp_control(pathLen, gapSize, objList=None):
    
    items = []
    for k in objList:
        objName=k
        ob = bpy.data.objects[objName]
        objSize = (ob.bound_box[7][1]-ob.bound_box[0][1])*ob.scale[1]
        totalSize = objSize+gapSize
        item = ((objName, int(totalSize*10), int(totalSize*10)))
        items.append(item)

    maxofequalhouse=20
    """
    fin=False
    x = 0
    while fin!=True:
        x+=1
        fin=True
        if maxofequalhouse!=1:
            a,b,c,d = knapsack_unbounded_dp(items,pathLen,maxofequalhouse)
            for k in d:
                for j in d:
                    if j[1]/k[1]>3:
                        maxofequalhouse -=1
                        fin=False     
                    
        if x==40:
            fin=True
            print(knapsack_unbounded_dp(items,pathLen,maxofequalhouse))
            print(items[0][1])
            print(items[1][1])
            print(items[2][1])
            print("ERRROR")        
    """
    
    #print("House Built")
    #print("value, size, numbagged, bagged")
    #print(knapsack_unbounded_dp(items,pathLen,maxofequalhouse))  
    a,b,c,d = knapsack_unbounded_dp(items,pathLen,maxofequalhouse)
                      
    return d,b

        

            
def duplicateAlongSegmentMix(pt1, pt2, gapSize, objList=None):
    """Duplicate an object several times along a path
    pt1 -- First extreme of the path
    pt2 -- Second extreme of the path
    objName -- the name of blender obj to be copied
    gapSize -- Desired space between objects. Will be adjusted to fit path
    """
    
    
    objName=objList[0]
    #Compute the orientation of segment pt1-pt2
    dx = pt1[0]-pt2[0]
    dy = pt1[1]-pt2[1]

    # return if pt1 == pt2
    if dx == 0 and dy == 0:
        return

    # Compute the angle with the Y-axis
    ang = acos(dy/sqrt((dx**2)+(dy**2)))
    if dx > 0:
        ang = -ang
        
    
    # Compute the direction of the segment
    pathVec = Vector(pt2)-Vector(pt1)
    pathLen = pathVec.length
    pathVec.normalize() 
    
   
    list,spaceUsed = knapsack_unbounded_dp_control(pathLen,gapSize,objList)
    objList=[]
    for m in list:
        for n in range(m[1]):
            objList.append(m[0])
    
    random.shuffle(objList)
    delta = (int(pathLen*10)-spaceUsed)/10
    if objList == []:
        return
    delta = delta/len(objList)
    
    iniPoint = Vector(pt1)
   
    for i in objList:
        ob = bpy.data.objects[i]
        objSize = (ob.bound_box[7][1]-ob.bound_box[0][1])*ob.scale[1]
        totalSize = objSize+gapSize+delta
        loc = iniPoint
        g1 = duplicateObject(ob, "_%s" % (objName))
        g1.rotation_euler = (0, 0, ang)
        g1.location = loc
        iniPoint = iniPoint + pathVec * totalSize

def makeGround(cList=[], objName="meshObj", meshName="mesh", radius=10.0, material='Floor3'):
    """Create a polygon to represent the ground around a city 
    cList    -- A list of 3D points with the vertex of the polygon (corners of the city block)
    objName  -- the name of the new object
    meshName -- the name of the new mesh
    radius   -- radius around the city
    """
    print("makeGround", datetime.now().time())
    #Create a mesh and an object
    me = bpy.data.meshes.new(meshName)
    ob = bpy.data.objects.new(objName, me)
    bpy.context.scene.objects.link(ob)  # Link object to scene

    # Fill the mesh with verts, edges, faces
    if cList:
        vectors = [vertices3D[i] for i in cList]
    else:
        #Create a 16-sides polygon centered on (0,0,0)
        step = 2 * math.pi / 16
        vectors = [(sin(step*i) * radius, cos(step*i) * radius, -0.1) for i in range(16)]
    
    me.from_pydata(vectors, [], [tuple(range(len(vectors)))])
    me.update(calc_edges=True)    # Update mesh with new data
    #Assign a material to this object
    me.materials.append(bpy.data.materials[material])



def makePolygon(cList, objName="meshObj", meshName="mesh", height=0.0, reduct=0.0, hide=True, nr=None, seed=None):
    """Create a polygon/prism to represent a city block
    cList    -- A list of 3D points with the vertex of the polygon (corners of the city block)
    objName  -- the name of the new object
    meshName -- the name of the new mesh
    height   -- the height of the prism
    reduct   -- a distance to reduce from corner
    nr       -- The number if for thiss region
    seed     -- Coordinates of the seed for this region
    """
    print(".", end="")
    
    nv = len(cList)

    if not seed:
        #Compute center of voronoi region
        seed = [0.0,0.0]
        for v in cList:
            seed[0] += v[0]
            seed[1] += v[1]
        seed[0] /= nv
        seed[1] /= nv
    #pprint("seed", seed)

    #Compute reduced region coordinates
    cList2 = []
    for i in range(nv):
        dx = cList[i][0]-seed[0]
        dy = cList[i][1]-seed[1]
        dist = sqrt(dx*dx+dy*dy)
        if dist < reduct:
            cList2.append(cList[i])
        else:
            vecx = reduct * dx / dist
            vecy = reduct * dy / dist
            cList2.append((cList[i][0]-vecx,cList[i][1]-vecy,cList[i][2]))

    # 1. Create a mesh for streets around this region
    # This is the space between polygons clist and clist2
    me = bpy.data.meshes.new("_Street")
    ob = bpy.data.objects.new("_Street", me)
    streetData = []
    for i in range(nv):
        streetData.append(((i-1) % nv, i, nv+i, nv+(i-1) % nv))
    # pprint(streetData)
    me.from_pydata(cList+cList2, [], streetData)
    me.update(calc_edges=True)
    me.materials.append(bpy.data.materials['Floor1'])
    bpy.context.scene.objects.link(ob)

    # 2. Create a mesh interior of this region
    # This is the space inside polygon clist2
    me = bpy.data.meshes.new("_Region")
    ob = bpy.data.objects.new("_Region", me)
    me.from_pydata(cList2, [], [tuple(range(nv))])
    me.update(calc_edges=True)
    me.materials.append(bpy.data.materials['Floor2'])
    #me.materials.append(bpy.data.materials['Grass'])
    bpy.context.scene.objects.link(ob)

    # OK 3. Put a tree in the center of the region
    #g1 = duplicateObject(bpy.data.objects["Tree"], "_Tree")
    #g1.location = (seed[0], seed[1], 0.0)
    #bpy.ops.object.text_add(location=(seed[0], seed[1], 0.0))
    
    # Debug: Create a text object with the number of the region 
    textCurve = bpy.data.curves.new(type="FONT",name="_textCurve")
    textOb = bpy.data.objects.new("_textOb",textCurve)
    textOb.location = (seed[0], seed[1], 0.3)
    textOb.color = (1,0,0,1)
    textOb.scale = (5,5,5)
    textOb.data.body = str(nr)
    bpy.context.scene.objects.link(textOb)
    

    # 4. Fill boundary of region with Curbs
    for i in range(nv):
        duplicateAlongSegment(cList2[i-1], cList2[i], "Curb", 0.1)
    
    # 5. Create Houses
    
    #Compute new reduced region coordinates
    cList3 = []
    cList4 = []
    reduct = reduct * 6
    for i in range(nv):
        dx = cList[i][0]-seed[0]
        dy = cList[i][1]-seed[1]
        dist = sqrt(dx*dx+dy*dy)
        if dist < reduct:
            cList3.append(cList[i])
        else:
            vecx = reduct * dx / dist
            vecy = reduct * dy / dist
            vecxM = reduct * 1.5 * dx / dist
            vecyM = reduct * 1.5 * dy / dist
            cList3.append((cList[i][0]-vecx,cList[i][1]-vecy,cList[i][2]))
            cList4.append((cList[i][0]-vecxM,cList[i][1]-vecyM,cList[i][2]))
    
    for i in range(nv):
        duplicateAlongSegmentMix (cList3[i-1], cList3[i], 1 ,args["inputHouses"])
        duplicateAlongSegment(cList4[i-1], cList4[i], "WallHouse", 0, True )

    """
    #Create a mesh for colision
    me = bpy.data.meshes.new(meshName)   # create a new mesh
    ob = bpy.data.objects.new(objName, me) # create an object with that mesh
    bpy.context.scene.objects.link(ob)  # Link object to scene

    # Fill the mesh with verts, edges, faces
    me.from_pydata(cList2,[],[tuple(range(len(cList2)))])   # (0,1,2,3..N)
    me.update(calc_edges=True)    # Update mesh with new data

    #Avoid extrusion if height == 0
    if (not height):
        return

    #Extrude the mesh in the direction of +Z axis
    if (bpy.context.scene.objects.active):
        bpy.context.scene.objects.active.select = False
    bpy.context.scene.objects.active = ob
    ob.select = True
    bpy.ops.object.mode_set(mode = 'EDIT')
    hVec=Vector((0.0,0.0,height))
    bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value":hVec})
    me.update(calc_edges=True)    # Update mesh with new data
    bpy.ops.object.mode_set(mode = 'OBJECT')

    ob.select = False
    #Hide this region
    ob.hide = hide
    """

def updateExternalTexts():
    """ Check modified external scripts in the scene and update if possible
    """
    ctx = bpy.context.copy()
    ctx['area'] = ctx['screen'].areas[0]
    for t in bpy.data.texts:
        if t.is_modified and not t.is_in_memory:
            print("  * Warning: Updating external script", t.name)
            # Change current context to contain a TEXT_EDITOR
            oldAreaType = ctx['area'].type
            ctx['area'].type = 'TEXT_EDITOR'
            ctx['edit_text'] = t
            bpy.ops.text.resolve_conflict(ctx, resolution='RELOAD')
            #Restore context
            ctx['area'].type = oldAreaType            

def importLibrary(filename, link=False, destinationLayer=1, importScripts=True):
    """Import all the objects/assets from an external blender file
    filename -- the name of the blender file to import
    link     -- Choose to copy or link the objects
    destinationLayer  -- The destination layer where to copy the objects
    importScripts -- Choose to import also the scripts (texts) 
    """
    print('Importing objects from file', filename)
    with bpy.data.libraries.load(filename, link=link) as (data_from, data_to):
        #Import all objects
        objNames = [o.name for o in bpy.data.objects]
        for objName in data_from.objects:
            if objName.startswith('_'):
                print('  - Ignore', filename, '->', objName, '(name starts with _)')
            else:
                print('  + Import', filename, '->', objName)
                if objName in objNames:
                    print('Warning: object', objName, 'is already in this file')
                else:
                    data_to.objects.append(objName)
        #Import groups
        for grName in data_from.groups:
            if grName.startswith('_'):
                print('  - Ignore', filename, '->', grName, '(name starts with _)')
            else:
                print('  + Import group', filename, '->', grName)
                if grName in [g.name for g in bpy.data.groups]:
                    print('Warning: group', grName, 'is already in this file')
                else:
                    data_to.groups.append(grName)
                    
        if importScripts:
            #Import all text/scripts
            textNames = [o.name for o in bpy.data.texts]
            for textName in data_from.texts:
                if textName in textNames:
                    print('  - Warning: script', textName, 'is already in this file')
                else:
                    print('  + Import', filename, '->', textName)
                    data_to.texts.append(textName)
                        
    #link to scene, and move to layer destinationLayer
    for o in bpy.data.objects :
        if o.users_scene == () :
            bpy.context.scene.objects.link(o)
            #Set the layer
            if destinationLayer:
                o.layers[destinationLayer] = True
                o.layers[0] = False

    updateExternalTexts()



def pointDistance(x1, y1, x2, y2):
    distance = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))
    return distance



def nearestSeed(vector, seeds):
    distance = None
    for i in seeds:
        d = pointDistance(vector[0], vector[1], i[0], i[1])
        if (distance == None):
            distance = d
            seed = seeds.index(i)
        else:
            if (d < distance):
                distance = d
                seed = seeds.index(i)
    return seed



def nearestSegment(x, y , vertices, vert_coords):
    distance = None
    for i in vertices:
        coordinates = vert_coords[i]
        d = pointDistance(x, y, coordinates[0], coordinates[1])

        if (distance == None):
            distance = d
            vertex = i
        else:
            if (d < distance):
                distance = d
                vertex = i

    vertex = vertices.index(vertex)
    segment1 = [vert_coords[vertices[vertex - 1]], vert_coords[vertices[vertex]]]
    segment2 = [vert_coords[vertices[vertex]], vert_coords[vertices[-len(vertices) + vertex + 1]]]

    dist1 = ((segment1[1][0] - segment1[0][0]) * (y - segment1[0][1]) - (
        segment1[1][1] - segment1[0][1]) * (x - segment1[0][0])) / (
            sqrt(pow(segment1[1][0] - segment1[0][0], 2) + pow(segment1[1][1] - segment1[0][1], 2)))

    dist2 = ((segment2[1][0] - segment2[0][0]) * (y - segment2[0][1]) - (
        segment2[1][1] - segment2[0][1]) * (x - segment2[0][0])) / (
            sqrt(pow(segment2[1][0] - segment2[0][0], 2) + pow(segment2[1][1] - segment2[0][1], 2)))

    if (dist1 < dist2):
        return (segment1, dist1)
    else:
        return (segment2, dist2)



def createLeaves(seeds, internalRegions, vertices):
    print("Creating leaves...")
    hojas = 0
    loops = 0

    while (hojas < 3000):
        loops = loops + 1
        (x, y) = (uniform(-300, 300), uniform(-300, 300))
        vector = Vector((x, y, 0.1))

        n = nearestSeed(vector, seeds)
        (s, d) = nearestSegment(x, y , internalRegions[n], vertices)

        if (d < 4.5 and d > 1.5):
            g1 = duplicateObject(bpy.data.objects["DryLeaf"], "_leave_" + str(hojas))
            g1.location = vector
            g1.rotation_euler = (0, 0, uniform(0, 360))
            hojas = hojas + 1

    print("\nLeaves created (", loops, "loops)")


       
###########################
# The one and only... main
def main():
    # Current time
    iniTime = datetime.now()
    filepath = bpy.data.filepath
    if filepath:
        print("Current blender file:", filepath)
        cwd = os.path.dirname(filepath)+'/'
    else:
        cwd = ''
    
    print("Current cwd directory:", cwd)
    
    # Set a default filename to read configuration
    argsFilename = 'cg-config.json'   

    #Check if there is arguments after '--'
    if '--' in sys.argv:
        argv = sys.argv[1+sys.argv.index('--'):]
        print("argv", argv)
        if argv:
            #By now, only use last argument as configuration file
            argsFilename = argv[-1]


    #Read options from external file
    print("Trying to read options from file:", argsFilename)   
    try:
        with open(argsFilename, 'r') as f:
            import json
            args.update(json.load(f))
            #print("Read args:", [x for x in args]);
            for n in args:
                print("  *",n,"=",args[n])
            #Python documentation say NOT to do this :-)
            #globals().update(args)
    except IOError:
        print("Could not read file:", argsFilename)
        pass

    # Ensure configuration of blenderplayer in mode 'GLSL'
    bpy.context.scene.render.engine = 'BLENDER_GAME'
    bpy.context.scene.game_settings.show_fullscreen = True
    bpy.context.scene.game_settings.use_desktop = True
    bpy.context.scene.game_settings.material_mode = 'GLSL'
    bpy.ops.file.autopack_toggle()
    #print("bpy.data.use_autopack ", bpy.data.use_autopack )
    # Enable this if you need debug properties on screen
    bpy.context.scene.game_settings.show_debug_properties = True
    #bpy.context.scene.game_settings.show_physics_visualization = True


    for a in bpy.data.screens['Default'].areas:
        if a.type == 'VIEW_3D':
            a.spaces[0].viewport_shade = 'MATERIAL'

    
    # Select Layer 0 and clear the scene
    bpy.context.scene.layers[0] = True
    for i in range(1, 20):
        bpy.context.scene.layers[i] = False
        
    #clean objects in layer 0
    if args['cleanLayer0']:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        #clean scripts
        for k in bpy.data.texts:
            if '.py' in k.name and 'run-' not in k.name:
                print("Remove script: ", k.name)
                bpy.data.texts.remove(k)
        #clean unused data
        for k in bpy.data.textures:
            if k.users == 0:
                print("Remove texture: ", k.name)
                bpy.data.textures.remove(k)
        for k in bpy.data.materials:
            if k.users == 0:
                print("Remove material: ", k.name)
                bpy.data.materials.remove(k)
        for k in bpy.data.actions:
            if k.users == 0:
                print("Remove action: ", k.name)
                bpy.data.actions.remove(k)

    # This is a hack to give blender a current working directory. If not, it will
    # write several warnings of the type "xxxxx can not make relative"
    print('Saving empty blender model:', cwd+'empty.blend')
    bpy.ops.wm.save_as_mainfile(filepath=cwd+'empty.blend', compress=False, copy=False)
    os.remove(cwd+'empty.blend')

    # Read point, vertex and regions from a json file (the output of cityGen2D)
    inputFilename = args['inputFilename']
    print("Read cityGen2D data from file", inputFilename)
    with open(cwd+inputFilename, 'r') as f:
        data = json.load(f)
        print("Data:", [x for x in data]);
        if 'name' in data:
            print("City name:", data['name'])
        seeds = data['seeds']
        vertices = [Vector(v) for v in data['vertices'] ]
        regions = data['regions']
        internalRegions = data['internalRegions']
        externalPoints = data['externalPoints']
        # This is a hack to convert dictionaries with string keys to integer.
        # Necessary because json.dump() saves integer keys as strings
        regions = { int(k):v for k,v in regions.items() }
        internalSeeds = [Vector(s) for s in seeds[:len(internalRegions)]]

    ###########################################################################
    #        Create a 3D model of the city
    ###########################################################################

    #Save a copy of input data as a text buffer in blend file
    if inputFilename in bpy.data.texts:
        bpy.data.texts.remove(bpy.data.texts[inputFilename])
    bpy.data.texts.load(inputFilename, True)
    
    #Save a copy of input AI data as a buffer in blend file
    inputFilenameAI = args['inputFilenameAI']
    if inputFilenameAI in bpy.data.texts:
        bpy.data.texts.remove(bpy.data.texts[inputFilenameAI])
    bpy.data.texts.load(inputFilenameAI, True)
    
    # Convert vertex from 2D to 3D
    vertices3D = [ v.to_3d() for v in vertices ]
    
    # Compute the radius of the city, as the max distance from any vertex to origimathn
        
    # Insert a camera and a light in the origin position
    # bpy.ops.object.camera_add(view_align=True, enter_editmode=False, location=(0,0,1.5), rotation=((math.radians(90)), 0, 0))
    # bpy.ops.object.lamp_add(type='SUN', view_align=False, location=(0, 0, 2))

    #Read all the assets for buildings from cg-library.blend
    if not isinstance(args['inputLibraries'], list):
        args['inputLibraries'] = [args['inputLibraries']]
    for l,lib in enumerate(args['inputLibraries']):
        importLibrary(lib, link=False, destinationLayer=1+l, importScripts=True)

    #Insert global ilumination to scene
    if args.get('createGlobalLight', False):
        print("Creating Global Light")
        bpy.ops.object.lamp_add(type='SUN', radius=1, view_align=False, location=(0,0,2), rotation=(0,0.175,0))
        bpy.data.lamps[-1].name='ASun1'
        bpy.ops.object.lamp_add(type='SUN', radius=1, view_align=False, location=(0,0,2), rotation=(0,-0.175,0))
        bpy.data.lamps[-1].name='ASun2'
    
    #Insert and scale skyDome
    if args.get('inputSkyDome', False):
        importLibrary(args['inputSkyDome'], link=False, destinationLayer=0, importScripts=True)
        #Compute the radius of the dome and apply scale
        skyDomeRadius = 50 + max([v.length for v in vertices])
        print("Scaling SkyDome object to radius",skyDomeRadius)
        bpy.data.objects["SkyDome"].scale=(skyDomeRadius, skyDomeRadius, skyDomeRadius/2)

        """ Nice, but still need some configuration
        importLibrary("cg-skyboxshader.blend", link=False, destinationLayer=0, importScripts=True)
        """
       
        """ OK. If you want mist
        #Add mist
        bpy.context.scene.world.mist_settings.use_mist = True
        bpy.context.scene.world.horizon_color = (0.685146, 0.800656, 0.728434)
        #"""
                
    # Create exterior boundary of the city
    if args.get('createDefenseWall', False):
        print("Creating External Boundary of the City, Defense Wall")
        wallVertices = data['wallVertices']

        # place defense wall. Avoid extreme corners
        axisX = Vector((1.0, 0.0))

        #Compute the position of the gate
        gate1 = Vector(wallVertices[0])
        gate2 = Vector(wallVertices[-1])
        gateMid = (gate1+gate2) * 0.5

        # Compute orientation of gate with axisX
        angGate = (gate1-gate2).angle_signed(axisX)-math.pi/2
        #Insert a gate at position gateMid
        for o in bpy.data.groups["StoneGate"].objects:
            g1 = duplicateObject(o, "_Gate1_"+o.name)
            g1.location = (gateMid[0], gateMid[1], 0)
            g1.rotation_euler = (0, 0, angGate)
        
        #Insert one tower at gate1
        g1 = duplicateObject(bpy.data.objects["StoneTower"], "_Gate1_Tower1")
        g1.location = (gate1[0], gate1[1], 0)
        g1.rotation_euler = (0, 0, angGate)
        
        # Place a door on point gate1, oriented to angR (next section of wall)
        g1 = duplicateObject(bpy.data.objects["StoneTowerDoor"], "_Door%03d_B" % i)
        g1.location = (gate1[0], gate1[1], 0)
        g1.rotation_euler = (0, 0, angGate+math.pi/2)
                
        # Build the defense wall around the city
        for i in range(1, len(wallVertices)):
            v1 = wallVertices[i-1]
            v2 = wallVertices[i]
            v3 = wallVertices[(i+1) % len(wallVertices) ]
            v_1_2 = Vector((v1[0]-v2[0], v1[1]-v2[1]))
            v_3_2 = Vector((v3[0]-v2[0], v3[1]-v2[1]))
            # Compute orientation of both walls with axisX
            angL = v_1_2.angle_signed(axisX)
            angR = v_3_2.angle_signed(axisX)
            # Force angR > angL, so ensure that angL < average < angR
            if (angL > angR):
                angR += 6.283185307

            # Compute the average of angL , angR
            ang = (angL+angR)*0.5
            
            # Place a new tower on point v2 (the endpoint of this section of wall)
            g1 = duplicateObject(bpy.data.objects["StoneTower"], "_Tower%03d" % i)
            g1.location = (v2[0], v2[1], 0)
            g1.rotation_euler = (0, 0, ang)
            # g1.show_name = True #Debug info
            # Place a new door on point v2, oriented to angL (this section of wall)
            g1 = duplicateObject(bpy.data.objects["StoneTowerDoor"], "_Door%03d_A" % i)
            g1.location = (v2[0], v2[1], 0)
            g1.rotation_euler = (0, 0, angL)

            # Place a second door on point v2, oriented to angR (next section of wall)
            if i < len(wallVertices)-1:
                g1 = duplicateObject(bpy.data.objects["StoneTowerDoor"], "_Door%03d_B" % i)
                g1.location = (v2[0], v2[1], 0)
                g1.rotation_euler = (0, 0, angR)
            
            # Fill this section of wall with wallBlocks
            sw = duplicateAlongSegment(v1, v2, "StoneWall", 0.0)
            # print("New StoneWall section", v1, "->", v2, "Size: ", len(sw) )
                
            # Create a quad-mesh for streets near of this section of wall
            me = bpy.data.meshes.new("_Street")
            ob = bpy.data.objects.new("_Street", me)
            # Create a list with the four vertex of this quad
            myVertex = [(v1[0], v1[1], 0), (v2[0], v2[1], 0), vertices3D[externalPoints[i]], vertices3D[externalPoints[i-1]]]            
            me.from_pydata(myVertex, [], [(0,1,2,3)])
            me.update(calc_edges=True)
            me.materials.append(bpy.data.materials['Floor1'])
            bpy.context.scene.objects.link(ob)
                    
    # Create a ground around the boundary
    if args.get('createGround', False):
        createGround = args['createGround']
        groundRadius = 50 + max([v.length for v in vertices])
        makeGround([], '_groundO', '_groundM', radius=groundRadius, material='Floor3')

    if args.get('createStreets', False):
        # Create paths and polygon for internal regions
        print("Creating Districts")
        for nr, region in enumerate(internalRegions):
            print(".", end="")
            corners = [vertices3D[i] for i in region]
            makePolygon(corners, "houseO", "houseM", height=0.5, reduct=1.0, nr=nr, seed=seeds[nr])
        print(".")

        # Merge streets meshes in one object
        streets = [x for x in bpy.data.objects if x.name.startswith("_Street")]
        for o in bpy.data.objects:
            o.select = (o in streets)
        bpy.context.scene.objects.active = streets[0]
        bpy.ops.object.join()
        
        # Merge region meshes in one object
        for o in bpy.data.objects:
            o.select = o.name.startswith("_Region")
        bpy.context.scene.objects.active = bpy.data.objects["_Region"]
        bpy.ops.object.join()

    if args.get('createLeaves', False):
        createLeaves(internalSeeds, internalRegions, vertices)


    #Save the current file, if outputCityFilename is set.
    if args.get('outputCityFilename', False):
        outputCityFilename = args['outputCityFilename']
        print('Saving blender model as:', outputCityFilename)
        bpy.ops.wm.save_as_mainfile(filepath=cwd+outputCityFilename, compress=True, copy=False)

    ###########################################################################
    #        Create lets-take-a-nice-walk game
    ###########################################################################

    #Import the player system
    if args.get('inputPlayer', False):
        importLibrary(args['inputPlayer'], destinationLayer=0, importScripts=True)

        #locate the object named Player
        player = bpy.data.objects['Player']

        #Search the vertex nearest to the center of the city
        vlength = [v.length for v in vertices]
        # https://stackoverflow.com/questions/2474015/
        playerVertex = min(range(len(vertices)), key=vlength.__getitem__)
        locP = vertices3D[playerVertex] + Vector((0,0,3))
        
        print('Player starts at vertex:', playerVertex, 'position:', locP.to_tuple())
        
        # Show/hide the token that marks the nearest street point to the player
        if 'debugVisibleTokens' in args and 'Target' in bpy.data.objects:
            bpy.data.objects['Target'].hide_render = not args['debugVisibleTokens']

        #Inject a new string property to the player object
        if 'playerName' not in player.game.properties:
            bpy.context.scene.objects.active = player
            bpy.ops.object.game_property_new(name="playerName", type='STRING')
            player.game.properties['playerName'].value='Askeladden'

        #Inject a string property with a json code that can be parsed by a controller
        if 'initPos' not in player.game.properties:
            bpy.context.scene.objects.active = player
            bpy.ops.object.game_property_new(name="initPos", type='STRING')
        player.game.properties['initPos'].value=str(locP)

    #Insert a background music
    if args.get('backgroundMusic', False):
        backgroundMusic = args['backgroundMusic']
        print('Insert background music file:', backgroundMusic)
        #bpy.ops.sequencer.sound_strip_add(filepath=backgroundMusic, relative_path=True, frame_start=1, channel=1)
        bpy.ops.sound.open(filepath=backgroundMusic, relative_path=True)
        bpy.ops.logic.sensor_add(name='playMusic', type='ALWAYS', object='Player')
        bpy.ops.logic.controller_add(name='playMusic', object='Player')
        bpy.ops.logic.actuator_add(name='playMusic', type='SOUND', object='Player') #Try to link to other object...
        player.game.actuators['playMusic'].sound = bpy.data.sounds[os.path.basename(backgroundMusic)]
        player.game.actuators['playMusic'].mode = 'LOOPEND'
        player.game.controllers['playMusic'].link(sensor=player.game.sensors['playMusic'], actuator=player.game.actuators['playMusic'])
            
    #Save the current file, if outputGameFilename is set.
    if args.get('outputTourFilename', False):
        outputTourFilename = args['outputTourFilename']
        print('Saving blender tourist as:', outputTourFilename)
        bpy.ops.wm.save_as_mainfile(filepath=cwd+outputTourFilename, compress=True, copy=False)

    ###########################################################################
    #        Create a get-me-outta-here game
    ###########################################################################
    
    #Read the number of monsters in the city
    numMonsters = args.get('numMonsters', 0)
            
    if numMonsters > 0:

        #Read all the assets for monsters and non-players
        if not isinstance(args['inputMonsterLibrary'], list):
            args['inputMonsterLibrary'] = [args['inputMonsterLibrary']]
        for l,lib in enumerate(args['inputMonsterLibrary']):            
            importLibrary(lib, link=False, destinationLayer=10+l, importScripts=True)

        if 'AI_Manager' not in bpy.data.objects:
            print("AI_Manager object not found in libraries")
            return            
        AI_Manager = bpy.data.objects['AI_Manager']
        
        #Bring AI_Manager to layer 0
        AI_Manager.layers[0] = True        
        
        #Inject a new python controller to the object, linked to an existing text
        #This is a trick so BGE can find a text object
        #http://blenderartists.org/forum/showthread.php?226148-reading-text-datablocks-via-python
        #See leeme.txt to find an example to search and parse a complex json-text
        bpy.context.scene.objects.active = AI_Manager
        bpy.ops.logic.controller_add(name='cg-data.json', type='PYTHON')
        AI_Manager.game.controllers['cg-data.json'].text = bpy.data.texts[inputFilename]        
        
        #Inject a new python controller to the object, linked to inputFilenameAI
        bpy.context.scene.objects.active = AI_Manager
        bpy.ops.logic.controller_add(name='cg-ia.json', type='PYTHON')
        AI_Manager.game.controllers['cg-ia.json'].text = bpy.data.texts[inputFilenameAI]
        
        AIData={}
        print("Read AI data from:", inputFilenameAI)
        with open(cwd+inputFilenameAI, 'r') as f:
            AIData.update(json.load(f))
            print("AIData:", [x for x in AIData]);       
            
        print("Choosing starting points for monsters...")
        #Build list of internal vertex
        internalPoints = [i for i in range(len(vertices)) if i not in externalPoints]
        #print("internalPoints=", internalPoints)
        monsterVertex=[]
        for i in range(numMonsters):
            maxDistance = -1
            maxDistVertex = None
            for v in [n for n in internalPoints if n not in monsterVertex]:
                #Sum of distances from vertex v to every other monster/player
                #distance = sum(AIData["shortestPathMatrix"][v][j] for j in [playerVertex]+monsterVertex)
                #Minimum distance from vertex v to every other monster/player
                distance = min(AIData["shortestPathMatrix"][v][j] for j in [playerVertex]+monsterVertex)
                #Choose the vertex v which maximizes the distance to others
                if distance > maxDistance and distance < float('Inf'):
                    maxDistance = distance
                    maxDistVertex = v

            #print("  + Selected vertex", maxDistVertex, "at distance", maxDistance)
            monsterVertex += [maxDistVertex]
        print("Starting vertex for monsters", monsterVertex)

        #Set the list of vertex where monsters spawn as a game property of AI
        bpy.context.scene.objects.active = AI_Manager
        bpy.ops.object.game_property_new(name="iniMonsters", type='STRING')
        AI_Manager.game.properties['iniMonsters'].value=str(monsterVertex)           

    # Check modified external scripts and update if necessary
    updateExternalTexts()

    #Save the current file, if outputGameFilename is set.
    if args.get('outputGameFilename', False):
        outputGameFilename = args['outputGameFilename']
        print('Saving blender game as:', outputGameFilename)
        bpy.ops.wm.save_as_mainfile(filepath=cwd+outputGameFilename, compress=True, copy=False)
        print ("Ready to run: blenderplayer", outputGameFilename);
            
    # totalTime = (datetime.now()-iniTime).total_seconds()
    # print("Regions:", len(regions), " Total Time:" totalTime)

#Call the main function
if __name__ == '__main__':
    main()


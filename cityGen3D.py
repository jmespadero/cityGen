"""
Game generator from project citygen
Reads a .json file generated with cityGen2D.py, and build a 3D model of the city
The process is fully configurable using the file cg-config.json

Copyright 2014 Jose M. Espadero <josemiguel.espadero@urjc.es>
Copyright 2014 Juan Ramos <juanillo07@gmail.com>
Copyright 2017 Sergio Fernandez <serfervic@gmail.com>

Run option 1:
blender --background --python cityGen3D.py

Run option 2:
Open blender and type this in the python console:
  exec(compile(open("cityGen3D.py").read(), "cityGen3D.py", 'exec'))

"""

"""
TODO:
  * Build random houses. Create non-rectangular houses for corners.
  * Procedural generation of starred night sky.
  * Remove that ugly knapsack_unbounded_dp method()
DONE:
  * When importing a blend file (cg-library, etc...) append its "readme.txt" to
    a "readme.txt" in the output. This will honor all the CC-By resources.
  * Fix armatures when set position to an armatured object (see initPos)
  * Add cg-temple to cities.
  * Add roads outside of the city (at least one in near the gate)
"""
import bpy, bmesh
import math, json, random, os, sys
from math import sqrt, acos, sin, cos, ceil
from pprint import pprint
from mathutils import Vector
from datetime import datetime
from random import random, uniform, choice, shuffle
from functools import reduce

#Set default values for args. Will be overwritten with values at cg-config.json
args={
'cleanLayer0' : True,       # Clean all objects in layer 0
'createGlobalLight' : True,         # Add new light to scene
'inputFilename' : 'city.data.json',  # Set a filename to read 2D city map data
'inputFilenameAI' : 'city.AI.json',   # Set a filename to read AI data
'inputLibraries' : 'cg-library.blend',  # Set a filename for assets (houses, wall, etc...) library.
'inputHouses' : ["House7", "House3","House4","House5","House6"],
'inputPlayer' : 'cg-playerBoy.blend',   # Set a filename for player system.
'inputTemple' : 'cg-temple.blend',
'inputMarket' : 'cg-market.blend',
'createDefenseWall' : True,  # Create exterior boundary of the city
'createGround' : True,       # Create ground boundary of the city
'createStreets' : True,      # Create streets of the city
'createLeaves' : True,      # Create leaves on the streets
'createRiver' : True,        # Create river
'createTrail' : True,        # Create trail
'createEspecialBuildings' : True,    # Create buildings on specific regions
'numMonsters' : 4,
'outputCityFilename' : 'outputcity.blend', #Output file with just the city
'outputTourFilename' : 'outputtour.blend', #Output file with complete game
'outputGameFilename' : 'outputgame.blend', #Output file with complete game
}

#global timer to profile 
initTime = datetime.now()

#################################################################
# Functions to create a new cityMap scene (does need run inside blender)

def joinObjectsByName(rootName):
    """Join a every blender object whose name match a string in one unique object
    """
    for o in bpy.data.objects:
        o.select = o.name.startswith(rootName)
    if rootName in bpy.data.objects:
        bpy.context.scene.objects.active = bpy.data.objects[rootName]
    else:
        for o in bpy.data.objects:
            if o.select:
                bpy.context.scene.objects.active = o
                break
    bpy.ops.object.join()
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.context.scene.objects.active.select = False
    bpy.context.scene.objects.active = None

def joinObjectsList(objList):
    """Join a list of blender object in one unique object
    """
    for o in objList:
        o.select = (o.type == 'MESH')
    bpy.context.scene.objects.active = objList[0]
    bpy.ops.object.join()
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    #Clean selected objects        
    bpy.context.scene.objects.active.select = False
    bpy.context.scene.objects.active = None
    return objList[0]

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


def duplicateAlongSegment(pt1, pt2, objName, gapSize, join=True, force=False):
    """Duplicate an object several times along a path
    pt1 -- First extreme of the path
    pt2 -- Second extreme of the path
    objName -- the name of blender obj to be copied
    gapSize -- Desired space between objects. Will be adjusted to fit path
    """

    # Compute the direction of the segment
    pathVec = pt2-pt1
    pathLen = pathVec.length
    pathVec.normalize()

    # Compute the angle with the Y-axis
    ang = (-pathVec.xy).angle_signed(Vector((0,1)))
    
    # Get the size of the replicated object in the Y dimension
    ob = bpy.data.objects[objName]
    objSize = (ob.bound_box[7][1]-ob.bound_box[0][1])*ob.scale[1]
    totalSize = objSize+gapSize

    # Check object size 
    if pathLen == 0 or (objSize > pathLen):
        return []
              
    #if gapSize is not zero, change the gap to one that adjust the object
    #Compute the num of (obj+gap) segments in the interval (pt1-pt2)
    if gapSize != 0:
        numObj = round(pathLen/totalSize)
        step = pathLen/numObj
        stepVec = pathVec * step
        iniPoint = pt1+(stepVec * 0.5)
    else:
        numObj = math.floor(pathLen/objSize)
        step = objSize
        stepVec = pathVec * step
        delta = pathLen-step*numObj #xke? (delta es el espacio que falta para completar una fila)
        iniPoint = pt1+(stepVec*0.5) #se multiplicaba esto por delta, xke?
        

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
        #Make a real duplicate of the first object only
        if join and len(objList) == 1:
            g1.data = ob.data.copy()
            ob = g1
    if force:
            loc = pt2 - stepVec * 0.5
            g1 = duplicateObject(ob, "_%s" % (objName))
            g1.rotation_euler = (0, 0, ang)
            g1.location = loc
            objList.append(g1)

    if join and objList:
        joinObjectsList(objList)
        objList = [objList[0]]

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
    
    pt1 = Vector(pt1)
    pt2 = Vector(pt2)
    
    # Compute the direction of the segment
    pathVec = pt2-pt1
    pathLen = pathVec.length
    pathVec.normalize()

    # Compute the angle with the Y-axis
    ang = (-pathVec.xy).angle_signed(Vector((0,1)))
    
    # Check object size 
    if pathLen == 0 :
        return 
                 
    list,spaceUsed = knapsack_unbounded_dp_control(pathLen,gapSize,objList)
    objList=[]
    for m in list:
        for n in range(m[1]):
            objList.append(m[0])
                
    if objList == []:
        return

    shuffle(objList)
    
    delta = (int(pathLen*10)-spaceUsed)/(10*len(objList))    
    iniPoint = pt1

    for objName in objList:
        ob = bpy.data.objects[objName]
        objSize = (ob.bound_box[7][1]-ob.bound_box[0][1])*ob.scale[1]
        totalSize = objSize+gapSize+delta
        g1 = duplicateObject(ob, "_%s" % objName)
        g1.rotation_euler = (0, 0, ang)
        g1.location = iniPoint
        iniPoint = iniPoint + pathVec * totalSize

def makeGround(corners=[], objName="meshObj", meshName="mesh", radius=10.0, material='Floor3'):
    """Create a polygon to represent the ground around a city 
    corners    -- A list of 3D points with the vertex of the polygon (corners of the city block)
    objName  -- the name of the new object
    meshName -- the name of the new mesh
    radius   -- radius around the city
    """
    #Create a mesh and an object
    me = bpy.data.meshes.new(meshName)
    ob = bpy.data.objects.new(objName, me)
    bpy.context.scene.objects.link(ob)  # Link object to scene

    # Fill the mesh with verts, edges, faces
    if corners:
        vectors = [vertices3D[i] for i in corners]
    else:
        #Create a 16-sides polygon centered on (0,0,0)
        step = 2 * math.pi / 16
        vectors = [(sin(step*i) * radius, cos(step*i) * radius, -0.1) for i in range(16)]
    
    me.from_pydata(vectors, [], [tuple(range(len(vectors)))])
    me.update(calc_edges=True)    # Update mesh with new data
    #Assign a material to this object
    me.materials.append(bpy.data.materials[material])


def computeEnvelope(vertexList, distance=0):
    """ Compute the envelope (surrounding polygon at given distance)
    vertexList -- list of coordinates
    distance -- Distance to displace the envelope (negative will reduce the polygon)
    """
    nv = len(vertexList)

    # Compute the unit 2D vector for each side (vertex to its previous)
    edgeP = [(vertexList[i]-vertexList[i-1]).xy.normalized() for i in range(nv)]
    #Compute edge vectors (vertex to its next)
    edgeN=[-edgeP[(i+1)%nv] for i in range(nv)]
    # Compute the normal to each side rotating each edgeP
    edgeNormals = [Vector((v[1], -v[0])) for v in edgeP]

    # compute tangent weights as tan((pi - alpha) / 2) = sin(alpha)/(1-cos(alpha))
    w = [edgeN[i].cross(edgeP[i])/(1.0 - edgeP[i].dot(edgeN[i])) for i in range(nv)]
    
    #Compute the weighted external bisector for each vertex
    bisector = [edgeNormals[i] + w[i]*edgeP[i] for i in range(nv)]

    # Displace the external vertices by the bisector
    envelope = [vertexList[i].xy + distance * bisector[i] for i in range(nv)]
    
    # Check if input is 2D or 3D
    if len(vertexList[0]) == 2:
       return envelope
    else:
       #Extend to 3D using original Z coordinates
       return [Vector((envelope[i].x, envelope[i].y, vertexList[i].z)) for i in range(nv)]

            
def bilinear_interpolation(u, v, points):
    """Bilinear interpolation of values associated with four points.
       Values for u, v are expected to be in 0..1
       The points are values taken at (0,0), (0,1), (1,0), (1,1)       
       See https://en.wikipedia.org/wiki/Bilinear_interpolation#Unit_square
    """
    # Precompute (1-u) and (1-v)
    _u = 1 - u
    _v = 1 - v
    return _u * _v * points[0] + _u * v * points[1] + u * _v * points[2] + u * v * points[3] 
           
def createLeaves2(corners, min=0.0, max=1.0, density=0.1, height=0.02, objNames=["DryLeaf"], changeScale=0):
    """Scatter objects in random locations inside a region 
    corners     -- A list of 3D points with the vertex of the region (corners of the city district)
    min         -- minimum distance from region boundary (usually, the used as curbLine)
    max         -- maximum distance from region boundary (usually, the used as housesLine)
    density     -- Number of object to scatter per unit area
    objNames    -- Names of the objects to scatter
    changeScale -- Randomize the scale of the objects in interval [1-changeScale .. 1+changeScale]
    """    
    if not isinstance(objNames, list):
        objNames = [objNames]

    #Compute the "Onion model" coordinates for min and max lines
    zDispl = Vector((0,0,height))
    minLine = [v + zDispl for v in computeEnvelope(corners, -min)]
    maxLine = [v + zDispl for v in computeEnvelope(corners, -max)]

    scene=bpy.context.scene            
    obs = []
    
    #for each side of the region    
    for i in range(len(corners)):
        # Get the four corners for this trapezoidal subregion
        pnts = [minLine[i-1], minLine[i], maxLine[i-1], maxLine[i]]
        # Compute the area of the trapezoid as (a+b) * h / 2
        area = ((pnts[0]-pnts[1]).length + (pnts[2]-pnts[3]).length) * (max-min) / 2
        # Compute the number of objects to scatter
        #print("subregion=", i, "area=", area, "num_objs=", round(density * area) )
        for _ in range(round(density * area)):
            #o = duplicateObject(bpy.data.objects[choice(objNames)], "_leaf")
            o = bpy.data.objects[choice(objNames)].copy()
            o.name = "_leaf"
            o.data = o.data.copy()
            # Create a random position inside this trapezoidal subregion            
            o.location = bilinear_interpolation(random(), random(), pnts)
            # Randomize orientation in [0 .. 2*pi]
            o.rotation_euler = (0, 0, 6.28 * random())
            # Randomize scale in interval [1-changeScale .. 1+changeScale]
            scale = uniform(1-changeScale, 1+changeScale)
            o.scale = (scale, scale, scale)
            scene.objects.link(o)            
            obs.append(o)

    if obs:
        joinObjectsList(obs)
            
def makeDistrict(corners, curbReduct=1, houseReduct=1.5, regionID=None, hideWalls=True):
    """Create a polygon/prism to represent a city block
    corners     -- List of 3D points with the vertex of the polygon (corners of the city block)
    curbReduct  -- Distance from streetLine to curbs
    houseReduct -- Distance from curbs to houses
    regionID    -- The ID of this region. Set to None for emptyRegions/specialBuildings
    hideWalls   -- Assign invisible material to collisionWalls
    """
    nv = len(corners)

    #Compute the "Onion model" coordinates for curbs
    curbLine = computeEnvelope(corners, -curbReduct)

    # 1. Create a mesh for streets around this region
    # This is the space between polygons clist and curbLine
    me = bpy.data.meshes.new("_Street")
    ob = bpy.data.objects.new("_Street", me)
    streetData = [ ((i-1) % nv, i, nv+i, nv+(i-1) % nv) for i in range(nv)]
    # pprint(streetData)
    me.from_pydata(corners+curbLine, [], streetData)
    me.update(calc_edges=True)
    me.materials.append(bpy.data.materials['Floor1'])
    bpy.context.scene.objects.link(ob)

    # 2. Create a mesh interior of this region
    # This is the space inside polygon curbLine
    me = bpy.data.meshes.new("_Region")
    ob = bpy.data.objects.new("_Region", me)
    me.from_pydata(curbLine, [], [tuple(range(nv))])
    me.update(calc_edges=True)
    me.materials.append(bpy.data.materials['Floor2'])
    #me.materials.append(bpy.data.materials['Grass'])
    bpy.context.scene.objects.link(ob)

    # 4. Fill boundary of region curbLine with curbs
    curbList = []
    for i in range(nv):
        curbList += duplicateAlongSegment(curbLine[i-1], curbLine[i], "Curb", gapSize=0.1, join=True)
    joinObjectsList(curbList)

    """ WIP: Work in progress
    # 4. Fill boundary of region curbLine with curbs
    curbLine1 = [v + Vector((0,0,.01)) for v in computeEnvelope(curbLine, 0.1)]
    curbLine2 = [v + Vector((0,0,.01)) for v in computeEnvelope(curbLine, -0.1)]
    # Repeat last coordinate to avoid overflow coordinate indexes
    curbLine1.append(curbLine1[0])
    curbLine2.append(curbLine2[0])
    
    me = bpy.data.meshes.new("_NewCurb")
    ob = bpy.data.objects.new("_NewCurb", me)
    streetData = [(i, i+1, nv+2+i, nv+1+i) for i in range(nv)]
    # pprint(streetData)
    me.from_pydata(curbLine1+curbLine2, [], streetData)
    me.update(calc_edges=True)
    me.materials.append(bpy.data.materials['Curb2'])
    bpy.context.scene.objects.link(ob)
    
    # Compute accumulative linear distances for curbLine    
    coordU = [0]
    for i in range(nv):
        coordU.append(coordU[-1] + (curbLine[(i+1)%nv]-curbLine[i]).length)        
    #Scale texture by 0.2 and scale U Coorfinates so last coordinate is integer
    #This makes the texture seamless beause any U integer is equal to U=0
    scale = round(0.2 * coordU[-1]) / coordU[-1]
    uvs = [(u*scale, 0.05) for u in coordU]+[(u*scale, 0.95) for u in coordU]

    bm = bmesh.new()
    bm.from_mesh(me)
    uv_layer = bm.loops.layers.uv.verify()
    bm.faces.layers.tex.verify()
    for f in bm.faces:
        for l in f.loops:
            # We're giving the uvs index list value for each uv coordinates
            l[uv_layer].uv = uvs[l.vert.index]            
    bm.to_mesh(me)
    bm.free()
    # """

    
    # Avoid adding any more objects if the region has no ID
    if regionID is None:
        return

    #Compute the "Onion model" coordinates for houses
    houseLine = computeEnvelope(corners, -houseReduct)    
        
    # 5. Fill boundary of region houseLine with houses (Old method)
    for i in range(nv):
        #This is a silly bugfix to avoid houses in corners 
        A = 0.97 * houseLine[i-1] + 0.03 * houseLine[i]
        B = 0.03 * houseLine[i-1] + 0.97 * houseLine[i]        
        duplicateAlongSegmentMix (A, B, 0.5, args["inputHouses"])

    # 6. Create a collision mesh avoid enter beyond houseLine
    zDisp = Vector((0,0,10))
    upperLine = [v + zDisp for v in computeEnvelope(houseLine, 0.2) ]
    me = bpy.data.meshes.new("_CollisionW")
    ob = bpy.data.objects.new("_CollisionW", me)    
    wallData = [ ((i-1) % nv, i, nv+i, nv+(i-1) % nv) for i in range(nv)]
    me.from_pydata(houseLine+upperLine, [], wallData)
    me.update(calc_edges=True)
    #Make this mesh invisible for BGE
    me.materials.append(bpy.data.materials['Invisible'])
    #Make this object hidden in Blender Editor
    ob.hide = hideWalls
    bpy.context.scene.objects.link(ob)

    # 7. Debug: Create a visible text label with the regionID
    if args.get('debugVisibleTokens', False):    
        centroid = sum(corners, Vector((0,0,0)))/nv
        textCurve = bpy.data.curves.new(type="FONT",name="_textCurve")
        textOb = bpy.data.objects.new("_textOb",textCurve)
        textOb.location = (centroid[0], centroid[1], 0.3)
        textOb.color = (1,0,0,1)
        textOb.scale = (5,5,5)
        textOb.data.body = str(regionID)
        bpy.context.scene.objects.link(textOb)    

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
    print('Importing objects from file', filename, 'into layer', destinationLayer)
    with bpy.data.libraries.load(os.getcwd() + "\\" + filename, link=link) as (data_from, data_to):
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
            #Set the destination layer. 
            if destinationLayer:
                o.layers[destinationLayer] = True
                o.layers[0] = False
                #Bring to layer 0 objects whose name ends with "Manager"
                if o.name.endswith("Manager"):
                    print('  + Move', o.name, 'object to layer 0')
                    o.layers[0] = True
                    
    updateExternalTexts()



def nearestSeed(v, seeds):
    """Search the nearest seed to point v"""
    distance = float('inf')
    v = v.xy
    for s in seeds:
        d = (v - s).length
        if (d < distance):
            distance = d
            minSeed = seeds.index(s)
    return minSeed



def nearestSegment(vector , vertices, vert_coords):
    """Search the nearest segment to point vector"""
    distance = float('inf')
    for v in vertices:
        d = (vector.xy - vert_coords[v]).length

        if (distance == None):
            distance = d
            vertex = v
        else:
            if (d < distance):
                distance = d
                vertex = v

    vertex = vertices.index(vertex)
    segment1 = [vert_coords[vertices[vertex - 1]], vert_coords[vertices[vertex]]]
    segment2 = [vert_coords[vertices[vertex]], vert_coords[vertices[-len(vertices) + vertex + 1]]]

    dist1 = ((segment1[1][0] - segment1[0][0]) * (vector.y - segment1[0][1]) - (
        segment1[1][1] - segment1[0][1]) * (vector.x - segment1[0][0])) / (
            sqrt(pow(segment1[1][0] - segment1[0][0], 2) + pow(segment1[1][1] - segment1[0][1], 2)))

    dist2 = ((segment2[1][0] - segment2[0][0]) * (vector.y - segment2[0][1]) - (
        segment2[1][1] - segment2[0][1]) * (vector.x - segment2[0][0])) / (
            sqrt(pow(segment2[1][0] - segment2[0][0], 2) + pow(segment2[1][1] - segment2[0][1], 2)))

    if (dist1 < dist2):
        return (segment1, dist1)
    else:
        return (segment2, dist2)

def newRMDFractalPoint(p1, p2, factor, list, res):
    """ New recursive level of the RMD Fractal algorithm
            origin    -- The origin of the curve
            end    -- The end of the curve
            factor    -- the percentage of lateral dispersion for the curve
            resolution    -- number of recursive levels (the exponent in base 2 for number of edges of the curve)
            skeleton    -- the list of points (must be empty)
            """
    if (res > 0):
        pm = (p1 + p2) * 0.5
        ds = Vector(((p1.y - pm.y), -(p1.x - pm.x), 0.0)) * uniform(-factor, factor)
        p3 = pm + ds

        newRMDFractalPoint(p1, p3, factor, list, res - 1)
        list.append(p3) # Adding the new point here, the list will be ordered
        newRMDFractalPoint(p3, p2, factor, list, res - 1)



def newRMDFractal(origin, end, factor, resolution):
    """ Create a polyline using the Random Midpoint Displacement Fractal algorithm
        origin    -- The origin of the curve
        end    -- The end of the curve
        factor    -- the percentage of lateral dispersion for the curve
        resolution    -- number of recursive levels (the exponent in base 2 for number of edges of the curve)
        """
    skeleton = []
    newRMDFractalPoint(origin, end, factor, skeleton, resolution)
    return [origin] + skeleton + [end]


def meshFromSkeleton(skeleton, width, river_side_a, river_side_b, faces_data, name = "mesh", material = None):
    skeleton = [skeleton[0]]+skeleton
    for index in range(0, len(skeleton) - 1):
        p0 = skeleton[index]
        p1 = skeleton[index-1]
        p2 = skeleton[index + 1]

        # The param 'width' controls the width of the river, after the normalizing of it.
        # This code line is equivalent to '(p1 - p2) * cross(V(0,0,1))'
        ds = Vector(((p1.y - p2.y), -(p1.x - p2.x), 0.0)).normalized() * width
        p3 = p0 + ds
        p4 = p0 - ds

        # Here, we are creating the two river sides point lists.
        river_side_a.append(p3)
        river_side_b.append(p4)

    # Creating an ordered list of the two river sides point lists
    ordered_points = river_side_a + river_side_b[::-1]
    last_index = len(ordered_points) - 1

    # Creating the triangle faces list to pass it to the from_pydata function to generate the river mesh.
    for i in range(len(river_side_a) - 1):
        faces_data.append((i, last_index - (i + 1), i + 1))
        faces_data.append((i, last_index - (i + 1), last_index - i))

    mesh = bpy.data.meshes.new(name)
    o = bpy.data.objects.new(name, mesh)
    mesh.from_pydata(ordered_points, [], faces_data)
    mesh.update(calc_edges=True)
    if material:
        mesh.materials.append(bpy.data.materials[material])
    bpy.context.scene.objects.link(o)


def createSandCircle(center, radius):
    #create radius one circle mesh
    angle=2*3.1415927 /24
    cpoints=[Vector((radius*cos(i*angle),radius*sin(i*angle),0.01)) for i in range(24)]

    mesh = bpy.data.meshes.new("gateArena")
    mesh.from_pydata(cpoints, [], [list(range(24))])
    mesh.update(calc_edges=True)
    mesh.materials.append(bpy.data.materials["Sand"])
    o = bpy.data.objects.new("gateArena", mesh)
    o.location = center
    bpy.context.scene.objects.link(o)

###########################
# The one and only... main
def main():
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
        internalRegions = data['internalRegions']
        externalPoints = data['externalPoints']
        cityRadius = data['cityRadius']
        # This is a hack to convert dictionaries with string keys to integer.
        # Necessary because json.dump() store integer keys as strings
        regions = { int(k):v for k,v in data['regions'].items() }
        # Same hack to convert dictionaries keys to integer.
        staticRegions = { int(k):v for k,v in data['staticRegions'].items() } 
        internalSeeds = [Vector(s) for s in seeds[:len(internalRegions)]]

    ###########################################################################
    #        Create a 3D model of the city
    ###########################################################################

    #Save a copy of input data as a text buffer in blend file
    if inputFilename in bpy.data.texts:
        bpy.data.texts.remove(bpy.data.texts[inputFilename])
    with open(inputFilename, 'r') as file:
        bpy.data.texts.new(inputFilename)
        bpy.data.texts[inputFilename].from_string(file.read())
    
    #Save a copy of input AI data as a buffer in blend file
    inputFilenameAI = args['inputFilenameAI']
    if inputFilenameAI in bpy.data.texts:
        bpy.data.texts.remove(bpy.data.texts[inputFilenameAI])
    with open(inputFilenameAI, 'r') as file:
        bpy.data.texts.new(inputFilenameAI)
        bpy.data.texts[inputFilenameAI].from_string(file.read())
    
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
       
    # Enable mist in BGE graphic engine
    if args.get('enableMist', False):
        bpy.context.scene.world.mist_settings.use_mist = True
        bpy.context.scene.world.horizon_color = (0.685146, 0.800656, 0.728434)
                
    # Create exterior boundary of the city
    if args.get('createDefenseWall', False):
        print("Creating External Boundary of the City, Defense Wall")
        # Convert input data to 3D vectors
        wallVertices = [Vector(v).to_3d() for v in data['wallVertices']]

        # place defense wall. Avoid extreme corners
        axisX = Vector((1, 0))

        #Compute the position of the gate
        gate1 = wallVertices[0]
        gate2 = wallVertices[-1]
        gateMid = (gate1+gate2) * 0.5

        # Compute orientation of gate with axisX
        angGate = (gate1-gate2).xy.angle_signed(axisX)-math.pi/2
        #Insert a StoneGate object at position gateMid
        for o in bpy.data.groups["StoneGate"].objects:
            g1 = duplicateObject(o, "_Gate1_"+o.name)
            g1.location = gateMid
            g1.rotation_euler = (0, 0, angGate)
        
        #Insert one tower at gate1
        g1 = duplicateObject(bpy.data.objects["StoneTower"], "_Gate1_Tower1")
        g1.location = gate1
        g1.rotation_euler = (0, 0, angGate)
        
        # Place a door on point gate1, oriented to angR (next section of wall)
        g1 = duplicateObject(bpy.data.objects["StoneTowerDoor"], "_Door%03d_B" % i)
        g1.location = gate1
        g1.rotation_euler = (0, 0, angGate+math.pi/2)
                
        # Build the defense wall around the city
        for i in range(1, len(wallVertices)):
            v1 = wallVertices[i-1]
            v2 = wallVertices[i]
            v3 = wallVertices[(i+1) % len(wallVertices) ]
            
            # Compute orientation of both walls with axisX
            angL = (v1-v2).xy.angle_signed(axisX)
            angR = (v3-v2).xy.angle_signed(axisX)
            # Force angR > angL, so ensure that angL < average < angR
            if (angL > angR):
                angR += 6.283185307
            
            # Place a new tower on point v2 (the endpoint of this section of wall)
            g1 = duplicateObject(bpy.data.objects["StoneTower"], "_Tower%03d" % i)
            g1.location = v2
            g1.rotation_euler = (0, 0, (angL+angR)*0.5 )
            # g1.show_name = True #Debug info
            # Place a new door on point v2, oriented to angL (this section of wall)
            g1 = duplicateObject(bpy.data.objects["StoneTowerDoor"], "_Door%03d_A" % i)
            g1.location = v2
            g1.rotation_euler = (0, 0, angL)

            # Place a second door on point v2, oriented to angR (next section of wall)
            if i < len(wallVertices)-1:
                g1 = duplicateObject(bpy.data.objects["StoneTowerDoor"], "_Door%03d_B" % i)
                g1.location = v2
                g1.rotation_euler = (0, 0, angR)
            
            # Fill this section of wall with wallBlocks
            sw = duplicateAlongSegment(v1, v2, "StoneWall", gapSize=0.0, join=True)
            # print("New StoneWall section", v1.xy, "->", v2.xy, "Size: ", len(sw) )
                
            # Create a quad-mesh for streets near of this section of wall
            me = bpy.data.meshes.new("_Street")
            ob = bpy.data.objects.new("_Street", me)
            # Create a list with the four vertex of this quad
            myVertex = [v1, v2, vertices3D[externalPoints[i]], vertices3D[externalPoints[i-1]]]            
            me.from_pydata(myVertex, [], [(0,1,2,3)])
            me.update(calc_edges=True)
            me.materials.append(bpy.data.materials['Floor1'])
            bpy.context.scene.objects.link(ob)
                    
    # Create a ground around the boundary
    if args.get('createGround', False):
        createGround = args['createGround']
        groundRadius = 50 + max([v.length for v in vertices])
        makeGround([], '_groundO', '_groundM', radius=groundRadius, material='Floor3')
        print("\nDone makeGround", (datetime.now()-initTime).total_seconds() )

    # Create paths and polygon for internal regions
    print("Processing", len(internalRegions), "internalRegions")
    for nr, region in enumerate(internalRegions):
        print(nr, end=" ", flush=True)
        corners = [vertices3D[i] for i in region]
        if args.get('createStreets', False):
            if nr in staticRegions:
                #Avoid creation of collisionWall, houses and regionLabels 
                makeDistrict(corners, 1.0, 2.5, regionID=None)
            else:            
                makeDistrict(corners, 1.0, 2.5, regionID=nr)
                
        if args.get('createLeaves', False):
            createLeaves2(corners, 1.0, 2.5, density=0.4, height=0.02, objNames=["DryLeaf"], changeScale=0.4)
            # Another posible usage is to scatter obstacles all the way like
            #createLeaves2(corners, 0.0, 2.0, density=0.2, height=0.02, objNames=["DryLeaf", "Valla"], changeScale=0.3)
            
    print("\nDone internalRegions", (datetime.now()-initTime).total_seconds() )

    """
    # Merge families of objects in one object
    for objName in [ "_Street"]:
        print("Join object named", objName)
        joinObjectsByName(objName)
    """

    if args.get('createEspecialBuildings', False):
        for region, building in staticRegions.items() :
            print("createEspecialBuildings", building, "in region", region)
            importLibrary(args['input' + building], destinationLayer=0, importScripts=True)
            bpy.data.objects[building].location.xy = internalSeeds[region]

    if args.get('createRiver', False):
        distance = cityRadius * 2
        skeleton_list = newRMDFractal(Vector((-distance, distance * 2, 0.1)),
                                      Vector((-distance, -distance * 2, 0.1)),
                                      0.25, 7, [])
        meshFromSkeleton(skeleton_list, 20, [], [], [], "_River", "Water")

    if args.get('createTrail', False):
        trailWidth = 3
        roadSkel3D = [Vector(x).to_3d() for x in data['roadSkel']]        
        meshFromSkeleton(roadSkel3D, trailWidth, [], [], [], "_Trail", "Sand")
        createSandCircle(gateMid.to_3d(), (gate1-gateMid).length)

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
        if 'Target' in bpy.data.objects:
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
        player.game.properties['initPos'].value=str(list(locP.to_tuple()))

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
        player.game.controllers['playMusic'].link(sensor=player.game.sensors['playMusic'],
                                                  actuator=player.game.actuators['playMusic'])
            
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

        if 'PlayerTarget' in bpy.data.objects:
            bpy.data.objects['PlayerTarget'].hide_render = not args['debugVisibleTokens']
            
    # Check modified external scripts and update if necessary
    updateExternalTexts()

    #Save the current file, if outputGameFilename is set.
    if args.get('outputGameFilename', False):
        outputGameFilename = args['outputGameFilename']
        print('Saving blender game as:', outputGameFilename)
        bpy.ops.wm.save_as_mainfile(filepath=cwd+outputGameFilename, compress=True, copy=False)
        print ("Ready to run: blenderplayer", outputGameFilename);
            
    print("\nTotal Time:", (datetime.now()-initTime).total_seconds() )


###########################################################################
#        Call the main function
###########################################################################
if __name__ == '__main__':
    main()


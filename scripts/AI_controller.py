import bge
import numpy as np
import json, math, random
from math import sqrt
from mathutils import Vector

co = bge.logic.getCurrentController()
Player = co.owner
scene = bge.logic.getCurrentScene()
cwd = bge.logic.expandPath("//") 
globalDict = bge.logic.globalDict

def pnt2line(pnt, s1, s2):
    """ Compute coordinates of the nearest point from segment (s1-s2) to point pnt 
    pnt -- Cordinates of a point (player position, for example)
    s1  -- Cordinates of the first extreme of a segment
    s2  -- Cordinates of the second extreme of a segment
    """
    # Translate all points, so 's1' is at the origin
    line_vec = s2-s1
    pnt_vec = pnt-s1
    
    # Compute the length of the segment s1-s2
    line_len = np.linalg.norm(line_vec)
    
    # If segment is really short, give an approximate solution
    if line_len < 0.001:
        nearest = (s1+s2)/2
        return nearest
    
    # Use dot product to compute the proyection of pnt_vec over line_vec
    t = np.dot(line_vec, pnt_vec) / (line_len * line_len)
    
    # Clip t to ensure the proyection is in the range 0 to 1.
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    # Compute the position using the parameter t
    # We could also use: nearest = s1 * (1.0-t) + s2 * t
    nearest = s1 + line_vec * t
    return nearest
    
def getNearestSeed(p):    
    """ Get the index of the seed/region of the city nearest to point p
    p -- Cordinates of a point (player position, for example)
    """
    seeds = np.array(globalDict['barrierSeeds'])
    p2 = np.array([p[0], p[1]])
    nearSeed = np.linalg.norm(seeds-p2, axis=1).argmin()
    #print("Debug: Nearest Seed to", p, "is seed", nearSeed, "at position", globalDict['barrierSeeds'][nearSeed])
    return nearSeed

def getNearestCorner(p):
    """ Get the index of the vertex/corner of the city nearest to point p
    p -- Cordinates of a point (player position, for example)
    """
    p2 = np.array([p[0], p[1]])
    try:
        # Method 1: Get current region, then try only vertex of that region
        # This is better, but can fail if we are out of the city (there is no region)
        nearSeed = getNearestSeed(p2)
        myregion = globalDict['regions'][nearSeed]
        regionVertex = np.array([globalDict['vertices'][x] for x in myregion if x != -1])
        nearCorner = myregion[np.linalg.norm(regionVertex-p2, axis=1).argmin()]    
    except IndexError:
        print("You are out of the city...")
        # Method 2: Brute force trying every vertex of the city
        vertices = np.array(globalDict['vertices'])
        nearCorner = np.linalg.norm(vertices-p2, axis=1).argmin()
    #print("Debug: Nearest corner to", p, "is", nearCorner, "at position", globalDict['vertices'][nearCorner])
    return nearCorner

def getNearestStreetPoint(p):
    """ Get the coordinates of the STREET point of the city nearest to point p
    p -- Cordinates of a point (player position, for example)
    """
    p2 = np.array([p[0], p[1]])
    try:
        # Method 1: Get current region, then try only vertex of that region
        # This is better, but can fail if we are out of the city (there is no region)
        nearSeed = getNearestSeed(p2)
        myregion = globalDict['regions'][nearSeed]
        regionVertex = np.array([globalDict['vertices'][x] for x in myregion if x != -1])
        nearCorner = np.linalg.norm(regionVertex-p2, axis=1).argmin()
        v1 = myregion[(nearCorner-1)%len(myregion)]
        v2 = myregion[nearCorner]
        v3 = myregion[(nearCorner+1)%len(myregion)]
        # print("myRegion", myregion, "Near edges", (v1,v2), "and", (v2,v3))
        x1 = pnt2line(p2, np.array(globalDict['vertices'][v1]), np.array(globalDict['vertices'][v2]))
        d1 = np.linalg.norm(p2-x1)
        # print("edge", (v1,v2), "=", globalDict['vertices'][v1], globalDict['vertices'][v2] )
        # print("distance", d1, "nearpoint", x1)
        x2 = pnt2line(p2, np.array(globalDict['vertices'][v2]), np.array(globalDict['vertices'][v3]))
        d2 = np.linalg.norm(p2-x2)
        # print("edge", (v2,v3), "=", globalDict['vertices'][v2], globalDict['vertices'][v3] )
        # print("distance", d2, "nearpoint", x2)
        # Choose the min distance
        if d1 < d2:
            return d1, x1
        else:
            return d2, x2
    except IndexError:
        print("You are out of the city...")
        # Method 2: Brute force trying every vertex of the city
        vertices = np.array(globalDict['vertices'])
        nearCorner = np.linalg.norm(vertices-p2, axis=1).argmin()
        return np.linalg.norm(vertices[nearCorner]-p2), vertices[nearCorner]
    
def dumpPosition():
    """ Test function to dump the state of AI controller
    """
    p = scene.objects['Player'].position
    print("Player position", p, "In region", getNearestSeed(p), "Nearest corner", getNearestCorner(p))
    d, x = getNearestStreetPoint(p)
    print("Nearest street point", x, "at", d, "of player position")
    #Show the position visually
    if 'Target' in scene.objects:
        scene.objects['Target'].position=(x[0], x[1], 0.0)
    

def initMiniMap():
    if 'MiniMap' not in bge.logic.getCurrentScene().objects:
        return

    import Rasterizer
    height = Rasterizer.getWindowHeight()
    width = Rasterizer.getWindowWidth()
    # Compute position of miniMap viewport
    #left = int(width * 1/4)
    #bottom = int(height * 3/4)
    #right = int(width * 3/4)
    #top = int(height * 95/100)

    # Wide map (great for debug game)
    left = int(width * 3 / 100)
    bottom = int(height * 3 / 100)
    right = int(width * 97 / 100)
    top = int(height * 97 / 100)
    
    # set the viewport coordinates
    camMiniMap=bge.logic.getCurrentScene().objects['MiniMap']
    camMiniMap.setViewport(left, bottom, right, top)
    
    # move camera to position player
    myPlayer=bge.logic.getCurrentScene().objects['Player']
    camMiniMap.position[0] = myPlayer.position[0]
    camMiniMap.position[1] = myPlayer.position[1]
    
    
def init():
    """ Initialise the AI structures and insert them in the globalDict
    """
    print ("Minimap initialisation...")
    initMiniMap()
    
    print ("AI initialisation...")
    globalDict['playerInit'] = True
    
    if 'numMonster' not in globalDict : 
        globalDict['numMonster'] = 0
            
    # Try to read from cg-data.json controller
    if 'cg-data.json' in Player.controllers:
        print("Reading values from cg-data.json controller")
        data = json.loads(Player.controllers['cg-data.json'].script)
        globalDict.update(data)
        print ("Read:",[x for x in data])
        globalDict['cg-data.json'] = 'From internal text'
    else:
        print("cg-data.json controller not found. Trying external file...")
        # This is the same, but reading from an external file
        graphFilename = 'city.graph.json'
        print("Read data from: %s" % graphFilename)
        with open(graphFilename, 'r') as f:
            data = json.load(f)
            globalDict.update(data)
            print ("Read:",[x for x in data])
            globalDict['cg-data.json'] = 'From file '+graphFilename
    if 'cityName' in globalDict:
        print("City name: %s" % globalDict['cityName'])

    # Try to read from cg-ia.json controller
    if 'cg-ia.json' in Player.controllers:
        print("Reading values from cg-ia.json controller")
        ia = json.loads(Player.controllers['cg-ia.json'].script)
        globalDict.update(ia)
        print ("Read:",[x for x in ia])
        globalDict['cg-ia.json'] = 'From internal text'
    else:
        print("cg-ia.json controller not found. Try external file...")
        # This is the same, but reading from an external file
        iaFilename = 'city.AI.json'
        print("Read ia from: %s" % iaFilename)
        with open(iaFilename, 'r') as f:
            ia = json.load(f)
            globalDict.update(ia)
            print ("Read:",[x for x in ia])
            globalDict['cg-ia.json'] = 'From file '+iaFilename

    # Build InternalPoints, if not supplied by the cg-ia.json data
    if 'internalPoints' not in globalDict : 
        # Build list of internal vertex
        nv = len(globalDict['vertices'])
        internalPoints = [i for i in range(nv) if i not in globalDict['externalPoints']]

        # Remove the vertex too near of the boundary (external points)
        nearToBoundary=[]
        for j in internalPoints:
             for i in globalDict['externalPoints']:
                   # Compute distance in 2D
                   vi=globalDict['vertices'][i]
                   vj=globalDict['vertices'][j]
                   dx = vi[0]-vj[0]
                   dy = vi[1]-vj[1]
                   distance = sqrt(dx*dx+dy*dy)
                   if distance<5:
                       nearToBoundary.append(j)
                       break
        # Remove vertex that are too near from boundary
        internalPoints = [i for i in internalPoints if i not in nearToBoundary]
        globalDict['internalPoints'] = internalPoints
        print('Debug: nearToBoundary=',nearToBoundary)
        print('Debug: internalPoints=', internalPoints)

    # convert 2D vertices to 3D
    vertices3D = []
    for v in globalDict['vertices']:
        vertices3D.append((v[0], v[1], 0.0))
    globalDict['vertices3D'] = vertices3D

    # Other stuff...
    globalDict['afterDist'] = []
    globalDict['comeBack'] = 0
    globalDict['timeToWhereIGo'] = 0
    globalDict['timeGame']=0

    # Search the index of the vertex nearest to player position
    minDist = float("Inf")
    playerVertex=None
    for k,vk in enumerate(vertices3D):
        distance=Player.getDistanceTo(vk)
        if distance < minDist:
            playerVertex = k
            minDist = distance
        
    globalDict['positionI'] = playerVertex
    globalDict['positionF'] = playerVertex
    globalDict['nextPoint'] = playerVertex

    # Minimap
    if 'MiniMapOn' not in globalDict:
        globalDict['MiniMapOn']= False
    
    # End of initialization 

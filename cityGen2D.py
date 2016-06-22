#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Citymap generator from project citygen.
Functions to create a new cityMap in 2D (does not use blender stuff)
Save the data as .json file which can be read by run-cityGen3D.sh script
  
Copyright 2014 Jose M. Espadero <josemiguel.espadero@urjc.es>
Copyright 2014 Juan Ramos <juanillo07@gmail.com>
"""

import math, json, importlib, random
from math import sqrt, acos
from pprint import pprint
from datetime import datetime
import numpy as np


def newVoronoiData(numSeeds=90, cityRadius=20, numBarriers=12, LloydSteps=2, gateLen=0., randomSeed=None):
    """Create a new set of regions from a voronoi diagram
    numSeeds   -- Number of seed to be used
    cityRadius -- Approximated radius of the city
    numBarriers -- Number of barrier nodes. Usually 12.
    LloydSteps -- Number of Lloyd's relaxation steps to apply 
    gateLen    -- Size of the gates in the external wall. Use 0.0 to avoid place gates 
    """

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

    def point_inside_polygon(point, l_poly):
        x = point[0]
        y = point[1]
        inside = False
        for i in range(len(l_poly)):
            # get two vertex from the polygon
            v1 = l_poly[i - 1]
            v2 = l_poly[i]
            if (v1[1] < y <= v2[1]) or (v2[1] < y <= v1[1]):
                if x > v1[0] + (y - v1[1]) / (v2[1] - v1[1]) * (v2[0] - v1[0]):
                    inside = not inside
        return inside

    def reorderRegions(vertices, regions, seeds):
        """Reorder a list of regions to be in the same order than the seeds
           This is just stetic, because scipy.spatial.Voronoi change
           the order of the output on each call, and makes the colors
           in the plot changing in each Lloyd's step.
        """
        myRegions = []
        # Insert internal regions in the same order than seeds
        for s in seeds:
            for r in regions:
                # Reorder the internal regions
                if r and -1 not in r:
                    if point_inside_polygon(s, [vertices[i] for i in r]):
                        myRegions.append(r)
                        break
        # Insert external regions at the end
        for r in regions:
            if -1 in r:
                myRegions.append(r)

        return myRegions

    # Check scipy.spatial is instaled
    try:
        importlib.import_module('scipy.spatial')
    except ImportError:
        print("This method needs module scipy.spatial.Voronoi, but is not available.")
        return

    # Import Voronoi from scipy
    from scipy.spatial import Voronoi
    print("createNewScene (numSeeds=%d, cityRadius=%g, numBarriers=%d, LloydSteps=%d" % (
    numSeeds, cityRadius, numBarriers, LloydSteps))

    # Initialize random.seed if not given
    if randomSeed == None:
        randomSeed = np.random.randint(99999)

    # A nice example value... np.random.seed(10)
    print("Using randomSeed", randomSeed)
    np.random.seed(randomSeed)

    ###########################################################        
    # Generate random seed in a square
    seeds = 2 * cityRadius * np.random.random((numSeeds, 2)) - cityRadius

    # Min distante allowed between seeds. See documentation
    minSeedDistance = 1.9 * cityRadius / sqrt(numSeeds)
    print("minSeedDistance = ", minSeedDistance)

    # Generate the array of seeds
    for i in range(numSeeds):
        # Check the distance with previous seeds
        for j in range(i):
            dist = np.linalg.norm(seeds[i] - seeds[j])
            if dist < minSeedDistance:
                # print("Seed %d is too near of seed %d . distance %g" % (i, j, dist) )
                # Generate a new position for seed[i] and repeat the check
                seeds[i] = 2 * cityRadius * np.random.random(2) - cityRadius
                i -= 1
                break

    # Create a dense barrier of points around the seeds, to avoid far voronoi vertex
    if numBarriers > 0:
        cosines = np.cos(np.arange(0, 6.28, 6.28 / numBarriers))
        sines = np.sin(np.arange(0, 6.28, 6.28 / numBarriers))
        barrier = 1.8 * cityRadius * np.column_stack((cosines, sines))
    else:
        barrier = np.empty((0, 2))
    barrierSeeds = np.concatenate((seeds, barrier), axis=0)

    DistanciaMaxima = np.linalg.norm(np.array(barrier[1]))
    DistanciaMaxima = 0.7 * DistanciaMaxima

    # Compute initial Voronoi Diagram
    vor = Voronoi(barrierSeeds)
    vor.regions = reorderRegions(vor.vertices, vor.regions, barrierSeeds)

    # Plot initial voronoi diagram
    plotVoronoiData(vor.vertices, vor.regions, barrierSeeds, 'tmp0.initialVoronoi', radius=2 * cityRadius)

    ###########################################################        
    # Apply several steps of Lloyd's Relaxation
    # See: https://en.wikipedia.org/wiki/Lloyd's_algorithm
    for w in range(LloydSteps):
        print("Lloyd Iteration", w + 1, "of", LloydSteps)
        internalRegions = [r for r in vor.regions if r and -1 not in r]
        for region in internalRegions:
            # Compute the center of the region
            vectors = [vor.vertices[i] for i in region]
            centroid = np.average(vectors, axis=0)

            # Search the seed for this region
            # TODO: si creamos un vector ordenado relacionando region con seeds, se podria ahorrar tiempo de computo.
            nearSeed = None
            for v in range(len(seeds)):
                if point_inside_polygon(seeds[v], vectors):
                    nearSeed = v
                    # print(region, "-> seed: ", nearSeed)

            newSeed = np.array(0.5 * (seeds[nearSeed] + centroid))
            dist = np.linalg.norm(newSeed)
            if dist < DistanciaMaxima:
                seeds[nearSeed] = newSeed
            else:
                print("dist=", dist, ">= DistanciaMaxima=", DistanciaMaxima)
                # seeds=seeds+(0.0,0.0)
        # Recompute Voronoi Diagram
        barrierSeeds = np.concatenate((seeds, barrier), axis=0)
        vor = Voronoi(barrierSeeds)
        vor.regions = reorderRegions(vor.vertices, vor.regions, barrierSeeds)
        plotVoronoiData(vor.vertices, vor.regions, barrierSeeds, 'tmp1.Lloyd-Step%d' % (w + 1), radius=2 * cityRadius)

    # Compute some usefull lists
    nv = len(vor.vertices)
    vor.regions = [r for r in vor.regions if r]
    # internalRegions = [r for r in vor.regions if -1 not in r]
    externalRegions = [r for r in vor.regions if -1 in r]
    externalVertex = set([v for v in sum(externalRegions, []) if v != -1])
    # internalVertex = set([v for v in sum(internalRegions,[]) if v not in externalVertex])
    # unusedVertex = set([v for v in range(nv) if v not in externalVertex and v not in internalVertex])
    unusedVertex = set()

    ###########################################################        
    # Check and solve pairs of vertex too near...
    for i in range(nv):
        for j in range(i + 1, nv):
            dist = np.linalg.norm(vor.vertices[i] - vor.vertices[j])
            isExternalEdge = i in externalVertex and j in externalVertex
            # TODO: Avoid a hardcoded value here. Maybe 2*pi*cityRadius / len(externalVertex)
            if dist < (10.0 + 10.0 * isExternalEdge):
                print("Distance from vertex", i, "to vertex", j, "=", dist, "(external edge)" * isExternalEdge)
                # Merge voronoi vertex i and j at its center
                midpoint = 0.5 * (np.array(vor.vertices[i]) + np.array(vor.vertices[j]))
                vor.vertices[i] = midpoint
                vor.vertices[j] = midpoint
                # Mark vertex j as unused
                unusedVertex.add(j)
                print("  * Vertex", i, "and vertex", j, "merged at position:", midpoint)
                # Change all reference to vertex j to vertex i. Vertex j will remain unused.
                for region in vor.regions:
                    if j in region:
                        if i in region:
                            # print("  * Remove vertex", j, "in region ", region)
                            region.remove(j)
                        else:
                            # print("  * Usage of vertex", j, "replaced by", i, "in region", region)
                            for k, v in enumerate(region):
                                if v == j:
                                    region[k] = i

    # Remove usage of unusedVertex
    if unusedVertex:
        print("Repacking unusedVertex", unusedVertex)
        vertexToReuse = [x for x in unusedVertex if x < nv - len(unusedVertex)]
        if vertexToReuse:
            vertexToRemove = [x for x in range(nv) if x not in unusedVertex][-len(vertexToReuse):]
            print("vertexToReuse=",vertexToReuse)
            print("vertexToRemove=",vertexToRemove)

            for i, vi in enumerate(vertexToRemove):
                vj = vertexToReuse[i]
                print("Using Vertex", vj, "instead vertex", vi)
                vor.vertices[vj] = vor.vertices[vi]
                for region in vor.regions:
                    if vi in region:
                        if vj in region:
                            # print("  * Remove vertex", vi, "in region ", region)
                            region.remove(vi)
                        else:
                            # print("  * Usage of vertex", vi, "replaced by", vj, "in region", region)
                            for k, vk in enumerate(region):
                                if vk == vi:
                                    region[k] = vj

        # Remove last vertex from vertices
        nv -= len(unusedVertex)
        vor.vertices = vor.vertices[0:nv]
        print("numVertex after repacking", nv)
        externalRegions = [r for r in vor.regions if -1 in r]
        externalVertex = set([v for v in sum(externalRegions, []) if v != -1])

    # Plot data after joining near vertex
    plotVoronoiData(vor.vertices, vor.regions, barrierSeeds, 'tmp2.mergeNears', radius=2 * cityRadius)

    ###########################################################
    # compute the centroid of the voronoi set (average of seeds)
    # centroid = np.average(vor.vertices, axis=0) #option1
    # centroid = np.average(barrierSeeds, axis=0) #option2
    centroid = np.array((0, 0))  # option3
    # Get the index of the voronoi vertex nearest to the centroid
    meanPos = (np.linalg.norm(vor.vertices - centroid, axis=1)).argmin()
    meanVertex = vor.vertices[meanPos]
    print("Current centroid", centroid, "Nearest Vertex", meanVertex)
    # Traslate all voronoi vertex so there is always a vertex in (0,0)
    vertices = vor.vertices - meanVertex
    barrierSeeds = barrierSeeds - meanVertex

    # Plot data after recentering
    plotVoronoiData(vertices, vor.regions, barrierSeeds, 'tmp3.recenter', radius=2 * cityRadius)

    ###########################################################
    # Extract the list of internal and external regions
    internalRegions = []
    externalRegions = []
    # regionAreas = []
    for region in vor.regions:
        # Ignore degenerate regions
        if len(region) > 2:
            # Check if region is external
            if -1 in region:
                externalRegions.append(region)
            else:
                # Compute the signed area of this region to ensure positive orientation
                # see https://en.wikipedia.org/wiki/Shoelace_formula
                signedArea = 0
                for i in range(len(region)):
                    x1 = vertices[region[i - 1]][0]
                    y1 = vertices[region[i - 1]][1]
                    x2 = vertices[region[i]][0]
                    y2 = vertices[region[i]][1]
                    signedArea += 0.5 * (x1 * y2 - x2 * y1)
                # Use sign of signedArea to determine the orientation
                if signedArea > 0.0:
                    # print("positive area region %s" % region)
                    internalRegions.append(region)
                else:
                    # print("negative area region %s" % region)
                    signedArea = -signedArea
                    internalRegions.append(region[::-1])
                    # regionAreas.append(signedArea);

    print("internalRegions=", len(internalRegions), " externalRegions=", len(externalRegions))
    # print("internalRegionsAreas=",regionAreas)

    # Create a list of external edges
    externalEdges = []
    for region in externalRegions:
        # pprint(region)
        for i in range(len(region)):
            v1 = region[i - 1]
            v2 = region[i]
            if (v1 != -1 and v2 != -1):
                externalEdges.append((v1, v2))
                # print("New external edge %d %d" % (v1, v2))

    # Sort the list of external segments, positive orientation
    externalPoints = []
    v1 = vertices[externalEdges[0][0]]
    v2 = vertices[externalEdges[0][1]]
    if (v1[0] * v2[1] > v2[0] * v1[1]):
        iniPoint = externalEdges[0][0]
    else:
        iniPoint = externalEdges[0][1]
    while externalEdges:
        for e in externalEdges:
            if e[0] == iniPoint:
                # print("Found", iniPoint)
                externalPoints.append(iniPoint)
                iniPoint = e[1]
                externalEdges.remove(e)
                break
            if e[1] == iniPoint:
                # print("Found", iniPoint)
                externalPoints.append(iniPoint)
                iniPoint = e[0]
                externalEdges.remove(e)
                break
    # print(externalPoints)

    # Compute the signed area to ensure positive orientation of the wall
    cityArea = 0
    for i in range(len(externalPoints)):
        x1 = vertices[externalPoints[i - 1]][0]
        y1 = vertices[externalPoints[i - 1]][1]
        x2 = vertices[externalPoints[i]][0]
        y2 = vertices[externalPoints[i]][1]
        cityArea += 0.5 * (x1 * y2 - x2 * y1)

    # Reverse externalPoints if area is negative
    if (cityArea < 0):
        externalPoints = externalPoints[::-1]
        cityArea = -cityArea

    print("City Area (inside external boundary):", cityArea)

    ###########################################################
    # Compute a surrounding polygon (usefull for city walls)
    print("Creating Wall Vertices")

    def computeEnvelop(vertexList, distance=4.0):
        """ Compute the envelop (surrouding polygon at given distance)
        vertexList -- list of coordinates (or an array of  2 columns)
        distance -- Distance to displace the envelop (negative will work)
        """
        nv = len(vertexList)
        #Create a copy of input as numpy.array
        envelop = np.array(vertexList)
        # Compute the vector for each side (vertex to its previous)
        edgeP = [envelop[i]-envelop[i-1] for i in range(nv)]
        # Normalice the vector for each side
        edgeP = [x/np.linalg.norm(x) for x in edgeP]
        #Compute edge vectors (vertex to its next)
        edgeN= np.array([-edgeP[(i+1)%nv] for i in range(nv)])
        # Compute the normal to each side
        edgeNormals = np.array([(x[1], -x[0]) for x in edgeP])

        # Compute internal angles (as cosines and sines)
        alphaC = np.array([np.dot(edgeP[i],edgeN[i]) for i in range(nv)])
        alphaS = np.array([np.cross(edgeN[i],edgeP[i]) for i in range(nv)])
        # compute tangent weights as tan((pi - alpha) / 2) = sin(alpha)/(1-cos(alpha))
        w = alphaS / (1.0 - alphaC)
        
        #Compute the weighted external bisector for each vertex
        bisector = edgeNormals + np.array([w[i]*edgeP[i] for i in range(nv)])

        # Displace the external vertices by the bisector
        envelop += distance * bisector
        
        return envelop

    wallVertices = computeEnvelop([vertices[i] for i in externalPoints], 4.0)
    
    # Plot data with external wall vertices. Trick to ploat a closed line.
    wv = wallVertices.tolist()+[wallVertices[0]]
    plotVoronoiData(vertices, internalRegions, wv, 'tmp4.envelope', radius=2 * cityRadius, extraR=True)

    ###########################################################
    # Search places to place gates to the city

    if gateLen > 0:
        # Place a gate in the external corner with angle nearest to 180
        nv = len(wallVertices)        
        #Compute edge vectors (vertex to its previous)
        edgeP = [wallVertices[i]-wallVertices[i-1] for i in range(nv)]
        # Normalice the vector for each side
        edgeP = [x/np.linalg.norm(x) for x in edgeP]
        #Compute edge vectors (vertex to its next)
        edgeN= np.array([-edgeP[(i+1)%nv] for i in range(nv)])
        # Compute internal angles (as cosines) and choose the max
        bestCorner = np.arccos([np.dot(edgeP[i],edgeN[i]) for i in range(nv)]).argmax()
        print("Best corner for a gate", bestCorner, "near external vertex ->", externalPoints[bestCorner])

        #Compute the tangent as an average of side vectors
        tangent = edgeP[bestCorner]-edgeN[bestCorner]
        tangent /= np.linalg.norm(tangent)
        #Displace the vertex at the corner in the direction of tangent
        gateMid = wallVertices[bestCorner]
        gate1 = wallVertices[bestCorner] - tangent * gateLen/2
        gate2 = wallVertices[bestCorner] + tangent * gateLen/2
        wv = [gate2]+wallVertices.tolist()[bestCorner+1:] + wallVertices.tolist()[:bestCorner]+[gate1]
        plotVoronoiData(vertices, internalRegions, wv, 'tmp5.gateFlatCorner', radius=2 * cityRadius, extraR=True)    

    """ We can also displace the corner to force a 180 angle. Works well, but needs to correct internal nodes
        # Reuse previous code for select bestCorner = alphaC.argmin()
        # Proyect the corner over a segment to ensure a perfect angle of 180 
        projection = pnt2line(wallVertices[bestCorner], wallVertices[bestCorner-1], wallVertices[(bestCorner+1)%nv])
        vertices[externalPoints[bestCorner]] += projection - wallVertices[bestCorner]
        wallVertices[bestCorner] = projection
        #Compute the tangent
        tangent = wallVertices[bestCorner]-wallVertices[bestCorner-1]
        tangent /= np.linalg.norm(tangent)        
        #Displace the vertex at the corner in the direction of tangent
        gateMid = wallVertices[bestCorner]
        gate1 = wallVertices[bestCorner] - tangent * gateLen/2
        gate2 = wallVertices[bestCorner] + tangent * gateLen/2
        wv = [gate2]+wallVertices.tolist()[bestCorner+1:] + wallVertices.tolist()[:bestCorner]+[gate1]
        plotVoronoiData(vertices, internalRegions, wv, 'tmp5.gatesCorner2', radius=2 * cityRadius, extraR=True)
    # """

    """ Discarded. Tends to select short sides of external polygon, and needs to correct internal nodes
    if gateLen > 0:
        # Place a gate in the external corner nearest to the projection over
        # the segment formed  by their two neighbours
        # Similar to the error used in the Ramer–Douglas–Peucker algorithm
        # https://en.wikipedia.org/wiki/Ramer–Douglas–Peucker_algorithm
        minDist = float('Inf')
        bestCorner = None
        bestProjection = None
        nv = len(wallVertices)
        for i in range(nv):
            projection = pnt2line(wallVertices[i], wallVertices[i-1], wallVertices[(i+1)%nv])
            dist = np.linalg.norm(wallVertices[i]-projection)
            if dist < minDist:
                minDist = dist
                bestCorner = i
                bestProjection = projection
                print("New minDistance", dist, "at corner", i, "near of", externalPoints[i])
        print("Best corner for a gate", bestCorner, " near external vertex ->", externalPoints[bestCorner])

        # Displace the wallVertex and the nearest vertex inside the city
        vertices[externalPoints[bestCorner]] += bestProjection - wallVertices[bestCorner]
        wallVertices[bestCorner] = bestProjection
        #Compute the tangent at bestCorner
        tangent = wallVertices[bestCorner]-wallVertices[bestCorner-1]
        tangent /= np.linalg.norm(tangent)
        #Displace the vertex at the corner in the direction of tangent
        gateMid = wallVertices[bestCorner]
        gate1 = wallVertices[bestCorner] - tangent * gateLen/2
        gate2 = wallVertices[bestCorner] + tangent * gateLen/2
        wv = [gate2]+wallVertices.tolist()[bestCorner+1:] + wallVertices.tolist()[:bestCorner]+[gate1]
        plotVoronoiData(vertices, internalRegions, wv, 'tmp5.gatesCorner2', radius=2 * cityRadius, extraR=True)
    # """
    
    if gateLen > 0:
        # Place a gate on the midpoint of the longest external wall

        #Compute the lenght of each side of polygon wallVertices
        wallEdges = np.linalg.norm(wallVertices-np.roll(wallVertices,1, axis=0), axis=1)        
        #Search the position of max edge
        longestEdge = wallEdges.argmax()
        print("Longest Wall Edge", longestEdge, " between external vertex ->", externalPoints[longestEdge-1],externalPoints[longestEdge])
        edgeVec = (wallVertices[longestEdge] - wallVertices[longestEdge-1])
        edgeLen = np.linalg.norm(edgeVec)
        edgeVec /= edgeLen
        gate1 = wallVertices[longestEdge-1] + edgeVec * (edgeLen-gateLen)/2
        gateMid = wallVertices[longestEdge-1] + edgeVec * edgeLen/2
        gate2 = wallVertices[longestEdge-1] + edgeVec * (edgeLen+gateLen)/2
        wv = [gate2]+wallVertices.tolist()[longestEdge:] + wallVertices.tolist()[:longestEdge]+[gate1]
        plotVoronoiData(vertices, internalRegions, wv, 'tmp5.gateLongestWall', radius=2 * cityRadius, extraR=True)

    if gateLen > 0:
        # Place a gate on the midpoint of a ramdom external wall

        # Choose a random edge and ensure is long enough to put a gate there
        edge = random.randrange(len(wallVertices))
        while np.linalg.norm(wallVertices[edge] - wallVertices[edge-1]) < 3 * gateLen:
            print("Random Wall Edge", edge, "was too short...")
            edge = random.randrange(len(wallVertices))
        print("Random Wall Edge", edge, " between external vertex ->", externalPoints[edge-1],externalPoints[edge])
        edgeVec = (wallVertices[edge] - wallVertices[edge-1])
        edgeLen = np.linalg.norm(edgeVec)
        edgeVec /= edgeLen
        gate1 = wallVertices[edge-1] + edgeVec * (edgeLen-gateLen)/2
        gateMid = wallVertices[edge-1] + edgeVec * edgeLen/2
        gate2 = wallVertices[edge-1] + edgeVec * (edgeLen+gateLen)/2
        wv = [gate2]+wallVertices.tolist()[edge:] + wallVertices.tolist()[:edge]+[gate1]
        plotVoronoiData(vertices, internalRegions, wv, 'tmp5.gateRandomWall', radius=2 * cityRadius, extraR=True)

        """ This is the same than previous code, but without using numpy
        maxdist = -9999
        max_v1 = None
        max_v2 = None                
        for i in range(len(wallVertices)):
            v1 = wallVertices[i - 1]
            v2 = wallVertices[i]
            # print("Check Segment %d -> %d" % (v1, v2))
            dist = np.linalg.norm(v1 - v2)
            # TODO: Avoid a hardcoded value here. Maybe 2*pi*cityRadius / numSeeds
            if dist > maxdist:
                maxdist = dist
                max_v1 = v1
                max_v2 = v2
                # print("New max", externalPoints[i-1], externalPoints[i], "=", dist)
        print("longest external edge:", max_v1, max_v2, "->", maxdist)
        midpoint = 0.5 * (max_v1 + max_v2)
        print("midpoint %s " % (midpoint))
        plotVoronoiData(vertices, internalRegions, [midpoint], 'tmp5.gatesWall2', radius=2 * cityRadius)
    # """
        

    """ DEBUG
    # Dump regions to .off file format for external debug
    with open('internalRegions.off', 'w') as f:
        f.write("OFF %d %d 0\n" % (len(vertices), len(internalRegions)))
        for v in vertices:
            f.write("%f %f %f\n" % (v[0],v[1],0.0))
        for r in internalRegions:
            f.write("%d " % len(r))
            for v in r:
                f.write("%d "% v)
            f.write("\n")
    with open('externalPoints.off', 'w') as f:
        f.write("OFF %d %d 0\n" % (len(vertices), 1))
        for v in vertices:
            f.write("%f %f %f\n" % (v[0],v[1],0.0))
        f.write("%d " % len(externalPoints))
        for v in externalPoints:
            f.write("%d "% v)
        f.write("\n")
    #"""

    # Assemble all information as a dict
    cityData = {
    'log': "-s %d -r %f --randomSeed %d %s" % (numSeeds, cityRadius, randomSeed, datetime.now()),
    'barrierSeeds': barrierSeeds.tolist(),
    'vertices': vertices.tolist(),
    'regions': internalRegions,
    'externalPoints': externalPoints,
    'wallVertices': wallVertices.tolist(),
    }
    return cityData

def newAIData(regions, vertices):
    """Compute the matrices used to drive the AI.
    see: https://en.wikipedia.org/wiki/Adjacency_matrix    
    """

    def distance2D(p1, p2):
        """Distance between 2D points
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return sqrt(dx * dx + dy * dy)

    rangoM = (len(vertices), len(vertices))

    # Initialize adjacencyMatrix
    adjacencyMatrix = np.zeros(rangoM, dtype=np.int)
    # Initialize directDistanceMatrix
    directDistanceMatrix = np.full(rangoM, np.inf)
    np.fill_diagonal(directDistanceMatrix, 0);
    # Initialize decisionMatrix
    decisionMatrix = np.zeros(rangoM, dtype=np.int)

    # Fill adjacencyMatrix and directDistanceMatrix
    for a in regions:
        # print("region: %s" % a)
        for i in range(len(a)):
            x = a[i - 1]
            y = a[i]
            # print("edge: %d -> %d"%(x,y))
            if directDistanceMatrix[x][y] == np.inf:  # optimization
                distance = distance2D(vertices[x], vertices[y])
                adjacencyMatrix[x][y] = 1
                adjacencyMatrix[y][x] = 1
                directDistanceMatrix[x][y] = distance
                directDistanceMatrix[y][x] = distance
                decisionMatrix[x][y] = y
                decisionMatrix[y][x] = x

    # Initialize shortestPathMatrix
    shortestPathMatrix = directDistanceMatrix.copy()

    # Compute shortestPathMatrix with Floyd-Warshall algorithm
    for k in range(len(vertices)):
        decisionMatrix[k][k] = k
        for j in range(len(vertices)):
            for i in range(len(vertices)):
                dist_ikj = shortestPathMatrix[i][k] + shortestPathMatrix[k][j]
                # check if path i -> k -> j is shorter that current i -> j
                if dist_ikj < shortestPathMatrix[i][j]:
                    shortestPathMatrix[i][j] = dist_ikj
                    # when going from i to j, go through node k
                    decisionMatrix[i][j] = decisionMatrix[i][k]

    # Assemble all information as a dict of plain python matrices (lists of lists)
    AIData = {'adjacencyMatrix': adjacencyMatrix.tolist(), 'directDistanceMatrix': directDistanceMatrix.tolist(),
              'shortestPathMatrix': shortestPathMatrix.tolist(), 'decisionMatrix': decisionMatrix.tolist()}
    return AIData


def plotVoronoiData(vertices, regions, extraV=[], filename='', show=False, labels=False, radius=None, extraR=False):
    """Plot a 2D representation of voronoi data as vertices, regions, seeds
    """
    # Check matplotlib.pyplot is installed
    try:
        importlib.import_module('matplotlib.pyplot')
    except ImportError:
        print("plotVoronoiData needs module matplotlib.pyplot, but is not available.")
        return

    import matplotlib.pyplot as plt
    plt.gcf().clear()
    my_dpi = 96
    plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)

    # Plot voronoi vertex
    plt.scatter([v[0] for v in vertices], [v[1] for v in vertices], marker='.')
    if labels:
        for i, v in enumerate(vertices):
            plt.annotate(i, xy=(v[0], v[1]))

    # Plot voronoi regions
    internalRegions = [r for r in regions if r and -1 not in r]
    for r, region in enumerate(internalRegions):
        polygon = [(vertices[i][0], vertices[i][1]) for i in region]
        plt.fill(*zip(*polygon), alpha=0.2)
        # plot a label for the region
        if labels:
            plt.annotate("r%d" % r, xy=np.average(polygon, axis=0))

    # Plot barrierSeeds/extra data
    plt.scatter([s[0] for s in extraV], [s[1] for s in extraV], marker='*')  

    #Plot Extra vertex as a polygon
    if extraR:
        #plt.fill(*zip(*extraV), fill=False)
        plt.plot(*zip(*extraV), color='black')

    # Choose axis
    if radius:
        plt.axis([-radius, radius, -radius, radius])

    plt.grid()

    # Save to file
    if filename:
        if filename.endswith('.png') or filename.endswith('.svg') or filename.endswith('.jpg'):
            plt.savefig(filename, dpi=my_dpi)
        else:
            # plt.savefig(filename + '.png', dpi=my_dpi)
            plt.savefig(filename + '.svg', dpi=my_dpi)

    # Interactive plot
    if show:
        plt.show()


def plotCityData(cityData, filename='', show=True, labels=False, radius=None):
    """Plot a 2D representation of voronoiData dict
    """
    if 'vertices' in cityData:
        vertices = cityData['vertices']
    else:
        vertices = []

    if 'regions' in cityData:
        regions = cityData['regions']
    else:
        regions = []

    extraV = []
    extraR = False
    if 'wallVertices' in cityData:
        # extraV = cityData['wallVertices'] # Repeat first element to close polygon
        extraV = list(cityData['wallVertices'])+[cityData['wallVertices'][0]]
        extraR = True    
    elif 'barrierSeeds' in cityData:
        extraV = cityData['barrierSeeds']
        extraR = False

    plotVoronoiData(vertices, regions, extraV, filename, show, labels, radius, extraR=extraR)


###########################
# Main code starts here
def main():
    # read arguments from command line or interface.
    import argparse
    parser = argparse.ArgumentParser(description='Citymap generator from project citygen.')

    parser.add_argument('-s', '--numSeeds', type=int, default=10, required=False,
                        help='Number of seeds used as Voronoi input (default=10)')
    parser.add_argument('-r', '--cityRadius', type=float, default=150, required=False,
                        help='Radius of the city (default=150)')
    parser.add_argument('-g', '--gateLen', type=float, default=13.08, required=False,
                        help='Size of gates in external wall. Use 0. to avoid gates. (default=13.08)')
    parser.add_argument('-n', '--cityName', default='city', required=False,
                        help='Name of the city (default="city")')
    parser.add_argument('-v', '--show', action='store_true',
                        help='Display the map of the city')
    parser.add_argument('--randomSeed', type=int, required=False,
                        help='Initial random seed value')
    parser.add_argument('-p', '--plot', required=False,
                        help='Replot a previous generated city (default="city.grap.json")')

    args = parser.parse_args()
    # print(args)

    """
    # Create a minimal case test 
    cityData = {}
    cityData['cityName'] = "Minimal city"
    cityData['barrierSeeds'] = [[-5.0, 0.0],[5.0, 5.0],[5.0, -5.0]]
    cityData['vertices'] = [[-10.0, -10.0],[0.0, -10.0],[10.0, -10.0],[-10.0, 0.0],[0.0, 0.0],[10.0, 0.0],[-10.0, 10.0],[0.0, 10.0],[10.0, 10.0]]
    cityData['regions'] = [[0, 1, 4, 7, 6, 3], [1,2,5,4], [4,5,8,7]]
    cityData['externalPoints'] = [0, 1, 2, 5, 8, 7, 6, 3]
    """

    if not args.plot:
        # Generate a new city map
        cityData = newVoronoiData(args.numSeeds, args.cityRadius, gateLen=args.gateLen, randomSeed=args.randomSeed)
        cityData['cityName'] = args.cityName
        # Save graph data
        graphFilename = args.cityName + '.graph.json'
        print("Save graph data to: %s" % graphFilename)
        with open(graphFilename, 'w') as f:
            json.dump(cityData, f, indent=4, separators=(',', ':'), sort_keys=True)
    else:
        print("Read data from file: %s" % args.plot)
        with open(args.plot, 'r') as f:
            cityData = json.load(f)
            if 'cityName' in cityData:
                print("City name: %s" % cityData['cityName'])
                args.cityName = cityData['cityName']

    # Compute matrixes used for AI path finding
    print("Computing matrixes used for AI path finding")
    AIData = newAIData(cityData['regions'], cityData['vertices'])
    AIFilename = args.cityName + '.AI.json'
    print("Save AI matrixes to: %s" % AIFilename)
    with open(AIFilename, 'w') as f:
        json.dump(AIData, f, separators=(',', ':'), sort_keys=True)

    # Plot debug info
    plotCityData(cityData, args.cityName + '.map', show=False, labels=False, radius=2 * args.cityRadius)
    plotCityData(cityData, args.cityName + '.map.verbose', show=False, labels=True, radius=2 * args.cityRadius)


# Call the main function
if __name__ == "__main__":
    main()
    print ("Ready to run: blender --background --python cityGen3D.py");

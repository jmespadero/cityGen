#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
City map generator from project citygen.
Generate a new cityMap in 2D (does not use blender stuff)
Save the data as .json file which can be read by run-cityGen3D.sh script
  
Copyright 2014 Jose M. Espadero <josemiguel.espadero@urjc.es>
Copyright 2014 Juan Ramos <juanillo07@gmail.com>

Run option 1: (using with system python)
python3 cityGen2D.py

Run option 2: (using python bundled with blender)
blender --background --python cityGen2D.py

"""

import math, json, importlib, random
from math import sqrt, acos
from pprint import pprint
from datetime import datetime
import numpy as np

class Delaunay2D:
    """
    Class to compute a Delaunay triangulation in 2D
    ref: http://en.wikipedia.org/wiki/Bowyer-Watson_algorithm
    ref: http://www.geom.uiuc.edu/~samuelp/del_project.html
    """

    def __init__(self, center=(0, 0), radius=9999):
        """ Init and create a new frame to contain the triangulation
        center -- Optional position for the center of the frame. Default (0,0)
        radius -- Optional distance from corners to the center.
        """
        center = np.asarray(center)
        # Create coordinates for the corners of the frame
        self.coords = [center+radius*np.array((-1, -1)),
                       center+radius*np.array((+1, -1)),
                       center+radius*np.array((+1, +1)),
                       center+radius*np.array((-1, +1))]

        # Create two dicts to store triangle neighbours and circumcircles.
        self.triangles = {}
        self.circles = {}

        # Create two CCW triangles for the frame
        T1 = (0, 1, 3)
        T2 = (2, 3, 1)
        self.triangles[T1] = [T2, None, None]
        self.triangles[T2] = [T1, None, None]

        # Compute circumcenters and circumradius for each triangle
        for t in self.triangles:
            self.circles[t] = self.circumcenter(t)

    def circumcenter(self, tri):
        """Compute circumcenter and circumradius of a triangle in 2D.
        Uses an extension of the method described here:
        http://www.ics.uci.edu/~eppstein/junkyard/circumcenter.html
        """
        pts = np.asarray([self.coords[v] for v in tri])
        pts2 = np.dot(pts, pts.T)
        A = np.bmat([[2 * pts2, [[1],
                                 [1],
                                 [1]]],
                      [[[1, 1, 1, 0]]]])

        b = np.hstack((np.sum(pts * pts, axis=1), [1]))
        x = np.linalg.solve(A, b)
        bary_coords = x[:-1]
        center = np.dot(bary_coords, pts)

        # radius = np.linalg.norm(pts[0] - center) # euclidean distance
        radius = np.sum(np.square(pts[0] - center))  # squared distance
        return (center, radius)

    def inCircleFast(self, tri, p):
        """Check if point p is inside of precomputed circumcircle of tri.
        """
        center, radius = self.circles[tri]
        return np.sum(np.square(center - p)) <= radius

    def inCircleRobust(self, tri, p):
        """Check if point p is inside of circumcircle around the triangle tri.
        This is a robust predicate, slower than compare distance to centers
        ref: http://www.cs.cmu.edu/~quake/robust.html
        """
        m1 = np.asarray([self.coords[v] - p for v in tri])
        m2 = np.sum(np.square(m1), axis=1).reshape((3, 1))
        m = np.hstack((m1, m2))    # The 3x3 matrix to check
        return np.linalg.det(m) <= 0

    def addPoint(self, p):
        """Add a point to the current DT, and refine it using Bowyer-Watson.
        """
        p = np.asarray(p)
        idx = len(self.coords)
        # print("coords[", idx,"] ->",p)
        self.coords.append(p)

        # Search the triangle(s) whose circumcircle contains p
        bad_triangles = []
        for T in self.triangles:
            # Choose one method: inCircleRobust(T, p) or inCircleFast(T, p)
            if self.inCircleFast(T, p):
                bad_triangles.append(T)

        # Find the CCW boundary (star shape) of the bad triangles,
        # expressed as a list of edges (point pairs) and the opposite
        # triangle to each edge.
        boundary = []
        # Choose a "random" triangle and edge
        T = bad_triangles[0]
        edge = 0
        # get the opposite triangle of this edge
        while True:
            # Check if edge of triangle T is on the boundary...
            # if opposite triangle of this edge is external to the list
            tri_op = self.triangles[T][edge]
            if tri_op not in bad_triangles:
                # Insert edge and external triangle into boundary list
                boundary.append((T[(edge+1) % 3], T[(edge-1) % 3], tri_op))

                # Move to next CCW edge in this triangle
                edge = (edge + 1) % 3

                # Check if boundary is a closed loop
                if boundary[0][0] == boundary[-1][1]:
                    break
            else:
                # Move to next CCW edge in opposite triangle
                edge = (self.triangles[tri_op].index(T) + 1) % 3
                T = tri_op

        # Remove triangles too near of point p of our solution
        for T in bad_triangles:
            del self.triangles[T]
            del self.circles[T]

        # Retriangle the hole left by bad_triangles
        new_triangles = []
        for (e0, e1, tri_op) in boundary:
            # Create a new triangle using point p and edge extremes
            T = (idx, e0, e1)

            # Store circumcenter and circumradius of the triangle
            self.circles[T] = self.circumcenter(T)

            # Set opposite triangle of the edge as neighbour of T
            self.triangles[T] = [tri_op, None, None]

            # Try to set T as neighbour of the opposite triangle
            if tri_op:
                # search the neighbour of tri_op that use edge (e1, e0)
                for i, neigh in enumerate(self.triangles[tri_op]):
                    if neigh:
                        if e1 in neigh and e0 in neigh:
                            # change link to use our new triangle
                            self.triangles[tri_op][i] = T

            # Add triangle to a temporal list
            new_triangles.append(T)

        # Link the new triangles each another
        N = len(new_triangles)
        for i, T in enumerate(new_triangles):
            self.triangles[T][1] = new_triangles[(i+1) % N]   # next
            self.triangles[T][2] = new_triangles[(i-1) % N]   # previous

    def exportTriangles(self):
        """Export the current list of Delaunay triangles
        """
        # Filter out triangles with any vertex in the extended BBox
        return [(a-4, b-4, c-4)
                for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]

    def exportCircles(self):
        """Export the circumcircles as a list of (center, radius)
        """
        # Remember to compute circumcircles if not done before
        # for t in self.triangles:
        #     self.circles[t] = self.circumcenter(t)

        # Filter out triangles with any vertex in the extended BBox
        # Do sqrt of radius before of return
        return [(self.circles[(a, b, c)][0], sqrt(self.circles[(a, b, c)][1]))
                for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]

    def exportDT(self):
        """Export the current set of Delaunay coordinates and triangles.
        """
        # Filter out coordinates in the extended BBox
        coord = self.coords[4:]

        # Filter out triangles with any vertex in the extended BBox
        tris = [(a-4, b-4, c-4)
                for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]
        return coord, tris

    def exportExtendedDT(self):
        """Export the Extended Delaunay Triangulation (with the frame vertex).
        """
        return self.coords, list(self.triangles)
        
    def exportVoronoiRegions(self):
        """Export coordinates and regions of Voronoi diagram as indexed data.
        """
        # Remember to compute circumcircles if not done before
        # for t in self.triangles:
        #     self.circles[t] = self.circumcenter(t)
        useVertex = {i:[] for i in range(len(self.coords))}
        vor_coors = []
        index={}
        # Build a list of coordinates and a index per triangle/region
        for tidx, (a, b, c) in enumerate(self.triangles):
            vor_coors.append(self.circles[(a,b,c)][0])
            # Insert triangle, rotating it so the key is the "last" vertex 
            useVertex[a]+=[(b, c, a)]
            useVertex[b]+=[(c, a, b)]
            useVertex[c]+=[(a, b, c)]
            # Set tidx as the index to use with this triangles
            index[(a, b, c)] = tidx
            index[(c, a, b)] = tidx
            index[(b, c, a)] = tidx
            
        # init regions per coordinate dictionary
        regions = {}
        # Sort each region in a coherent order, and substitude each triangle
        # by its index
        for i in range (4, len(self.coords)):
            v = useVertex[i][0][0]  # Get a vertex of a triangle
            r=[]
            for _ in range(len(useVertex[i])):
                # Search the triangle beginning with vertex v
                t = [t for t in useVertex[i] if t[0] == v][0]
                r.append(index[t])  # Add the index of this triangle to region
                v = t[1]            # Choose the next vertex to search
            regions[i-4]=r          # Store region.
            
        return vor_coors, regions

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
    dt = Delaunay2D(radius = 10 * cityRadius)
    
    # Insert all seeds and barriers one by one
    for s in barrierSeeds:
        dt.addPoint(s)
        
    # Get the voronoi regions
    vor_vertices, vor_regions = dt.exportVoronoiRegions()
    internalRegions = [vor_regions[r] for r in range(len(seeds))]

    # Plot initial voronoi diagram
    plotVoronoiData(vor_vertices, internalRegions, barrierSeeds, 'tmp0.initialVoronoi', radius=2 * cityRadius)

    ###########################################################        
    # Apply several steps of Lloyd's Relaxation
    # See: https://en.wikipedia.org/wiki/Lloyd's_algorithm
    for w in range(LloydSteps):
        print("Lloyd Iteration", w + 1, "of", LloydSteps)
        for r,region in enumerate(internalRegions):
            # Compute the center of the region
            vectors = [vor_vertices[i] for i in region]
            centroid = np.average(vectors, axis=0)

            # Relax the seed for this region
            newSeed = np.array(0.5 * (seeds[r] + centroid))
            dist = np.linalg.norm(newSeed)
            if dist < DistanciaMaxima:
                seeds[r] = newSeed
            else:
                print("dist=", dist, ">= DistanciaMaxima=", DistanciaMaxima)

        # Recompute Voronoi Diagram
        barrierSeeds = np.concatenate((seeds, barrier), axis=0)
        dt = Delaunay2D(radius = 10 * cityRadius)
        for s in barrierSeeds:
            dt.addPoint(s)
        vor_vertices, vor_regions = dt.exportVoronoiRegions()
        internalRegions = [vor_regions[r] for r in range(len(seeds))]

        # Plot initial voronoi diagram
        plotVoronoiData(vor_vertices, internalRegions, barrierSeeds, 'tmp1.Lloyd-Step%d' % (w + 1), radius=2 * cityRadius)
    
    # Compute some usefull lists
    nv = len(vor_vertices)
    externalRegions = [vor_regions[r] for r in range(numSeeds, numSeeds+numBarriers)]
    externalVertex = set([v for v in sum(externalRegions, []) if v != -1])
    # internalVertex = set([v for v in sum(internalRegions,[]) if v not in externalVertex])
    # unusedVertex = set([v for v in range(nv) if v not in externalVertex and v not in internalVertex])
    unusedVertex = set()

    ###########################################################        
    # Check and solve pairs of vertex too near...
    print("Check and merge pairs of vertex too near...")
    for i in range(nv):
        for j in range(i + 1, nv):
            dist = np.linalg.norm(vor_vertices[i] - vor_vertices[j])
            isExternalEdge = i in externalVertex and j in externalVertex
            # TODO: Avoid a hardcoded value here. Maybe 2*pi*cityRadius / len(externalVertex)
            if dist < (10.0 + 10.0 * isExternalEdge):
                print("Distance from vertex", i, "to vertex", j, "=", dist, "(external edge)" * isExternalEdge)
                # Merge voronoi vertex i and j in the midpoint
                midpoint = 0.5 * (np.array(vor_vertices[i]) + np.array(vor_vertices[j]))
                vor_vertices[i] = midpoint
                vor_vertices[j] = midpoint
                # print("  * Vertex", i, "and vertex", j, "merged at position:", midpoint)
                # Mark vertex j as unused
                unusedVertex.add(j)
                # Change all references to vertex j to vertex i. Vertex j will remain unused.
                for r in vor_regions:
                    if j in vor_regions[r]:
                        if i in vor_regions[r]:
                            # print("  * Remove vertex", j, "in region ", region)
                            vor_regions[r].remove(j)
                        else:
                            # print("  * Usage of vertex", j, "replaced by", i, "in region", region)
                            for k, v in enumerate(vor_regions[r]):
                                if v == j:
                                    vor_regions[r][k] = i

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
                vor_vertices[vj] = vor_vertices[vi]
                for r in vor_regions:
                    if vi in vor_regions[r]:
                        if vj in vor_regions[r]:
                            # print("  * Remove vertex", vi, "in region ", region)
                            vor_regions[r].remove(vi)
                        else:
                            # print("  * Usage of vertex", vi, "replaced by", vj, "in region", region)
                            for k, vk in enumerate(vor_regions[r]):
                                if vk == vi:
                                    vor_regions[r][k] = vj

        # Remove last vertex from vertices
        nv -= len(unusedVertex)
        vor_vertices = vor_vertices[0:nv]
        print("numVertex after repacking", nv)
        externalRegions = [vor_regions[r] for r in range(numSeeds, len(vor_regions))]
        externalVertex = set([v for v in sum(externalRegions, []) if v != -1])

    # Plot data after joining near vertex
    plotVoronoiData(vor_vertices, internalRegions, barrierSeeds, 'tmp2.mergeNears', radius=2 * cityRadius)

    ###########################################################
    # Extract the list of internal and external regions
    internalRegions = [vor_regions[r] for r in range(numSeeds)]
    externalRegions = [vor_regions[r] for r in range(numSeeds, len(vor_regions))]

    print("internalRegions=", len(internalRegions), " externalRegions=", len(externalRegions))
    # print("internalRegionsAreas=",regionAreas)

    # Build the list of edges in internal regions
    vor_edges = []
    for r in internalRegions:
        # add all edges of regions
        vor_edges += list(zip(r[-1:]+r[:-1], r))

    # Build the list of external edges (as a dict)
    externalEdgesDict = {a:b for (a,b) in vor_edges if (b, a) not in vor_edges}

    # sort the edges in CCW order and extract the external vertex
    v = next(iter(externalEdgesDict))  # get a random key in the dict
    externalPoints = []
    for _ in range(len(externalEdgesDict)):
        externalPoints.append(v)      # Add vertex to boundary
        v = externalEdgesDict[v]      # go to next vertex
    print("externalPoints:", externalPoints)
    
    ###########################################################
    # Smooth externalPoints to get a rounder shape.
    externalRadius = 0
    for i in externalPoints:
        externalRadius += np.linalg.norm(vor_vertices[i])
    externalRadius /= len(externalPoints)
    print("Average external radius", externalRadius)
    
    for i in externalPoints:
        r = np.linalg.norm(vor_vertices[i])
        # 75% of original position + 25% circle position
        vor_vertices[i] *= 0.75 + 0.25 * externalRadius / r

    # Plot data after recentering
    plotVoronoiData(vor_vertices, internalRegions, barrierSeeds, 'tmp3.1.smooth', radius=2 * cityRadius)

    ###########################################################
    # compute the centroid of the voronoi set (average of seeds)
    # centroid = np.average(vor.vertices, axis=0) #option1
    # centroid = np.average(barrierSeeds, axis=0) #option2
    centroid = np.array((0, 0))  # option3
    # Get the index of the voronoi vertex nearest to the centroid
    meanPos = (np.linalg.norm(vor_vertices - centroid, axis=1)).argmin()
    meanVertex = vor_vertices[meanPos]
    print("Current centroid", centroid, "Nearest Vertex", meanVertex)
    # Traslate all voronoi vertex so there is always a vertex in (0,0)
    vertices = vor_vertices - meanVertex
    barrierSeeds = barrierSeeds - meanVertex

    # Plot data after recentering
    plotVoronoiData(vertices, internalRegions, barrierSeeds, 'tmp3.2.recenter', radius=2 * cityRadius)

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

    def computeEnvelope(vertexList, distance=4.0):
        """ Compute the envelope (surrounding polygon at given distance)
        vertexList -- list of coordinates (or an array of 2 columns)
        distance -- Distance to displace the envelope (negative will work)
        """
        nv = len(vertexList)
        #Create a copy of input as numpy.array
        envelope = np.array(vertexList)
        # Compute the vector for each side (vertex to its previous)
        edgeP = [envelope[i]-envelope[i-1] for i in range(nv)]
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
        envelope += distance * bisector
        
        return envelope

    wallVertices = computeEnvelope([vertices[i] for i in externalPoints], 4.0)
    
    # Plot data with external wall vertices. Tricked to plot a closed line.
    wv = wallVertices.tolist()+[wallVertices[0]]
    plotVoronoiData(vertices, internalRegions, wv, 'tmp4.envelope', radius=2 * cityRadius, extraR=True)

    ###########################################################
    # Search places to place gates to the city

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

    """ OK, but will prefer gates on corners
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
    # """
        
    if gateLen > 0:
        # Place a gate in the external corner with angle nearest to 180
        nv = len(wallVertices)        
        #Compute edge vectors (vertex to its previous)
        edgeP = [wallVertices[i]-wallVertices[i-1] for i in range(nv)]
        # Normalice the vector for each side
        edgeP = [x/np.linalg.norm(x) for x in edgeP]
        #Compute edge vectors (vertex to its next)
        edgeN= np.array([-edgeP[(i+1)%nv] for i in range(nv)])
        # Compute corner angles (as arccosines, will clip to 180) and choose the max angle
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

        # Change wallVertices, so choose this gate
        wallVertices = np.array(wv)
        externalPoints = externalPoints[bestCorner:]+externalPoints[:bestCorner+1]

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

    # Initialize adjacencyMatrix as a sparse matrix
    neighbours = { v:set() for v in range(len(vertices)) }
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
                neighbours[x].add(y)
                neighbours[y].add(x)
                directDistanceMatrix[x][y] = distance
                directDistanceMatrix[y][x] = distance
                decisionMatrix[x][y] = y
                decisionMatrix[y][x] = x

    # Convert sets to lists
    neighbours = {int(v):list(neighbours[v]) for v in neighbours}
    
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
    AIData = {'neighbours': neighbours, 'directDistanceMatrix': directDistanceMatrix.tolist(),
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
    for r, region in enumerate(regions):
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
            plt.savefig(filename, dpi=my_dpi, bbox_inches='tight')
        else:
            # plt.savefig(filename + '.png', dpi=my_dpi, bbox_inches='tight')
            plt.savefig(filename + '.svg', dpi=my_dpi, bbox_inches='tight')

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
        extraV = list(cityData['wallVertices'])
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
    parser.add_argument('--background', required=False, action='store_true')
    parser.add_argument('--python', required=False)

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

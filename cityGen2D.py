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

TODO:
* Try a Poison-disk sampling for generating the random seeds
  see: https://sighack.com/post/poisson-disk-sampling-bridsons-algorithm
"""

import math, json, importlib, random
from math import sqrt, acos, ceil
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



class CityData(dict):
    """
    Class to compute a new cityData map in 2D
    """

    def __init__(self, args, numBarriers=12, LloydSteps=4):
        """Create a new set of regions from a voronoi diagram
        args.numSeeds   -- Number of seed to be used
        args.cityRadius -- Approximated radius of the city
        args.numBarriers -- Number of barrier seeds. Usually 12.
        args.LloydSteps -- Number of Lloyd's relaxation steps to apply 
        args.gateLen    -- Size of the gates in the external wall. Use 0.0 to avoid place gates
        args.randomSeed -- Random seed (to make deterministic)
        args.debugSVG   -- Create debug SVG files on each step.
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

        # Extract variables from args
        numSeeds = args.numSeeds
        cityRadius = args.cityRadius
        randomSeed = args.randomSeed
        debugSVG = args.debug
        debugSVG = args.debug
        
        print("createNewScene (numSeeds=%d, cityRadius=%g, numBarriers=%d, LloydSteps=%d" % (
        numSeeds, cityRadius, numBarriers, LloydSteps))

        # Initialize random.seed if not given
        if randomSeed == None:
            randomSeed = np.random.randint(99999)

        # A nice example value... np.random.seed(10)
        print("Using randomSeed", randomSeed)
        np.random.seed(randomSeed)

        # Min distante allowed between seeds. See documentation
        minSeedDistance = ceil(1.9 * cityRadius / sqrt(numSeeds))
        print("minSeedDistance = ", minSeedDistance)

        ###########################################################        
        # Generate random seeds in a square an store as a numpy array n x 2
        seeds = 2 * cityRadius * np.random.random((numSeeds, 2)) - cityRadius

        # Generate seed position for fixed regions (temple, market, etc...)
        print("Creating requested fixed regions:", args.models)
        numFixedSeeds = 0
        staticRegions = {}

        for name in args.models:
            # Load relative seeds from "cg-XXXXXXX.json" file
            with open("cg-" + name + ".json", 'r') as f:
                regionSeeds = np.array([[0, 0]] + json.load(f))
                # Compute the radius of this set of seeds
                radius = minSeedDistance/2 + max([np.linalg.norm(x) for x in regionSeeds])
                # print("Read file cg-" + name + ".json", " -> radius",radius)

                #Find a position in plane with no previous seeds nearest than radius
                pos = np.asarray([0.0, 0.0])
                if len(staticRegions) > 0:
                    # Compute distances from point pos to center of previous regions
                    # print("previous fixed regions", [r[2] for r in staticRegions.values()])
                    # r[2] is position of fixedRegion. r[1] is radius of fixedRegion
                    while (min([np.linalg.norm(r[2] - pos) - r[1] for r in staticRegions.values()]) < radius):
                        # print("Invalid pos. Repeat...")
                        pos = (1.5 * cityRadius * np.random.random(2) - cityRadius / 2).round(2)

                # Displace regionSeeds to pos and store it into the static seeds list
                seeds[numFixedSeeds:numFixedSeeds+len(regionSeeds)] = regionSeeds + pos

                # Debug info
                print(" * Build", name, "in region", numFixedSeeds, "position", pos, "radius", radius)
                staticRegions[numFixedSeeds] = [name, radius, pos]
                numFixedSeeds += len(regionSeeds)        
            
        # TODO: Try a method to create points not relaying in while(random) 
        # Generate the non-fixed seeds and check none is too near of previous seeds
        for i in range(numFixedSeeds, numSeeds):
            # Check minimun distance from seed[i] to previous seeds
            while(min(np.linalg.norm(seeds[0:i]-seeds[i], axis=1)) < minSeedDistance):
                #print("Seed",  i, "is too near of previous seeds.")
                # Generate a new position for seed[i] and repeat the check
                seeds[i] = 2 * cityRadius * np.random.random(2) - cityRadius                                           

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
        if debugSVG:
            #plotVoronoiData(vor_vertices, [], seeds, 'tmp0.1.seeds', cityRadius)
            #plotVoronoiData(vor_vertices, [], barrierSeeds, 'tmp0.2.barrierSeeds', cityRadius)
            plotVoronoiData(vor_vertices, internalRegions, barrierSeeds, 'tmp0.initialVoronoi', cityRadius)

        ###########################################################        
        # Apply several steps of Lloyd's Relaxation to non-fixed regions
        # See: https://en.wikipedia.org/wiki/Lloyd's_algorithm
        for w in range(LloydSteps):
            print("Lloyd Iteration", w + 1, "of", LloydSteps)
            for r in range(numFixedSeeds, len(internalRegions)):
                # Compute the center of the region
                centroid = np.average([vor_vertices[i] for i in internalRegions[r]], axis=0)

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
            if debugSVG:
                plotVoronoiData(vor_vertices, internalRegions, barrierSeeds, 'tmp1.Lloyd-Step%d' % (w + 1), cityRadius)
        
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
                    print("  Distance from vertex", i, "to vertex", j, "=", dist, "(external edge)" * isExternalEdge)
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
            print("  Repacking unusedVertex", unusedVertex)
            vertexToReuse = [x for x in unusedVertex if x < nv - len(unusedVertex)]
            if vertexToReuse:
                vertexToRemove = [x for x in range(nv) if x not in unusedVertex][-len(vertexToReuse):]
                #print("vertexToReuse=",vertexToReuse)
                #print("vertexToRemove=",vertexToRemove)

                for i, vi in enumerate(vertexToRemove):
                    vj = vertexToReuse[i]
                    #print("Using Vertex", vj, "instead vertex", vi)
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
            print("  numVertex after repacking", nv)
            externalRegions = [vor_regions[r] for r in range(numSeeds, len(vor_regions))]
            externalVertex = set([v for v in sum(externalRegions, []) if v != -1])

        # Plot data after joining near vertex
        if debugSVG:
            plotVoronoiData(vor_vertices, internalRegions, barrierSeeds, 'tmp2.mergeNears', cityRadius)

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
        # Smooth externalPoints distance to origin to get a rounder shape.
        externalRadius = sum([np.linalg.norm(vor_vertices[i]) for i in externalPoints])
        externalRadius /= len(externalPoints)
        print("Average external radius", externalRadius)
        
        for i in externalPoints:
            r = np.linalg.norm(vor_vertices[i])
            # 75% of original position + 25% circle position
            vor_vertices[i] *= 0.75 + 0.25 * externalRadius / r

        # Plot data after smoothing
        if debugSVG:
            plotVoronoiData(vor_vertices, internalRegions, barrierSeeds, 'tmp3.1.smooth', cityRadius)

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
        if debugSVG:
            plotVoronoiData(vertices, internalRegions, barrierSeeds, 'tmp3.2.recenter', cityRadius)

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
        if debugSVG:
            plotVoronoiData(vertices, internalRegions, wv, 'tmp4.envelope', cityRadius, extraR=True)

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
            if debugSVG:
                plotVoronoiData(vertices, internalRegions, wv, 'tmp5.gatesCorner2', cityRadius, extraR=True)
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
            if debugSVG:
                plotVoronoiData(vertices, internalRegions, wv, 'tmp5.gateLongestWall', cityRadius, extraR=True)

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
            if debugSVG:
                plotVoronoiData(vertices, internalRegions, wv, 'tmp5.gateRandomWall', cityRadius, extraR=True)
        # """
            
        if args.gateLen > 0:
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
            gate1 = wallVertices[bestCorner] - tangent * args.gateLen/2
            gate2 = wallVertices[bestCorner] + tangent * args.gateLen/2
            wv = [gate2]+wallVertices.tolist()[bestCorner+1:] + wallVertices.tolist()[:bestCorner]+[gate1]
            if debugSVG:
                plotVoronoiData(vertices, internalRegions, wv, 'tmp5.gateFlatCorner', cityRadius, extraR=True)

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
            if debugSVG:
                plotVoronoiData(vertices, internalRegions, wv, 'tmp5.gatesCorner2', cityRadius, extraR=True)
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

        def RMDF_Point(a, b, noiseFactor=0.0):
            """
            Compute midpoint of segment [a,b] displaced by a random noise factor
            """
            # Compute the midpoint
            midPoint = 0.5 * np.array(b+a)
            # Compute the orientation of displacement, perpendicular to segment a->b
            disp = np.array(b-a)
            # Rotate d around Z axis
            if disp.size > 1:
                tmp = disp[0]
                disp[0] = -disp[1]
                disp[1] = tmp
                
            # Compute randomly displaced midpoint
            return midPoint + disp * noiseFactor * (np.random.random_sample() - 0.5)

        def RMDF_Polyline(L, maxDistance, noiseFactor=0.0, circular=False):
            """
            Compute Random Midpoint Displacement Fractal for each segment of a polyline.

            Args:
                L (list): A list of vertex (scalar or vector)
                maxDistance (float): Max length allowed for each segment in result.
                noiseFactor (float) : Noise strength used in the displacement. 
                circular (bool): See L as a closed (circular) polyline.

            Returns:
                list: The list of vertex for the RMDF subdivision
            """
            
            # Repeat first element to build a closed polyline
            if circular:
                #L = L + [L[0]]
                L = np.append(L, [L[0]], axis=0)
                    
            # Repeat subdivision while any segment is longer than maxDistance
            doAgain=True
            while doAgain:
                doAgain=False
                L2=[]
                # Iterate over pairs of elements of L
                for i in range(len(L)-1):
                    L2.append(L[i])
                    #distance=np.sqrt((L[i+1]-L[i])*(L[i+1]-L[i]))
                    #Check if segment L[i]->L[i+1] should be subdivided
                    distance = np.linalg.norm(L[i+1]-L[i])
                    if distance > maxDistance:        
                        L2.append(RMDF_Point(L[i], L[i+1], noiseFactor))
                        doAgain=True
                # Append last element of polyline
                L = L2 + [L[-1]]

            return L

        # Add a road to the door
        gatePosition = (wallVertices[0]+wallVertices[-1])/2
        #Build a segment that go out of the city
        roadSkel = [gatePosition, 3*gatePosition]
        #Random Midpoint Displacement Fractal previous roadSkel
        roadSkel = np.array(RMDF_Polyline(roadSkel, 25, noiseFactor=0.4))
        
        """
        if args.get('createTrail', False):
            origin = gateMid.to_3d()
            trailWidth = 5

            createSandCircle(gateMid.to_3d(), 2*(gate1-gateMid).length)
            skeleton_list = newRMDFractal(origin, (origin * 3), 0.20, 7, [])
            meshFromSkeleton(skeleton_list, trailWidth, [], [], [], "_Trail", "Sand")
        """

        # Assemble all information as a dict
        data  = {
        'log': "-s %d -r %f --randomSeed %d %s" % (numSeeds, cityRadius, randomSeed, datetime.now()),
        'seeds': barrierSeeds.tolist(),
        'vertices': vertices.tolist(),
        'regions': vor_regions,    
        'internalRegions': internalRegions,
        'externalPoints': externalPoints,
        'wallVertices': wallVertices.tolist(),
        'roadSkel':roadSkel.tolist(),
        'staticRegions': { k:v[0] for k,v in staticRegions.items() }  ,
        'cityRadius': cityRadius,
        }
        self.update(data)
        self.data = data

    def exportJSON(self, filename):
        """Save data to JSON to be read by cityGen3D
        """
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=4, separators=(',', ':'), sort_keys=True)
        
    def exportSVG(self, filename='', labels=False, radius=None):
        """Plot a 2D representation of cityData dict
        """
            
        #Coordinates or the origin (center of the image)
        OX = OY = 3 * radius
            
        svgHeader = '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" >\n'%(2*OX,2*OY)
        svgHeader += '<rect id="background" width="100%" height="100%" style="fill:white"/>\n'
        svgFooter = '<line x1="50%" y1="5%" x2="50%" y2="95%" stroke-dasharray="1, 5" style="stroke:black;" />\n'
        svgFooter += '<line x1="5%" y1="50%" x2="95%" y2="50%" stroke-dasharray="1, 5" style="stroke:black" />\n'
        svgFooter += '</svg>\n'
        
        svgRegions = '<g id="regions" style="fill:#ffeeaa;stroke:black;stroke-width:1">\n'
        svgLabels = '<g id="labels" style="fill:black;text-anchor:middle">\n'
        palette=["#9c9fff", "#ff89b5", "#ffdc89", "#90d4f7", "#71e096", "#f5a26f", "#ed6d79", "#cff381"]
        
        # Plot voronoi regions
        vertices = self['vertices']
        for r, region in enumerate(self['internalRegions']):
            polygon = [(OX+vertices[i][0], OY-vertices[i][1]) for i in region]
            svgRegions += '  <polygon style="fill:'+palette[r%len(palette)]
            svgRegions += '" points="' + ' '.join("%g,%g" % v for v in polygon) 
            svgRegions += '" />\n'
            if labels:
                # plot a label for the region in the centroid of the region
                xy=np.average(polygon, axis=0)
                svgLabels += '<text x="%g" y="%g">r%d</text>\n' % (xy[0], xy[1], r)
        
        # Labels for voronoi vertex 
        if labels:
            for i, v in enumerate(self['vertices']):
                svgLabels += '<text x="%g" y="%g">%d</text>\n' % (OX+v[0], OY-v[1], i)

        extraData = []
        if 'wallVertices' in self:
            extraData.append((list(self['wallVertices']), True, "black"))
        if 'roadSkel' in self:
            extraData.append((list(self['roadSkel']), True, "brown"))
        if 'barrierSeeds' in self:
            extraData.append((self['barrierSeeds'], False))

        # Plot extra data
        for extraV, extraR, color in extraData:
            #Plot Extra vertex as a polygon
            if extraR:
                svgRegions += '  <polyline style="fill:none;stroke:%s;stroke-width:2"' % color
                svgRegions += ' points="' + ' '.join("%g,%g"%(OX+v[0],OY-v[1]) for v in extraV) 
                svgRegions += '" />\n'
                
            # Plot barrierSeeds/extra data
            for v in extraV:
                svgRegions += '<circle cx="%g" cy="%g" r="3" stroke="%s" stroke-width="1" fill="red" />' % (OX+v[0], OY-v[1], color)

        if not filename.endswith('.svg'):
            filename += ".svg"

        with open(filename, "w") as svg_file:
            svg_file.write(svgHeader+svgRegions+'\n</g>\n'+svgLabels+'\n</g>\n'+svgFooter)
            
def plotVoronoiData(vertices, regions, extraV, filename, radius, labels=False, extraR=False):
    """Plot a 2D representation of voronoi data as vertices, regions, seeds
    """   
    radius = 2*radius
        
    svgHeader = '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" >\n'%(2*radius,2*radius)
    svgHeader += '<rect id="background" width="100%" height="100%" style="fill:white"/>\n'
    svgFooter = '<line x1="50%" y1="5%" x2="50%" y2="95%" stroke-dasharray="1, 5" style="stroke:black;" />\n'
    svgFooter += '<line x1="5%" y1="50%" x2="95%" y2="50%" stroke-dasharray="1, 5" style="stroke:black" />\n'
    svgFooter += '</svg>\n'
    
    svgRegions = '<g id="regions" style="fill:#ffeeaa;stroke:black;stroke-width:1">\n'
    svgLabels = '<g id="labels" style="fill:black;text-anchor:middle">\n'
    palette=["#9c9fff", "#ff89b5", "#ffdc89", "#90d4f7", "#71e096", "#f5a26f", "#ed6d79", "#cff381"]

    # Plot voronoi regions
    for r, region in enumerate(regions):
        polygon = [(radius+vertices[i][0], radius-vertices[i][1]) for i in region]
        svgRegions += '  <polygon style="fill:'+palette[r%len(palette)]
        svgRegions += '" points="' + ' '.join("%g,%g" % v for v in polygon) 
        svgRegions += '" />\n'
        if labels:
            # plot a label for the region in the centroid of the region
            xy=np.average(polygon, axis=0)
            svgLabels += '<text x="%g" y="%g">r%d</text>\n' % (xy[0], xy[1], r)
    
    # Labels for voronoi vertex 
    if labels:
        for i, v in enumerate(vertices):
            svgLabels += '<text x="%g" y="%g">%d</text>\n' % (radius+v[0], radius-v[1], i)

    #Plot Extra vertex as a polygon
    if extraR:
        svgRegions += '  <polyline style="fill:none;stroke:black;stroke-width:2"'
        svgRegions += ' points="' + ' '.join("%g,%g"%(radius+v[0],radius-v[1]) for v in extraV) 
        svgRegions += '" />\n'
        
    # Plot barrierSeeds/extra data
    for v in extraV:
        svgRegions += '<circle cx="%g" cy="%g" r="3" stroke="black" stroke-width="1" fill="red" />' % (radius+v[0], radius-v[1])

    if not filename.endswith('.svg'):
        filename += ".svg"

    with open(filename, "w") as svg_file:
        svg_file.write(svgHeader+svgRegions+'\n</g>\n'+svgLabels+'\n</g>\n'+svgFooter)
    
def newAIData(regions, vertices):
    """Compute the matrices used to drive the AI.
    see: https://en.wikipedia.org/wiki/Adjacency_matrix    
    """

    def distance2D(p1, p2):
        """Euclidean distance between 2D points"""
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




###########################
# Main code starts here
def main():
    # read arguments from command line or interface.
    import argparse
    parser = argparse.ArgumentParser(description='Citymap generator from project citygen.')

    parser.add_argument('-s', '--numSeeds', type=int, default=14, required=False,
                        help='Number of seeds used as Voronoi input (default=10)')
    parser.add_argument('-r', '--cityRadius', type=float, default=150, required=False,
                        help='Radius of the city (default=150)')
    parser.add_argument('-g', '--gateLen', type=float, default=13.08, required=False,
                        help='Size of gates in external wall. Use 0. to avoid gates. (default=13.08)')
    parser.add_argument('-n', '--cityName', default='city', required=False,
                        help='Name of the city (default="city")')
    parser.add_argument('--randomSeed', type=int, required=False,
                        help='Initial random seed value')
    parser.add_argument('-p', '--plot', required=False,
                        help='Replot a previous generated city (default="city.data.json")')
    parser.add_argument('-m', '--models', type=str, required=False, nargs='+', default=['Temple'], 
                        help='Add a list of static models defined in a .json+.blend files')
    parser.add_argument('--debug', required=False, action='store_true',
                        help='Create debug SVG files')
    parser.add_argument('--background', required=False, action='store_true')
    parser.add_argument('-P', '--python', required=False)

    # Parse arguments
    args = parser.parse_args()
    # print("args", args)
            
    if not args.plot:
        # Generate a new city map
        cityData = CityData(args)
        cityData['cityName'] = args.cityName
        # Save cityData data as a json file
        cityDataFilename = args.cityName + '.data.json'
        print("Save graph data to:", cityDataFilename)
        cityData.exportJSON(cityDataFilename)
        """
        with open(cityDataFilename, 'w') as f:
            json.dump(cityData, f, indent=4, separators=(',', ':'), sort_keys=True)
        """
    else:
        print("Read data from file:", args.plot)
        with open(args.plot, 'r') as f:
            cityData = json.load(f)
            if 'cityName' in cityData:
                print("City name: %s" % cityData['cityName'])
                args.cityName = cityData['cityName']

    # Compute matrixes used for AI path finding
    print("Computing matrixes used for AI path finding")
    AIData = newAIData(cityData['internalRegions'], cityData['vertices'])
    AIFilename = args.cityName + '.AI.json'
    print("Save AI matrixes to:", AIFilename)
    with open(AIFilename, 'w') as f:
        json.dump(AIData, f, separators=(',', ':'), sort_keys=True)

    # Plot debug info
    cityData.exportSVG(args.cityName + '.map.svg', labels=False, radius= args.cityRadius)
    cityData.exportSVG(args.cityName + '.map.verbose.svg', labels=True, radius= args.cityRadius)


# Call the main function
if __name__ == "__main__":
    main()
    print ("Ready to run: blender --background --python cityGen3D.py");

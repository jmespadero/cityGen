Procedural Generation: Where do I start?
========================================
Reddit has a channel with lots of ideas and a community that helps
* https://www.reddit.com/r/proceduralgeneration/
* https://www.reddit.com/r/proceduralgeneration/comments/a7b3vr/where_do_i_start/
A blog about drawing maps
* https://heredragonsabound.blogspot.com/


Medieval city generator 
=======================
Algorithm used in
https://www.reddit.com/r/proceduralgeneration/comments/668sqb/fantasy_medieval_cities_for_the_monthly_challenge/

1. I start with a Voronoi diagram. "Seeds" are placed spirally with random 
   fluctuations to ensure resulting regions to be relatively evenly spread.
2. The first (central region) is assigned plaza, the next N regions are city 
   wards within city walls, then comes a citadel. The rest of them are farms, 
   plains and woods.
3. To make these regions look a bit less like a Voronoi diagram and create proper
   crossroads I merge close vertices.
4. I find a border of the the city wards, this polygon will be drawn as city walls.
   I smooth it to get a rounder shape.
5. Some vertices of the city walls will be city gates. Each gate is connected to 
   the closest corner of the plaza by a wider street. Then the street is extended 
   beyond city limits.
6. The last step is creating an inner structure of each ward. In most cases I just
   cut a ward into smaller pieces with or without gaps between them (with different parameters).

Now, there is an entire channel dedicated to this project
https://www.reddit.com/r/FantasyCities/


Chaikin curves
==============

There is a lot of tutorials in the blog:
https://sighack.com/

Introduction and usages of Chaikin curves
https://sighack.com/post/chaikin-curves


ProcJam
=======

http://www.procjam.com/
PROCJAM is a fun, friendly community of people who like getting computers to make things
like art, games, toys, tools, music, stories, poetry, mistakes, languages, maps, patterns, 
recipes and more. We get together for events we call jams where we make new things, finish 
old things and share ideas, all in the same week.
They publish a annual fan-zine at http://www.procjam.com/seeds/


Implementing doors in BGE:
=========================

Door Open and Close System 
https://www.youtube.com/watch?v=Lg1s6NbBUTE

Blender Game Engine Basics Tutorial #22: Doors & Keys
https://www.youtube.com/watch?v=djx3YzGaG8E

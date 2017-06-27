
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

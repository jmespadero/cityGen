import bge
import numpy as np
import json, math, random
from math import sqrt
from mathutils import Vector

import AI_controller

g= bge.logic
co= bge.logic.getCurrentController()
Player= co.owner
scene = bge.logic.getCurrentScene()
cwd=bge.logic.expandPath("//") 
globalDict = bge.logic.globalDict

rangeControl = 4
timeToAsk = 30


def whereIgo(Player,Origin):
    regions = globalDict['regions'] 
    vertices3D = globalDict['vertices3D']
    afterDist = globalDict['afterDist']
    nextPoint = globalDict['nextPoint']
    posI=Origin
    comeBack = globalDict['comeBack']
    distReturn = Player.getDistanceTo(vertices3D[Origin])

    if Player.getDistanceTo(vertices3D[Origin])>rangeControl:
                
        if comeBack>distReturn and Player.getDistanceTo(vertices3D[nextPoint])>distReturn:
            nextPoint = Origin
            globalDict['comeBack'] = distReturn
            afterDist.clear()
        else:      
            counter = -2
            for thisreg in regions:   # go over regions
                if Origin in thisreg:   
                    counter = counter + 2
                    posList = thisreg.index(Origin)   # find index in region for Origin point
                    if posList != len(thisreg)-1:
                        next=posList+1
                    else: 
                        next = 0
                    past = posList-1
                    if len(afterDist) <= counter+1:  # add points to calculate distances
                        afterDist.append(Player.getDistanceTo(vertices3D[thisreg[next]]))
                        afterDist.append(Player.getDistanceTo(vertices3D[thisreg[past]]))
                    else:    # When have the points calculate them
                        if afterDist[counter]>Player.getDistanceTo(vertices3D[thisreg[next]]):
                            nextPoint=thisreg[next]
                            afterDist[counter]=Player.getDistanceTo(vertices3D[thisreg[next]])
                        elif afterDist[counter+1]>Player.getDistanceTo(vertices3D[thisreg[past]]):
                            nextPoint=thisreg[past]
                            afterDist[counter+1]=Player.getDistanceTo(vertices3D[thisreg[past]])
    if Player.getDistanceTo(vertices3D[nextPoint])<rangeControl:
        posI = nextPoint
        afterDist.clear()
    posF = nextPoint 
    globalDict['nextPoint'] = nextPoint                       
    globalDict['comeBack'] = distReturn
                        
    return posI,posF

def distTo(obj1,obj2):
    dist = obj2.position[0]-obj1.position[0]+obj2.position[1]-obj1.position[1]
    dist = sqrt(dist*dist)
    # print("obj dist", dist)
    return dist

def myDist(obj, node):
    # BUG ! Esto esta mal calculado!!!!
    dist = node[0]-obj[0]+node[1]-obj[1]
    dist = sqrt(dist*dist)
    return dist

def die():
    print("U Have Been Destroy!")
    return
        
def activateMonsters(numMonsters):
    myscene=bge.logic.getSceneList()
    if numMonsters !=4:
        for m in range(0, numMonsters):
            nameMonst = 'MonsterToken ' + str(m)
            for obj in myscene[0].objects:
                if obj.name == nameMonst:
                    mostPostLabel = 'MonsterTokenPos'+str(m)
                    lastLabel = mostPostLabel+str("last") 
                    vertices=globalDict['vertices3D']
                    if (mostPostLabel not in globalDict): # Where is the monster
                        farfaraway=99999                
                        for k in range(len(vertices)):
                            dist = (vertices[k][0]-obj.position[0]+vertices[k][1]-obj.position[1])
                            dist = sqrt(dist*dist)                 
                            if dist < farfaraway:
                               point=k
                               farfaraway = dist
                        globalDict[mostPostLabel]=point   
                        globalDict[lastLabel] = 99999 
                              
                    else:    
                        
                        point = globalDict[mostPostLabel]
                        decisionMatrix = globalDict['decisionMatrix']
                        goto = decisionMatrix[point][globalDict['positionI']]
                        pos = vertices[int(goto)]  
                        distTo1 = myDist(obj.position,pos)
                        distTo2 = myDist(obj.position,vertices[int(point)])
                        if (distTo1 < 0.5 or distTo2 < 0.5):
                            # print(nameMonst,"I stay in", point," I go to: ", goto)
                            obj.position[0]= pos[0]
                            obj.position[1]= pos[1]
                            if globalDict[lastLabel] == 99999:
                                globalDict[lastLabel] = goto
                            else:
                                name = 'Monster ' + str(m)
                                dist = distTo(obj, myscene[0].objects[name])
                                if dist < 0.5:
                                    globalDict[mostPostLabel] = goto
                                    globalDict[lastLabel] = goto
                                    # print("reload position")

                        # obj.position[vertices[int(globalDict[mostPostLabel])]]   
                        # pos = (globalDict['vertices'][globalDict['positionI']])
                        # obj.position[0]= pos[0]
                        # obj.position[1]= pos[1]
    else:
        #for m in range(0,numMonsters):
        #    obj = myscene[0].objects
        #    if obj['MonsterToken 1']:
        #        print("hola ",)
        #
        
          
        obj = myscene[0].objects
        vertices=globalDict['vertices3D']
        for m in range(0,numMonsters):
            nameM = 'MonsterToken ' +str(m)
            mon = obj[nameM]
            mostPostLabel = 'MonsterTokenPos'+str(m)
            lastLabel = mostPostLabel+str("last") 
            if (mostPostLabel not in globalDict):
                farfaraway=99999                
                for k in range(len(vertices)):
                    dist = (vertices[k][0]-mon.position[0]+vertices[k][1]-mon.position[1])
                    dist = sqrt(dist*dist)                 
                    if dist < farfaraway:
                       point=k
                       farfaraway = dist
                globalDict[mostPostLabel]=point   
                globalDict[lastLabel] = 99999 
                
                
        # Monster 0 Direct:
        decisionMatrix = globalDict['decisionMatrix']
        mon0=obj['MonsterToken 0']
        realMon0 = obj['Monster 0']
        point = int(globalDict['MonsterTokenPos0'])
        shortestMatrix = globalDict['shortestPathMatrix']
        if 'Monster0from' not in globalDict:
            globalDict['Monster0from'] = point
        fromto = int(globalDict['Monster0from'])
        #print("from",fromto,vertices[int(fromto)])
        #print("point",point,vertices[int(point)]) 
        #print("tokpos", mon0.position)
        #print("monpos", realMon0.position)
        pos = vertices[int(fromto)]  
        distTo1 = myDist(realMon0.position,pos) # distance Monster to vertice Dest
        pos = vertices[int(point)]
        distTo2 = myDist(realMon0.position,pos) # distance Monster to vertice Orig
        myPlayer = obj['Player']
        positionI=int(globalDict['positionI'])
        positionF=int(globalDict['positionF'])
        distPlayerToF = myDist(myPlayer.position,vertices[positionF]) # distance Player to vertice Dest
        distPlayerToI = myDist(myPlayer.position,vertices[positionI]) # distance Player to vertice Orig
        dist1F=distTo1 + shortestMatrix[fromto][positionF] + distPlayerToF
        dist1I=distTo1 + shortestMatrix[fromto][positionI] + distPlayerToI
        dist2F=distTo2 + shortestMatrix[point][positionF] + distPlayerToF
        dist2I=distTo2 + shortestMatrix[point][positionI] + distPlayerToI
        list = [dist1F,dist1I,dist2F,dist2I]
        dist = min(list)
        """
        if  dist1F == dist1I and dist2F == dist2I:
            print("Misma distancia")
        else:
            print("1f", "%.2f" % dist1F, "1I", "%.2f" % dist1I)
            print("2f", "%.2f" % dist2F, "2i", "%.2f" % dist2I)
            print("%.2f" % dist)    
        """
        if  dist1F == dist1I and dist2F == dist2I:
            if distTo(mon0,realMon0)<0.3:
                playerPosF = globalDict['positionF']
                if distTo2<distTo1:
                    globalDict['Monster0from'] = point
                    fromto = decisionMatrix[point][playerPosF]
                    globalDict['MonsterTokenPos0'] = fromto
                else:
                    globalDict['Monster0from'] = fromto
                    fromto = decisionMatrix[fromto][playerPosF] 
                    globalDict['MonsterTokenPos0'] = fromto
                pos = vertices[int(fromto)]
                mon0.position[0]= pos[0]
                mon0.position[1]= pos[1]
        else:
            if distTo(mon0,realMon0)>0.3:
                if dist == dist1I or dist==dist1F:
                    pos = vertices[int(fromto)]
                    mon0.position[0]= pos[0]
                    mon0.position[1]= pos[1]
                if dist == dist2I or dist==dist2F:
                    pos = vertices[int(point)]
                    mon0.position[0]= pos[0]
                    mon0.position[1]= pos[1]
            else:
                if dist == dist2F or dist==dist1F:
                    playerPosF = globalDict['positionF']
                    if distTo2<distTo1:
                        globalDict['Monster0from'] = point
                        fromto = decisionMatrix[point][playerPosF]
                        globalDict['MonsterTokenPos0'] = fromto
                    else:
                        globalDict['Monster0from'] = fromto
                        fromto = decisionMatrix[fromto][playerPosF] 
                        globalDict['MonsterTokenPos0'] = fromto
                if dist == dist2F or dist==dist1F:
                    playerPosI = globalDict['positionI']     
                    if distTo2<distTo1:
                        globalDict['Monster0from'] = point
                        fromto = decisionMatrix[point][playerPosF]
                        globalDict['MonsterTokenPos0'] = fromto
                    else:
                        globalDict['Monster0from'] = fromto
                        fromto = decisionMatrix[fromto][playerPosF] 
                        globalDict['MonsterTokenPos0'] = fromto 
                        
        #goto = globalDict['Monster0goto']
        #point = globalDict['MonsterTokenPos0']
        """
        if dist == dist1F or dist == dist1I and distTo1 > 0.2:
            if dist == dist1F:
                pos = vertices[int(goto)]
            else:
                pos = vertices[int(point)]
            globalDict['MonsterTokenPos0'] = point
            mon0.position[0]= pos[0]
            mon0.position[1]= pos[1]
            globalDict['Monster0goto'] = goto
        elif dist == dist1F:
            globalDict['MonsterTokenPos0'] = point
            goto = decisionMatrix[goto][globalDict['positionF']]
            pos = vertices[int(goto)]
            mon0.position[0]= pos[0]
            mon0.position[1]= pos[1]
            globalDict['Monster0goto'] = goto
        elif dist == dist1I:
            globalDict['MonsterTokenPos0'] = point
            goto = decisionMatrix[point][globalDict['positionI']]
            pos = vertices[int(goto)]
            mon0.position[0]= pos[0]
            mon0.position[1]= pos[1]
            globalDict['Monster0goto'] = goto
        
        if dist == dist2F or dist == dist2I and distTo2 > 0.2:
            if dist == dist2F:
                pos = vertices[int(goto)]
            else:
                pos = vertices[int(point)]
            globalDict['MonsterTokenPos0'] = point    
            mon0.position[0]= pos[0]
            mon0.position[1]= pos[1]
            globalDict['Monster0goto'] = goto
        elif dist == dist2F:
            globalDict['MonsterTokenPos0'] = point
            goto = decisionMatrix[goto][globalDict['positionF']]
            pos = vertices[int(goto)]
            mon0.position[0]= pos[0]
            mon0.position[1]= pos[1]
            globalDict['Monster0goto'] = goto
        elif dist == dist2I:
            globalDict['MonsterTokenPos0'] = point
            goto = decisionMatrix[point][globalDict['positionI']]
            pos = vertices[int(goto)]
            mon0.position[0]= pos[0]
            mon0.position[1]= pos[1]
            globalDict['Monster0goto'] = goto
        if  dist1F == dist1I and dist2F == dist2I:
            if distTo1<0.5:
                globalDict['MonsterTokenPos0'] = point
                goto = decisionMatrix[goto][globalDict['positionF']]
                pos = vertices[int(goto)]
                mon0.position[0]= pos[0]
                mon0.position[1]= pos[1]
                globalDict['Monster0goto'] = goto
            if distTo2<0.5:
                globalDict['MonsterTokenPos0'] = point
                goto = decisionMatrix[point][globalDict['positionF']]
                pos = vertices[int(goto)]
                mon0.position[0]= pos[0]
                mon0.position[1]= pos[1]
                globalDict['Monster0goto'] = goto
        
        print("go 0 to:", globalDict['Monster0goto'], "stay", globalDict['MonsterTokenPos0'])
        
        """
        """
        point = globalDict['MonsterTokenPos0']
        decisionMatrix = globalDict['decisionMatrix']
        goto = decisionMatrix[point][globalDict['positionI']]
        print("I'm, M0 and i go to", goto, "positionI ", globalDict['positionI'])
        pos = vertices[int(goto)]  
        distTo1 = myDist(mon0.position,pos)
        distTo2 = myDist(mon0.position,vertices[int(point)])
        if (distTo1 < 0.5 or distTo2 < 0.5):
            #print(nameMonst,"I stay in", point," I go to: ", goto)
            mon0.position[0]= pos[0]
            mon0.position[1]= pos[1]
            if globalDict['MonsterTokenPos0last'] == 99999:
                globalDict['MonsterTokenPos0last'] = goto
            else:
                name = 'Monster 0'
                dist = distTo(mon0, myscene[0].objects[name])
                
                if dist < 0.5:
                    globalDict['MonsterTokenPos0'] = goto
                    globalDict['MonsterTokenPos0last'] = goto
        """          
        # Check if any monster is near of the player
        for j in range(0,numMonsters):
            nameM = 'Monster ' +str(j)
            mon = obj[nameM]
            distToPlayer = myDist(myPlayer.position,mon.position)
            nameToken = 'MonsterToken ' + str(j)
            monToken = obj[nameToken]
            if distToPlayer < 5:
                monToken.position = myPlayer.position
            if distToPlayer < 0.5:
                die()
        
        # Jump Monster to Monster and Stay Together
        for j in range(0,numMonsters):
            nameM1 = 'Monster ' +str(j)
            mon1 = obj[nameM1]
            for i in range(0,numMonsters):
                nameM2 = 'Monster ' +str(i)
                mon2 = obj[nameM2]
                if nameM1 != nameM2:
                    dist = distTo(mon1,mon2)
                    if dist<0.5 and mon2.position[2]<1:
                            globalDict['Jump'] = nameM1
                            mon1.position[2]=3
        for j in range(0,numMonsters):
            nameM1 = 'MonsterToken ' +str(j)
            mon1 = obj[nameM1]
            for i in range(0,numMonsters):
                nameM2 = 'MonsterToken ' +str(i)
                mon2 = obj[nameM2]
                if mon1!=mon2:
                    dist = distTo(mon1,mon2)
                    if dist == 0:
                        sep = 0.5
                        mon1.position[0] = mon1.position[0] + sep
                        mon1.position[1] = mon1.position[1] + sep
                        mon2.position[0] = mon2.position[0] - sep
                        mon2.position[1] = mon2.position[1] - sep

                   
        # Monster 1 Forward:
        mon1=obj['MonsterToken 1']
        realMon1 = obj['Monster 1']
        if distTo(mon1,realMon1)<0.5:
            playerPosF = int(globalDict['positionF'])
            point = int(globalDict['MonsterTokenPos1'])
            decisionMatrix = globalDict['decisionMatrix']
            goto = decisionMatrix[point][playerPosF]
            pos = vertices[int(goto)]
            mon1.position[0]= pos[0]
            mon1.position[1]= pos[1]
            globalDict['MonsterTokenPos1'] = goto
            #print("go 1", goto)
        
        # Monster 2 Back:
        mon2=obj['MonsterToken 2']
        realMon2 = obj['Monster 2']
        if distTo(mon2,realMon2)<0.5:
            playerPosF = int(globalDict['positionI'])
            point = int(globalDict['MonsterTokenPos2'])
            decisionMatrix = globalDict['decisionMatrix']
            goto = decisionMatrix[point][playerPosF]
            pos = vertices[int(goto)]
            mon2.position[0]= pos[0]
            mon2.position[1]= pos[1]
            globalDict['MonsterTokenPos2'] = goto
            #print("go 2",goto)
        
        # Monster 3 Random:
        mon3=obj['MonsterToken 3']
        realMon3 = obj['Monster 3']
        dist = distTo(mon3,realMon3)
        if dist < 0.5:
            decisionMatrix = globalDict['decisionMatrix']
            point = int(globalDict['MonsterTokenPos3'])
            # Choose a random vertex from the list of internalPoints
            myRand = random.choice(globalDict['internalPoints'])
            goto = decisionMatrix[point][myRand]
            #print("I'm, M3 and i go to", goto)
            pos = vertices[int(goto)] 
            mon3.position[0]= pos[0]
            mon3.position[1]= pos[1]
            globalDict['MonsterTokenPos3']=goto
    return
            
#def activateMonsters(numMonsters):
#    for w in range(0, numMonsters):
#        coMonst = globalDict['Controler Monster 0']
#        actMonstMot = coMonst.actuators["MonsMot"]
#        actMonstMot.dLoc=(0.0,+0.05,0.0)
#        co.activate("MonsMot")
#    return

##Main

if 'playerInit' not in globalDict or not globalDict['playerInit']:
    print("Force call to AI_controller.init()")
    AI_controller.init()
    
positionI = globalDict['positionI'] 
positionF = globalDict['positionF']

#position = ob.game.properties[0].value
timeToWhereIGo =  globalDict['timeToWhereIGo']
timeGame = globalDict['timeGame']
myPosition = Player.position
if ('playerPosition' in globalDict) : 
    globalDict['playerPosition'] = myPosition

#print("mi posicion" , myPosition)


# Sensors

keysensor = co.sensors["Keyboard"]

# Actuators
actmot = co.actuators["Move"]
actrot = co.actuators["Rotation"]
actani = co.actuators["Animate"]
actjmp = co.actuators["Jump"]


Akey = keysensor.getKeyStatus(97)
Ckey = keysensor.getKeyStatus(99)
Dkey = keysensor.getKeyStatus(100)
Mkey = keysensor.getKeyStatus(109)
Rkey = keysensor.getKeyStatus(114)
Skey = keysensor.getKeyStatus(115)
Wkey = keysensor.getKeyStatus(119)

Spacekey = keysensor.getKeyStatus(32)


#print("A, ", Akey)
#print("S, ", Skey)
#print("D, ", Dkey)
#print("W, ", Wkey)


# Return to Center or Random Point
if Ckey == 1:
    print("Player traslated to position (0,0,0)")    
    myPlayer=scene.objects['Player']
    myPlayer.position=(0.0, 0.0, 1.0)

if Rkey == 1:
    # Choose a random element from globalDict['internalPoints']
    newVertex = random.choice(globalDict['internalPoints'])
    pos = globalDict['vertices3D'][newVertex]
    print("Player traslated to vertex", newVertex, "position", pos)    
    myPlayer=scene.objects['Player']
    myPlayer.position=(pos[0], pos[1], 1.0)

# MiniMap
if Mkey == (1 or 2 or 3):
    # Change status of minimap
    globalDict['MiniMapOn'] = not globalDict['MiniMapOn']
    bge.logic.getCurrentScene().objects['MiniMap'].useViewport = globalDict['MiniMapOn']
        
# Jump

#print (Spacekey)

if Spacekey == 1:
    #actmot.force = (0.0, 0.0, 400.0)
    a = actjmp.dLoc[2] + 0.15
    actjmp.dLoc = (0.0, 0.0, a)
    co.activate("Jump")    

if Spacekey == 2: 
    #a = actmot.force[2] - 2.5
    #actmot.force = (0.0, 0.0, a)
    a = actjmp.dLoc[2] - 0.005
    actjmp.dLoc = (0.0, 0.0, a)
    if a<0:
        actjmp.dLoc = (0.0, 0.0, 0.0)
    co.activate("Jump")

if Spacekey == 3: 
    while actjmp.dLoc[2] > 0:
        a = actjmp.dLoc[2] - 0.09
        actjmp.dLoc = (0.0, 0.0, a)
        co.activate("Jump")
    actjmp.dLoc = (0.0, 0.0, 0.0)
    co.activate("Jump")
#Forward
speed=0.1
if Wkey == 1:
    actani.action = "run"
    actani.frameStart = 1.0
    actani.frameEnd = 20.0
    actani.blendIn = 10
    actani.mode = 4
    actani.priority = 0
    co.activate("Animate")
    
    actmot.dLoc=(0.0,+speed,0.0)
    co.activate("Move")
    
    timeToWhereIGo = timeToWhereIGo +1
    if timeToWhereIGo==timeToAsk:
        timeToWhereIGo=0
        positionI, positionF = whereIgo(Player,positionI)
        #print("I go to: ", positionF, " I stay in: ", positionI)
        #print("I go to: ",positionF)
        #print("Stay in: ",positionI)
        globalDict['positionI'] = positionI
        globalDict['positionF'] = positionF
#Back
if Skey == 1:
    actani.action = "run"
    actani.frameStart = 20.0
    actani.frameEnd = 1.0
    actani.blendIn = 10
    actani.mode = 4
    actani.priority = 0
    co.activate("Animate")
    
    actmot.dLoc=(0.0,-speed,0.0)
    co.activate("Move")
    
    timeToWhereIGo = timeToWhereIGo +1
    if timeToWhereIGo==timeToAsk:
        timeToWhereIGo=0
        positionI, positionF = whereIgo(Player,positionI)
        print("I go to: ",positionF)
        print("Stay in: ",positionI)
        globalDict['positionI'] = positionI
        globalDict['positionF'] = positionF

# Calculate Direction
if Skey == 2 or Wkey ==2:
    timeToWhereIGo = timeToWhereIGo +1
    if timeToWhereIGo==timeToAsk:
        timeToWhereIGo=0
        activateMonsters(globalDict['numMonster'])
        positionI, positionF = whereIgo(Player,positionI)
        #print("I go to: ",positionF)
        #print("Stay in: ",positionI)
        globalDict['positionI'] = positionI
        globalDict['positionF'] = positionF
        #ob.game.properties['myPlayerOrigin'].value = positionI
        #ob.game.properties['myPlayerDestiny'].value = positionF

##Time save

timeGame = timeGame+1
if timeGame == timeToAsk:
    timeGame = 0
    activateMonsters(globalDict['numMonster'])

globalDict['timeToWhereIGo'] = timeToWhereIGo
globalDict['timeGame'] = timeGame

# NoMov
if Skey == 3 and Wkey == 3:
    actmot.dLoc=(0.0,0.0,0.0)
    co.activate("Move")
    
    actani.action = "idle"
    actani.frameStart = 1.0
    actani.frameEnd = 30.0
    actani.blendIn = 10
    actani.mode = 4
    actani.priority = 0
    co.activate("Animate")
    
if Skey == 0 and Wkey == 3:
    actmot.dLoc=(0.0,0.0,0.0)
    co.activate("Move")
    
    actani.action = "idle"
    actani.frameStart = 1.0
    actani.frameEnd = 30.0
    actani.blendIn = 10
    actani.mode = 4
    actani.priority = 0
    co.activate("Animate")
    
if Skey == 3 and Wkey == 0:
    actmot.dLoc=(0.0,0.0,0.0)
    co.activate("Move")
    
    actani.action = "idle"
    actani.frameStart = 1.0
    actani.frameEnd = 30.0
    actani.blendIn = 5
    actani.mode = 4
    actani.priority = 0
    co.activate("Animate")


# Rotation

# RotLeft
rotValue=0.05
if Akey == 1:
    actrot.dRot=(0.0,0.0,rotValue)
    co.activate("Rotation")
    
# RotRight
if Dkey == 1:
    actrot.dRot=(0.0,0.0,-rotValue)
    co.activate("Rotation")
    
# NotRot
if Dkey==3 and Akey==3:
    actrot.dRot=(0.0,0.0,0.0)
    co.activate("Rotation")
    
if Dkey==3 and Akey==0:
    actrot.dRot=(0.0,0.0,0.0)
    co.activate("Rotation")
    
if Dkey==0 and Akey==3:
    actrot.dRot=(0.0,0.0,0.0)
    co.activate("Rotation")

if globalDict['MiniMapOn']:
    # move MiniMap camera to position player
    camMiniMap=scene.objects['MiniMap']
    myPlayer=scene.objects['Player']
    camMiniMap.position[0] = myPlayer.position[0]
    camMiniMap.position[1] = myPlayer.position[1]


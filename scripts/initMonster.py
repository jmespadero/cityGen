import bge
globalDict = bge.logic.globalDict
co=bge.logic.getCurrentController()
sensorAlways=co.sensors

#!!!This script is associated with the token, not the monster....
#print('monsterName:', own['monsterName'])
    
if ('numMonster' in globalDict):
    imMonster= 'Controler Monster '+ str(globalDict['numMonster'])
    globalDict['numMonster'] += 1
    #This produce the Error, bge.logic.globalDict could not be marshal'd
    #See http://blenderartists.org/forum/showthread.php?328958-Problem-with-using-globalDict-and-stand-alone-player&p=2589721&viewfull=1#post2589721
    #globalDict[imMonster] = co
    globalDict[imMonster] = True
    sensorAlways[0].usePosPulseMode = 0
    print(imMonster)
else:
    sensorAlways[0].usePosPulseMode = 1
    

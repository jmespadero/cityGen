
# cityGen
## An automatic generator for medieval scenes and games

![](demos/cityGameDemo.jpg)
<https://github.com/jmespadero/cityGen>

## Description
CityGen is a set of tools to automatically generate random 2D 
maps of medieval cities, which are then converted into 3D scenes
and videogames.

The scripts are mainly written in [python3](https://www.python.org), 
models are generated with [blender3D](https://www.blender.org) and
games are playable using blenderplayer.

Most of the artwork has been done by external artists that have released
their work in several domain licenses. Please refer to the Readme.txt file
bundled inside of each .blend file to get detailed info about their authors 
and licenses applyed to their work. If you feel that you are owner of any 
artwork used here, send me a message.

Special acknowledge to (Daniel Andersson)[http://www.blendswap.com/user/Daniel74]
and (Dennis Haupt)[http://traevaine.com/] artworks.

## Developers
CityGen development started in 2014 as the CS final project of Juan Ramos. 
Its work is currently continued by Sergio Fernandez under the supervision
of Jose M. Espadero.

## How do I create a model?
``` sh
    git clone https://github.com/jmespadero/cityGen.git
    cd cityGen
    python3 cityGen2D.py && blender --background --python cityGen3D.py
```

Some nice values for randomSeed:
* ./cityGen2D.py  --randomSeed 10
* ./cityGen2D.py -s 30 --randomSeed 47932


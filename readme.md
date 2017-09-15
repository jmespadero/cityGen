
# Medieval CityGen
## An automatic generator for medieval scenes and games

![](demos/cityGameDemo.jpg)
<https://github.com/jmespadero/cityGen>

## Description
Medieval CityGen is a set of tools to automatically generate random 2D 
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

The project give a special acknowledge to [Daniel Andersson](http://www.blendswap.com/user/Daniel74)
and [Dennis Haupt](http://traevaine.com/) 3D artworks.

## Developers
Medieval CityGen development started in 2014 as the CS final project of Juan Ramos. 
Its work is currently continued by Sergio Fernandez under the supervision
of Jose M. Espadero.

## How do I create a model?
First step is (obviously) to install blender3D, and then clone this repository:
``` sh
    git clone https://github.com/jmespadero/cityGen.git
    cd cityGen
```

To create a new 2D map and 3D model from scratch, execute the command:
``` sh
    blender --background --python cityGen2D.py --python cityGen3D.py
```

As alternative, you can run the 2D script and 3D script separated, so you can 
change the configuration file and create several 3D models from the same 2 map.
``` sh
    python3 cityGen2D.py 
    blender --background --python cityGen3D.py
```

You can use some parameter to control values as city size and initial random seed
(so you create the same 2D map several times)
* python3 cityGen2D.py  --randomSeed 10
* python3 cityGen2D.py -s 30 --randomSeed 47932




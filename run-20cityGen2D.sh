#!/bin/bash
#
for i in `seq -f %04g 20`
do
    echo $i
    #Run cityGen2D and keep the output and city.map.verbose.svg
    python3 cityGen2D.py -s 24 -m Temple Market Temple > test.$i.txt
    mv city.map.verbose.svg test.$i.svg
done


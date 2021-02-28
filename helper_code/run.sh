#!/bin/bash

for i in `seq  -f %06g 150 193`
do
   echo localdeform$i.xyz
   ./2d_phase ../Ringanalysis/localdeform$i.xyz
   mv output.xyz output/$i.xyz
done

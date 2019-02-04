#!/bin/bash
for (( c=1; c<=140; c++ ))
do
    nsml model rm team_17/ir_ph2/$c "*"
done

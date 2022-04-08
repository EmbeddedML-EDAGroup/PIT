#!/bin/bash

#strength=( 1.0e-08 2.5e-08 5.0e-08 7.5e-08 1.0e-07 2.5e-07 5.0e-07 1.0e-06 )
strength=( 1e-10 1.0e-09 1.0e-08 1.0e-07 1.0e-06 1.0e-05 1.0e-04 )
#strength=( 1.0e-9 2.5e-8 7.5e-08 1.0e-06 5.0e-6 1.0e-5 )
#strength=( 1.0e-4 1.0e-3 )

for i in "${strength[@]}"
do
    echo "Strength: $i"
    source fbnet_dalia.sh $i
done
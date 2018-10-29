#!/bin/bash

if [ -d output ]; then
    mv output output_$(date -d "today" +"%Y%m%d%H%M%S")
    mkdir output
fi

if [ ! -d output ]; then
    mkdir output
fi

./create_results.sh 2> /dev/null

exit 0

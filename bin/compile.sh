#!/bin/bash

nvcc -m64 -arch=compute_61 -use_fast_math src/common/lodepng.o src/main.cu -o main -lGL -lglut -lm -lpng -g -G

exit 0

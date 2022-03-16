#!/bin/bash

nvcc -m64 -arch=native -use_fast_math `pkg-config --cflags glfw3`\
  common/glad.o src/main.cu -o main `pkg-config --static --libs glfw3`\
  -lm -lturbojpeg -llz4 -Xcompiler -fopenmp -lgomp #-g -G

exit 0
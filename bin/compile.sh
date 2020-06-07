#!/bin/bash

nvcc -m64 -arch=compute_61 -rdc=true -use_fast_math `pkg-config --cflags glfw3`\
  common/glad.o src/main.cu -o main `pkg-config --static --libs glfw3`\
  -lm -lturbojpeg -Xcompiler -fopenmp -lgomp

exit 0
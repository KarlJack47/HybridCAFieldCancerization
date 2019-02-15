#!/bin/bash

nvcc -m64 -arch=compute_61 -use_fast_math `pkg-config --cflags glfw3` src/common/glad.o src/common/lodepng.o src/main.cu -o main `pkg-config --static --libs glfw3` -lm -lpng
exit 0

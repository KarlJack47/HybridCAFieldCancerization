#ifndef __GPU_ANIM_H__
#define __GPU_ANIM_H__

#include "general.h"

#include "glad/glad.h"
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>
#include <iostream>
#include <unistd.h>

static void error_callback(int error, const char* description) {
	fprintf(stderr, "Error: %s\n", description);
}

struct GPUAnimBitmap {
    GLFWwindow *windows[2];
    GLuint bufferObj;
    cudaGraphicsResource *resource;
    int width, height;
    void *dataBlock;
    void (*fAnimCA)(uchar4*,void*,int);
    void (*fAnimCarcin)(uchar4*,void*,int,int);
    void (*fAnimTimer)(void*,bool,int);
    int dragStartX, dragStartY;
    int display;
    int n_carcin;
    int current_carcin;

    GPUAnimBitmap(int w, int h, void *d=NULL, int show=1, int n_car=1) {
        width = w;
        height = h;
        dataBlock = d;
	display = show;
	n_carcin = n_car;

	glfwSetErrorCallback(error_callback);

	if (!glfwInit())
		exit(EXIT_FAILURE);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

	windows[0] = glfwCreateWindow(width, height, "CA", NULL, NULL);
	if (!windows[0]) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwSetKeyCallback(windows[0], key_callback);

	glfwMakeContextCurrent(windows[0]);

	gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);

	glfwSwapInterval(1);

	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*4, NULL, GL_DYNAMIC_DRAW_ARB);
	cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);
    }

    ~GPUAnimBitmap(void) {
        free_resources();
    }

    static GPUAnimBitmap** get_bitmap_ptr(void) {
	static GPUAnimBitmap *gBitmap;
	return &gBitmap;
    }

    void hide_window(GLFWwindow *window) {
	glfwHideWindow(window);
    }

    void free_resources(void) {
        cudaGraphicsUnregisterResource(resource);

	for (int i = 0; i < 2; i++) {
		glfwMakeContextCurrent(windows[i]);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	}
	glDeleteBuffers(1, &bufferObj);

	glfwTerminate();
    }

    long image_size(void) const { return width * height * 4; }

    void anim(void(*fCA)(uchar4*,void*,int), void(*fCarcin)(uchar4*,void*,int,int), void(*fTime)(void*,bool,int), char **carcin_names) {
	GPUAnimBitmap **bitmap = get_bitmap_ptr();
	*bitmap = this;

        fAnimCA = fCA;
	fAnimCarcin = fCarcin;
	fAnimTimer = fTime;

	windows[1] = glfwCreateWindow(width, height, carcin_names[0], NULL, windows[0]);
	if (!windows[1]) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwMakeContextCurrent(windows[1]);
	int xpos, ypos, left, right, w;
	glfwGetWindowSize(windows[0], &w, NULL);
	glfwGetWindowFrameSize(windows[0], &left, NULL, &right, NULL);
	glfwGetWindowPos(windows[0], &xpos, &ypos);
	glfwSetWindowPos(windows[1], xpos+w+left+right, ypos);
	glfwSetKeyCallback(windows[1], key_callback);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*4, NULL, GL_DYNAMIC_DRAW_ARB);

	if (display == 0) hide_window(windows[1]);

	int ticks = 0;
	current_carcin = 0;
	while (!glfwWindowShouldClose(windows[0]) && !glfwWindowShouldClose(windows[1])) {
        	uchar4* devPtr;
        	size_t size;

		fAnimTimer(dataBlock, true, ticks);

		cudaGraphicsMapResources(1, &(resource), NULL);
        	cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource);

		fAnimCA(devPtr, dataBlock, ticks);
		if (display == 1)
			draw(windows[0], width, height);

		cudaGraphicsUnmapResources(1, &(resource), NULL);

		for (int i = 0; i < n_carcin; i++) {
			cudaGraphicsMapResources(1, &(resource), NULL);
        		cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource);

			fAnimCarcin(devPtr, dataBlock, i, ticks);
			if (display == 1 && i == current_carcin) {
				glfwSetWindowTitle(windows[1], carcin_names[current_carcin]);
				draw(windows[1], width, height);
			}

			cudaGraphicsUnmapResources(1, &(resource), NULL);
		}

		fAnimTimer(dataBlock, false, ticks);

		ticks++;

		glfwWaitEvents();
	}
    }

    // static method used for callbacks
    static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
	GPUAnimBitmap *bitmap = *(get_bitmap_ptr());
        switch (key) {
            case GLFW_KEY_ESCAPE:
		if (action == GLFW_PRESS) {
			glfwSetWindowShouldClose(window, GLFW_TRUE);
			exit(EXIT_SUCCESS);
		}
	    case GLFW_KEY_RIGHT:
		if (action == GLFW_PRESS) {
			if (bitmap->current_carcin != bitmap->n_carcin-1) bitmap->current_carcin++;
			else bitmap->current_carcin = 0;
		}
            case GLFW_KEY_LEFT:
		if (action == GLFW_PRESS) {
			if (bitmap->current_carcin != 0) bitmap->current_carcin--;
			else bitmap->current_carcin = bitmap->n_carcin-1;
		}
	}
    }

    void draw(GLFWwindow *window, int width, int height) {
	glfwMakeContextCurrent(window);
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glDrawPixels(width, height, GL_RGBA,
		     GL_UNSIGNED_BYTE, 0);
	glfwSwapBuffers(window);
    }
};

#endif  // __GPU_ANIM_H__

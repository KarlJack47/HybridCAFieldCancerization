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
	GLFWwindow *windows[3];
	GLuint bufferObjs[3];
	cudaGraphicsResource *resources[3];
	uchar4 *devPtrs[3]; size_t sizes[3];
	int width, height;
	void *dataBlock;
	void (*fAnimCA)(uchar4*,void*,int);
	void (*fAnimCarcin)(uchar4*,void*,int,int);
	void (*fAnimCell)(uchar4*,void*,int,int);
	void (*fAnimTimer)(void*,bool,int);
	int dragStartX, dragStartY;
	int display;
	int grid_size;
	int maxT;
	int ticks;
	char **carcin_names;
	int current_carcin;
	int current_cell[2];
	bool paused;
	bool excise;

	GPUAnimBitmap(int w, int h, void *d=NULL, int show=1, int g_size=512, int T=600, char **car_names=NULL) {
		width = w;
		height = h;
		dataBlock = d;
		display = show;
		grid_size = g_size;
		paused = false;
		if (display == 1) paused = true;
		excise = false;
		maxT = T;
		ticks = 0;

		if (car_names != NULL) {
			carcin_names = (char**)malloc(NUM_CARCIN*sizeof(char*));
			for (int i = 0; i < NUM_CARCIN; i++) {
				carcin_names[i] = (char*)malloc((strlen(car_names[i])+1)*sizeof(char));
				strcpy(carcin_names[i], car_names[i]);
			}
		}

		glfwSetErrorCallback(error_callback);

		if (!glfwInit())
			exit(EXIT_FAILURE);

		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		char window_name[12] = { '\0' };
		strcpy(window_name, "CA");
		create_window(0, width, height, window_name, &key_ca);
		strcpy(window_name, "Cell (0, 0)");
		create_window(1, width, height, window_name, &key_cell);
		if (car_names != NULL)
			create_window(2, width, height, carcin_names[0], &key_carcin);
	}

	~GPUAnimBitmap(void) { free_resources(); }

	static GPUAnimBitmap** get_bitmap_ptr(void) {
		static GPUAnimBitmap *gBitmap;
		return &gBitmap;
    	}

	void hide_window(GLFWwindow *window) { glfwHideWindow(window); }
	void show_window(GLFWwindow *window) { glfwShowWindow(window); }

	void free_resources(void) {
		if (carcin_names != NULL) {
			for (int i = 0; i < NUM_CARCIN; i++) free(carcin_names[i]);
			free(carcin_names);
		}

		for (int i = 0; i < 3; i++) {
			CudaSafeCall(cudaGraphicsUnregisterResource(resources[i]));
			glfwMakeContextCurrent(windows[i]);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		}
		glDeleteBuffers(3, bufferObjs);

		glfwTerminate();
    	}

	long image_size(void) const { return width * height * sizeof(uchar4); }

	void create_window(int window_idx, int w, int h, char *name,
			   void(*key)(GLFWwindow *,int,int,int,int)) {
		windows[window_idx] = glfwCreateWindow(w, h, name, NULL, NULL);
		if (!windows[window_idx]) {
			glfwTerminate();
			exit(EXIT_FAILURE);
		}

		glfwMakeContextCurrent(windows[window_idx]);

		if (window_idx != 0) {
			int xpos, ypos, left, right, wid;
			glfwGetWindowSize(windows[window_idx-1], &wid, NULL);
			glfwGetWindowFrameSize(windows[window_idx-1], &left, NULL, &right, NULL);
			glfwGetWindowPos(windows[window_idx-1], &xpos, &ypos);
			glfwSetWindowPos(windows[window_idx], xpos+wid+left+right, ypos);
		}

		glfwSetKeyCallback(windows[window_idx], key);

		if (window_idx == 0)
			gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);

		glGenBuffers(1, &bufferObjs[window_idx]);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObjs[window_idx]);
		glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, image_size(), NULL, GL_DYNAMIC_DRAW_ARB);
		CudaSafeCall(cudaGraphicsGLRegisterBuffer(&resources[window_idx], bufferObjs[window_idx], cudaGraphicsRegisterFlagsNone));

		glfwSwapInterval(1);
	}

	void update_window(int window_idx, int ticks, int idx=0, int current=0) {
		CudaSafeCall(cudaGraphicsMapResources(1, &resources[window_idx], NULL));
		CudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&devPtrs[window_idx], &sizes[window_idx], resources[window_idx]));

		if (window_idx == 0) fAnimCA(devPtrs[window_idx], dataBlock, ticks);
		else if (window_idx == 1) fAnimCell(devPtrs[window_idx], dataBlock, idx, ticks);
		else if (window_idx == 2) fAnimCarcin(devPtrs[window_idx], dataBlock, idx, ticks);

		CudaSafeCall(cudaGraphicsUnmapResources(1, &resources[window_idx], NULL));

		if (display == 1 && idx == current) draw(window_idx, width, height);
	}

	void anim(void(*fCA)(uchar4*,void*,int), void(*fCarcin)(uchar4*,void*,int,int),
		  void(*fCell)(uchar4*,void*,int,int), void(*fTime)(void*,bool,int)) {

		GPUAnimBitmap **bitmap = get_bitmap_ptr();
		*bitmap = this;

		fAnimCA = fCA;
		fAnimCarcin = fCarcin;
		fAnimCell = fCell;
		fAnimTimer = fTime;

		current_carcin = 0;
		current_cell[0] = 0;
		current_cell[1] = 0;
		while (!glfwWindowShouldClose(windows[0]) &&
		       !glfwWindowShouldClose(windows[1]) &&
		       !glfwWindowShouldClose(windows[2])) {

			if (display == 1) glfwPollEvents();

			char cell_name[18] = { '\0' };
			sprintf(&cell_name[strlen(cell_name)], "Cell (%d, %d)", current_cell[0], current_cell[1]);
			glfwSetWindowTitle(windows[1], cell_name);

			glfwSetWindowTitle(windows[2], carcin_names[current_carcin]);

			if (!paused) fAnimTimer(dataBlock, true, ticks);

			if (!paused) update_window(0, ticks);
			update_window(1, ticks, current_cell[0]*grid_size+current_cell[1], current_cell[0]*grid_size+current_cell[1]);
			if (paused) glfwSwapBuffers(windows[1]);
			for (int i = 0; i < NUM_CARCIN; i++)
				if ((paused && i == current_carcin) || !paused)
					update_window(2, ticks, i, current_carcin);

			if (!paused) fAnimTimer(dataBlock, false, ticks);

			if (!paused) ticks++;
		}
	}

	static void change_current(GLFWwindow *window, int *current, int n, bool incr=true) {
		int amount = -1; int limit = 0; int prev = n-1;
		if (incr) { amount = 1; limit = n-1; prev = 0; }
		if (*current != limit) *current += amount;
		else *current = prev;
	}

	static void key_ca(GLFWwindow *window, int key, int scancode, int action, int mods) {
		GPUAnimBitmap *bitmap = *(get_bitmap_ptr());
		switch (key) {
			case GLFW_KEY_ESCAPE:
				if (action == GLFW_PRESS) {
					glfwSetWindowShouldClose(window, GLFW_TRUE);
					exit(EXIT_SUCCESS);
				}
				break;
			case GLFW_KEY_SPACE:
				if (action == GLFW_PRESS) {
					if (bitmap->paused) bitmap->paused = false;
					else bitmap->paused = true;
				}
				break;
			case GLFW_KEY_RIGHT:
				if (action == GLFW_PRESS) bitmap->paused = false;
				else if (action == GLFW_RELEASE) bitmap->paused = true;
				break;
			case GLFW_KEY_K:
				if (action == GLFW_PRESS) {
					if (bitmap->excise == false) {
						bitmap->excise = true;
						printf("Tumour excision mode is activated.\n");
					}
					else {
						bitmap->excise = false;
						printf("Tumour excision mode is deactivated.\n");
					}
				}
				break;
		}
	}

	static void key_carcin(GLFWwindow *window, int key, int scancode, int action, int mods) {
		GPUAnimBitmap *bitmap = *(get_bitmap_ptr());
		switch (key) {
			case GLFW_KEY_ESCAPE:
				if (action == GLFW_PRESS) {
					glfwSetWindowShouldClose(window, GLFW_TRUE);
					exit(EXIT_SUCCESS);
				}
				break;
			case GLFW_KEY_SPACE:
				if (action == GLFW_PRESS) {
					if (bitmap->paused) bitmap->paused = false;
					else bitmap->paused = true;
				}
				break;
			case GLFW_KEY_RIGHT:
				if (action == GLFW_PRESS)
					change_current(window, &bitmap->current_carcin, NUM_CARCIN);
				break;
			case GLFW_KEY_LEFT:
				if (action == GLFW_PRESS)
					change_current(window, &bitmap->current_carcin, NUM_CARCIN, false);
				break;
		}
	}

	static void key_cell(GLFWwindow *window, int key, int scancode, int action, int mods) {
		GPUAnimBitmap *bitmap = *(get_bitmap_ptr());
		switch (key) {
			case GLFW_KEY_ESCAPE:
				if (action == GLFW_PRESS) {
					glfwSetWindowShouldClose(window, GLFW_TRUE);
					exit(EXIT_SUCCESS);
				}
				break;
			case GLFW_KEY_SPACE:
				if (action == GLFW_PRESS) {
					if (bitmap->paused) bitmap->paused = false;
					else bitmap->paused = true;
				}
				break;
			case GLFW_KEY_RIGHT:
				if (action == GLFW_PRESS)
					change_current(window, &bitmap->current_cell[0], bitmap->grid_size);
				break;
			case GLFW_KEY_LEFT:
				if (action == GLFW_PRESS)
					change_current(window, &bitmap->current_cell[0], bitmap->grid_size, false);
				break;
			case GLFW_KEY_UP:
				if (action == GLFW_PRESS)
					change_current(window, &bitmap->current_cell[1], bitmap->grid_size);
				break;
			case GLFW_KEY_DOWN:
				if (action == GLFW_PRESS)
					change_current(window, &bitmap->current_cell[1], bitmap->grid_size, false);
				break;
			case GLFW_KEY_S:
				if (action == GLFW_PRESS && bitmap->paused) {
					char fname[100] = { '\0' };
					sprintf(&fname[strlen(fname)], "cell(%d, %d)_", bitmap->current_cell[0], bitmap->current_cell[1]);
					int dig_max = numDigits(bitmap->maxT);
					int dig = numDigits(bitmap->ticks-1);
					for (int i = 0; i < dig_max-dig; i++) strcat(fname, "0");
					sprintf(&fname[strlen(fname)], "%d.png", bitmap->ticks-1);
					unsigned char *frame;
					CudaSafeCall(cudaMallocManaged((void**)&frame, bitmap->width*bitmap->height*4*sizeof(unsigned char)));
					CudaSafeCall(cudaMemPrefetchAsync(frame, bitmap->width*bitmap->height*4*sizeof(unsigned char), 1, NULL));
					dim3 blocks(bitmap->width/BLOCK_SIZE, bitmap->height/BLOCK_SIZE);
					dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
					copy_frame<<< blocks, threads >>>(bitmap->devPtrs[1], frame);
					CudaCheckError();
					CudaSafeCall(cudaDeviceSynchronize());

					unsigned error = lodepng_encode32_file(fname, frame, bitmap->width, bitmap->height);
					if (error) printf("error %u: %s\n", error, lodepng_error_text(error));

					CudaSafeCall(cudaFree(frame));
			}
			break;
		}
	}

	void draw(int window_idx, int w, int h) {
		glfwMakeContextCurrent(windows[window_idx]);
		glClearColor(0.0, 0.0, 0.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT);
		glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		if (!paused) glfwSwapBuffers(windows[window_idx]);
	}
};

#endif  // __GPU_ANIM_H__

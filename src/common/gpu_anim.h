/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */


#ifndef __GPU_ANIM_H__
#define __GPU_ANIM_H__

#include "general.h"
#include <GL/glut.h>
#include <GL/glx.h>
#include <GL/glext.h>

#include <cuda_gl_interop.h>
#include <iostream>
#include <unistd.h>

#define GET_PROC_ADDRESS(str) glXGetProcAddress((const GLubyte *)str)

PFNGLBINDBUFFERARBPROC glBindBuffer = NULL;
PFNGLDELETEBUFFERSARBPROC glDeleteBuffers = NULL;
PFNGLGENBUFFERSARBPROC glGenBuffers = NULL;
PFNGLBUFFERDATAARBPROC glBufferData = NULL;

struct GPUAnimBitmap {
    GLuint  bufferObj;
    cudaGraphicsResource *resource;
    int width, height;
    void *dataBlock;
    void (*fAnim)(uchar4*,void*,int);
    void (*animExit)(void*);
    void (*clickDrag)(void*,int,int,int,int);
    int dragStartX, dragStartY;
    int maxT;
    int save_frames;
    int display;

    GPUAnimBitmap(int w, int h, void *d=NULL, int T=200, int save=1, int show=1) {
        width = w;
        height = h;
        dataBlock = d;
        clickDrag = NULL;
	maxT = T;
	save_frames = save;
	display = show;

	int c=1;
	char* temp = (char*)malloc(sizeof(char));
	temp[0] = 'a';
       	glutInit(&c, &temp);
	free(temp);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(width, height);
	glutCreateWindow("CA");

        glBindBuffer    = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
        glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
        glGenBuffers    = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
        glBufferData    = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");

        glGenBuffers(1, &bufferObj);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
        glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * 4,
                     NULL, GL_DYNAMIC_DRAW_ARB);

        cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone);
    }

    ~GPUAnimBitmap(void) {
        free_resources();
    }

    void hide_window(void) {
	glutHideWindow();
    }

    void free_resources(void) {
        cudaGraphicsUnregisterResource(resource);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        glDeleteBuffers(1, &bufferObj);
    }


    long image_size(void) const { return width * height * 4; }

    void click_drag(void (*f)(void*,int,int,int,int)) {
        clickDrag = f;
    }

    void anim_and_exit(void (*f)(uchar4*,void*,int), void(*e)(void*)) {
        GPUAnimBitmap** bitmap = get_bitmap_ptr();
        *bitmap = this;
        fAnim = f;
        animExit = e;

	glutKeyboardFunc(Key);
	if (display == 1) {
		glutDisplayFunc(Draw);
	}
        if (clickDrag != NULL)
            glutMouseFunc(mouse_func);
        glutIdleFunc(idle_func);
	glutMainLoop();
    }

    // static method used for glut callbacks
    static GPUAnimBitmap** get_bitmap_ptr(void) {
        static GPUAnimBitmap* gBitmap;
        return &gBitmap;
    }

    // static method used for glut callbacks
    static void mouse_func(int button, int state, int mx, int my) {
        if (button == GLUT_LEFT_BUTTON) {
            GPUAnimBitmap* bitmap = *(get_bitmap_ptr());
            if (state == GLUT_DOWN) {
                bitmap->dragStartX = mx;
                bitmap->dragStartY = my;
            } else if (state == GLUT_UP) {
                bitmap->clickDrag(bitmap->dataBlock,
                                  bitmap->dragStartX,
                                  bitmap->dragStartY,
                                  mx, my);
            }
        }
    }

    // static method used for glut callbacks
    static void idle_func(void) {
	static int ticks = 0;
        GPUAnimBitmap* bitmap = *(get_bitmap_ptr());
        uchar4* devPtr;
        size_t size;

        cudaGraphicsMapResources(1, &(bitmap->resource), NULL);
        cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, bitmap->resource);

        bitmap->fAnim(devPtr, bitmap->dataBlock, ticks++);

        cudaGraphicsUnmapResources(1, &(bitmap->resource), NULL);

	if (bitmap->display == 1)
		glutPostRedisplay();

	if (ticks == bitmap->maxT+1) {
		if (bitmap->save_frames == 1) {
			if (numDigits(bitmap->maxT) == 1)
                        	system("ffmpeg -y -v quiet -framerate 5 -start_number 0 -i %d.png -c:v libx264 -pix_fmt yuv420p out.mp4");
                	else if (numDigits(bitmap->maxT) == 2)
                       		system("ffmpeg -y -v quiet -framerate 5 -start_number 0 -i %02d.png -c:v libx264 -pix_fmt yuv420p out.mp4");
                	else if (numDigits(bitmap->maxT) == 3)
                        	system("ffmpeg -y -v quiet -framerate 5 -start_number 0 -i %03d.png -c:v libx264 -pix_fmt yuv420p out.mp4");
		}
		char command[15] = { 'k', 'i', 'l', 'l', ' ' };
		int digs_pid = numDigits(getpid());
		char pid[10];
		sprintf(pid, "%d", getpid());
		for (int i = 5; i < digs_pid+5; i++) command[i] = pid[i-5];
		system(command);
	}
    }

    // static method used for glut callbacks
    static void Key(unsigned char key, int x, int y) {
        switch (key) {
            case 27:
                GPUAnimBitmap* bitmap = *(get_bitmap_ptr());
                if (bitmap->animExit)
                    bitmap->animExit(bitmap->dataBlock);
                bitmap->free_resources();
                exit(0);
        }
    }

    // static method used for glut callbacks
    static void Draw(void) {
        GPUAnimBitmap* bitmap = *(get_bitmap_ptr());
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawPixels(bitmap->width, bitmap->height, GL_RGBA,
                     GL_UNSIGNED_BYTE, 0);
        glutSwapBuffers();
    }
};

#endif  // __GPU_ANIM_H__

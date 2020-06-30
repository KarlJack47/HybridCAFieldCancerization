#ifndef __GUI_H__
#define __GUI_H__

#include "../common/glad/glad.h"
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>
#include <iostream>
#include <unistd.h>

int save_image(uchar4*,size_t,unsigned,char*,unsigned,int,unsigned);

static void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

struct GUI {
    const static unsigned nWindows = 4;
    GLFWwindow *windows[nWindows];
    GLuint bufferObjs[nWindows];
    cudaGraphicsResource *resrcs[nWindows];
    uchar4 *devPtrs[nWindows]; size_t sizes[nWindows];
    void *dataBlock;
    void (*fAnimCA)(uchar4*,unsigned,void*,unsigned,bool,bool,bool,
                    unsigned, unsigned,unsigned,bool*,bool*);
    void (*fAnimGenes)(uchar4*,unsigned,void*,
                       unsigned,bool,bool);
    void (*fAnimCarcin)(uchar4*,unsigned,void*,unsigned,
                        unsigned,bool,bool);
    void (*fAnimCell)(uchar4*,unsigned,void*,
                      unsigned,unsigned,bool);
    void (*fAnimTimerAndSaver)(void*,bool,unsigned,bool,bool);
    bool display, paused, excise, *perfectExcision, windowsShouldClose,
         detached[nWindows-1], keys[8], *carcinogens;
    unsigned width, height, ticks, gridSize, *nCarcin, maxNCarcin, *maxT,
             currCarcin, currContext, currCell[2], radius, centerX, centerY;
    char **carcinNames;

    GUI()
    {
        fAnimCA = NULL;
        fAnimGenes = NULL;
        fAnimCarcin = NULL;
        fAnimCell = NULL;
        carcinNames = NULL;
        carcinogens = NULL;
    }

    GUI(unsigned w=1024, unsigned h=1024, void *d=NULL, bool show=true,
        bool *perfectexcision=NULL, unsigned gSize=256,
        unsigned *T=NULL, unsigned *ncarcin=NULL, unsigned maxncarcin=2,
        bool *carcin=NULL, char **carNames=NULL)
    {
        unsigned i;

        width = w;
        height = h;
        dataBlock = d;
        display = show;
        paused = false;
        if (display) paused = true;
        excise = false;
        perfectExcision = perfectexcision;
        radius = 0;
        gridSize = gSize;
        maxT = T;
        nCarcin = ncarcin; maxNCarcin = maxncarcin;
        if (carcin != NULL)
            carcinogens = carcin;

        glfwSetErrorCallback(error_callback);

        if (!glfwInit())
            return;

        if (carNames != NULL) {
            carcinNames = (char**)malloc(maxNCarcin*sizeof(char*));
            for (i = 0; i < maxNCarcin; i++) {
                carcinNames[i] = (char*)calloc((strlen(carNames[i])+1), 1);
                strcpy(carcinNames[i], carNames[i]);
            }
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

        char windowName[20] = { '\0' };

        strcpy(windowName, "CA");
        create_window(CA_GRID, width, height, windowName,
                      &key_ca, &mouse_button_ca);
        if (!windows[CA_GRID]) return;

        strcpy(windowName, "Gene Families");
        create_window(GENE_INFO, width, height, windowName, &key_genes, NULL);
        if (!windows[GENE_INFO]) return;

        strcpy(windowName, "Cell (0, 0)");
        create_window(CELL_INFO, width, height, windowName, &key_cell, NULL);
        if (!windows[CELL_INFO]) return;

        strcpy(windowName, "Carcinogens");
        if (carcinNames != NULL)
            for (i = 0; i < maxNCarcin; i++)
                if (carcinogens[i]) {
                    strcpy(windowName, carcinNames[i]);
                    currCarcin = i;
                    break;
                }
        create_window(CARCIN_GRID, width, height, windowName,
                      &key_carcin, NULL);
    }

    static GUI** get_gui_ptr(void)
    {
        static GUI *gui;

        return &gui;
    }

    void hide_window(GLFWwindow *window)
    {
        if (window != NULL)
            glfwHideWindow(window);
    }

    void show_window(GLFWwindow *window)
    {
        if (window != NULL)
            glfwShowWindow(window);
    }

    void free_resources(void)
    {
        unsigned i;

        if (carcinNames != NULL) {
            for (i = 0; i < maxNCarcin; i++) {
                free(carcinNames[i]); carcinNames[i] = NULL;
            }
            free(carcinNames); carcinNames = NULL;
        }

        for (i = 0; i < nWindows; i++) {
            if (!windows[i]) continue;
            CudaSafeCall(cudaGraphicsUnregisterResource(resrcs[i]));
            glfwMakeContextCurrent(windows[i]);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
            glDeleteBuffers(1, &bufferObjs[i]);
        }

        glfwTerminate();
    }

    long image_size(void) const
    {
        return width * height * sizeof(uchar4);
    }

    void create_window(unsigned windowIdx, unsigned w, unsigned h, char *name,
                       void(*key)(GLFWwindow *,int,int,int,int),
                       void(*mouse)(GLFWwindow *, int, int, int))
    {
        windows[windowIdx] = glfwCreateWindow(w, h, name, NULL, NULL);
        if (!windows[windowIdx]) {
            glfwTerminate();
            return;
        }

        glfwMakeContextCurrent(windows[windowIdx]);

        glfwSetKeyCallback(windows[windowIdx], key);
        glfwSetMouseButtonCallback(windows[windowIdx], mouse);

        if (windowIdx == CA_GRID)
            gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);

        glGenBuffers(1, &bufferObjs[windowIdx]);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObjs[windowIdx]);
        glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, image_size(),
                     NULL, GL_DYNAMIC_DRAW_ARB);
        CudaSafeCall(cudaGraphicsGLRegisterBuffer(&resrcs[windowIdx],
                                                  bufferObjs[windowIdx],
                                                  cudaGraphicsRegisterFlagsNone));

        glfwSwapInterval(1);
    }

    void update_window(unsigned windowIdx, unsigned ticks,
                       unsigned idx=0, unsigned curr=0)
    {
        CudaSafeCall(cudaGraphicsMapResources(1, &resrcs[windowIdx], NULL));
        CudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&devPtrs[windowIdx],
                                                          &sizes[windowIdx],
                                                          resrcs[windowIdx]));

        if (windowIdx == CA_GRID)
            fAnimCA(devPtrs[windowIdx], width, dataBlock, ticks, display,
                    paused, excise, radius, centerX, centerY,
                    &windowsShouldClose, keys);
        else if (windowIdx == GENE_INFO)
            fAnimGenes(devPtrs[windowIdx], width, dataBlock, ticks, display,
                       paused);
        else if (windowIdx == CELL_INFO)
            fAnimCell(devPtrs[windowIdx], width, dataBlock, idx, ticks,
                      display);
        else if (windowIdx == CARCIN_GRID)
            fAnimCarcin(devPtrs[windowIdx], width, dataBlock, idx, ticks,
                        display, paused);

        CudaSafeCall(cudaGraphicsUnmapResources(1, &resrcs[windowIdx], NULL));

        if (display && (currContext == windowIdx
         || (windowIdx != 0 && detached[windowIdx-1]))) {
            if (idx == curr)
                draw(windowIdx, width, height);
        }
    }

    void anim(void(*fCA)(uchar4*,unsigned,void*,unsigned,bool,bool,bool,
                         unsigned,unsigned,unsigned,bool*,bool*),
              void(*fGenes)(uchar4*,unsigned,void*,
                            unsigned,bool,bool),
              void(*fCarcin)(uchar4*,unsigned,void*,unsigned,
                             unsigned,bool,bool),
              void(*fCell)(uchar4*,unsigned,void*,
                           unsigned,unsigned,bool),
              void(*fTimeAndSave)(void*,bool,unsigned,bool,bool))
    {
        //                              s    e    p   c   a   b    x    h
        unsigned i, idx, keyVal[8] = { 115, 101, 112, 99, 97, 98, 120, 104 };
        bool exciseBefore, displayBefore, changeMaxT;
        int xpos, ypos, key;
        struct termios oldt;
        char cellName[18] = { '\0' };
        GUI **gui = get_gui_ptr();

        *gui = this;
        fAnimCA = fCA;
        fAnimGenes = fGenes;
        fAnimCarcin = fCarcin;
        fAnimCell = fCell;
        fAnimTimerAndSaver = fTimeAndSave;
        currCell[0] = 0; currCell[1] = 0;
        currContext = 0;
        detached[0] = false; detached[1] = false; detached[2] = false;
        windowsShouldClose = false;
        ticks = 0;
        if (display) show_window(windows[0]);

        while (!windowsShouldClose) {
            exciseBefore = excise; displayBefore = display;
            changeMaxT = false;
            for (i = 0; i < 8; i++)
                keys[i] = false;
            set_terminal_mode(&oldt);
            while (kbhit()) {
                key = getch();
                if (key == 32) // space bar
                    paused ? paused = false : paused = true;
                else if (key == 27) // ESC key
                    windowsShouldClose = true;
                else if (key == 107) // k key
                    excise ? excise = false : excise = true;
                else if (key == 103) // g key
                    display ? display = false : display = true;
                else if (key == 116) // t key
                    changeMaxT ? changeMaxT = false : changeMaxT = true;
                for (i = 0; i < 8; i++)
                    if (key == keyVal[i]) {
                        keys[i] ? keys[i] = false : keys[i] = true;
                        break;
                    }
            }
            reset_terminal_mode(&oldt);
            fflush(stdout);
            if (excise && exciseBefore != excise) {
                printf("Excision mode is activated.\n");
                printf("Pick circle center location (0-%d 0-%d): ",
                       gridSize, gridSize);
                scanf("%d %d", &centerX, &centerY);
                printf("Pick the radius of excision: ");
                scanf("%d", &radius);
            }
            else if (!excise && exciseBefore != excise)
                printf("Excision mode is deactivated.\n");
            if (displayBefore != display)
                for (i = 0; i < nWindows; i++)
                    if (i == CA_GRID || detached[i-1]) {
                        if (display) show_window(windows[i]);
                        else hide_window(windows[i]);
                    }
            if (changeMaxT) {
                printf("Pick a new maximum time (<999999): ");
                scanf("%u", maxT);
            }

            if (glfwWindowShouldClose(windows[CA_GRID])
             || glfwWindowShouldClose(windows[GENE_INFO])
             || glfwWindowShouldClose(windows[CELL_INFO])
             || (nCarcin != 0 && glfwWindowShouldClose(windows[CARCIN_GRID]))) {
                idx = GENE_INFO;
                if (glfwWindowShouldClose(windows[CELL_INFO])) idx = CELL_INFO;
                else if (nCarcin != 0 && glfwWindowShouldClose(windows[CARCIN_GRID]))
                    idx = CARCIN_GRID;

                if (detached[idx-1]) {
                    detach_window(windows, idx-1, &currContext, detached);
                    glfwSetWindowShouldClose(windows[idx], GLFW_FALSE);
                } else if (!detached[idx-1]
                        || glfwWindowShouldClose(windows[0])) {
                    windowsShouldClose = true;
                    paused = true;
                }
            }

            glfwGetWindowPos(windows[currContext], &xpos, &ypos);
            for (i = 0; i < nWindows; i++) {
                if (i == currContext) continue;
                if (i == CA_GRID || (i != CA_GRID && !detached[i-1]))
                        glfwSetWindowPos(windows[i], xpos, ypos);
            }

            if (display) glfwPollEvents();

            if (display && (detached[1] || currContext == CELL_INFO)) {
                memset(cellName, '\0', 18);
                sprintf(cellName, "Cell (%d, %d)",
                        currCell[0], currCell[1]);
                glfwSetWindowTitle(windows[CELL_INFO], cellName);
            }

            if (display && (detached[2] || currContext == CARCIN_GRID))
                glfwSetWindowTitle(windows[CARCIN_GRID], carcinNames[currCarcin]);

            if (!paused || windowsShouldClose)
                fAnimTimerAndSaver(dataBlock, true, ticks, paused,
                                   windowsShouldClose);

            update_window(CA_GRID, ticks);
            if (paused) glfwSwapBuffers(windows[CA_GRID]);

            update_window(GENE_INFO, ticks);
            if (paused) glfwSwapBuffers(windows[GENE_INFO]);

            if (display && (detached[1] || currContext == CELL_INFO)) {
                update_window(CELL_INFO, ticks,
                              currCell[1]*gridSize+currCell[0],
                              currCell[1]*gridSize+currCell[0]);
                if (paused) glfwSwapBuffers(windows[CELL_INFO]);
            }

            if (*nCarcin != 0)
                for (i = 0; i < maxNCarcin; i++) {
                    if (carcinogens[i] && ((paused && i == currCarcin)
                     || !paused)) {
                        update_window(CARCIN_GRID, ticks, i, currCarcin);
                        if (paused) glfwSwapBuffers(windows[CARCIN_GRID]);
                    }
                }

            if (windowsShouldClose)
                for (i = 0; i < nWindows; i++)
                    if (i == CA_GRID || detached[i-1])
                        hide_window(windows[i]);

            if (!paused || windowsShouldClose)
                fAnimTimerAndSaver(dataBlock, false, ticks, paused,
                                   windowsShouldClose);

            if (!paused) ticks++;

            if (ticks == *maxT+1) break;
        }
    }

    static void change_current(unsigned *curr, unsigned n, bool incr=true)
    {
        int amount = -1; int limit = 0; int prev = n-1;
        if (incr) { amount = 1; limit = n-1; prev = 0; }
        if (*curr != limit) *curr += amount;
        else *curr = prev;
    }

    static void change_context(GLFWwindow **windows, unsigned *curr,
                               bool *separated, unsigned n = 4, bool incr=true)
    {
        unsigned i;

        if (separated[0] && separated[1] && (n == 4 && separated[2])) {
            *curr = CA_GRID;
            glfwShowWindow(windows[CA_GRID]);
            return;
        }

        change_current(curr, n, incr);
        while (*curr != CA_GRID && separated[*curr-1]) {
            if (*curr == CA_GRID) break;
            change_current(curr, n, incr);
        }

        for (i = 0; i < n; i++) {
            if (i == *curr || (i != CA_GRID && separated[i-1]))
                glfwShowWindow(windows[i]);
            else glfwHideWindow(windows[i]);
        }

        glfwFocusWindow(windows[*curr]);
    }

    static void detach_window(GLFWwindow **windows, unsigned windowIdx,
                              unsigned *curr, bool *separated)
    {
        int xpos, ypos = 0;

        if (separated[windowIdx]) {
            separated[windowIdx] = false;
            glfwHideWindow(windows[*curr]);
            glfwGetWindowPos(windows[*curr], &xpos, &ypos);
            glfwSetWindowPos(windows[windowIdx+1], xpos, ypos);
            *curr = windowIdx+1;
        } else {
            separated[windowIdx] = true;
            change_context(windows, curr, separated);
            glfwFocusWindow(windows[windowIdx+1]);
        }
    }

    static void key_ca(GLFWwindow *window, int key,
                       int scancode, int action, int mods)
    {
        GUI *gui = *(get_gui_ptr());
        unsigned n = 4;
        if (*gui->nCarcin == 0) n = 3;

        switch (key) {
            case GLFW_KEY_ESCAPE:
                if (action == GLFW_PRESS) {
                    glfwSetWindowShouldClose(window, GLFW_TRUE);
                    gui->windowsShouldClose = true;
                }
                break;
            case GLFW_KEY_SPACE:
                if (action == GLFW_PRESS) {
                    if (gui->paused) gui->paused = false;
                    else gui->paused = true;
                }
                break;
            case GLFW_KEY_RIGHT:
                if (action == GLFW_PRESS) gui->paused = false;
                else if (action == GLFW_RELEASE) gui->paused = true;
                break;
            case GLFW_KEY_K:
                if (action == GLFW_PRESS) {
                    if (!gui->excise) {
                        gui->excise = true;
                        printf("Excision mode is activated.\n");
                    }
                    else {
                        gui->excise = false;
                        printf("Excision mode is deactivated.\n");
                    }
                }
                break;
            case GLFW_KEY_A:
                if (action == GLFW_PRESS)
                    change_context(gui->windows, &gui->currContext,
                                   gui->detached, n, false);
                break;
            case GLFW_KEY_D:
                if (action == GLFW_PRESS)
                    change_context(gui->windows, &gui->currContext,
                                   gui->detached, n);
                break;
        }
    }

    static void key_genes(GLFWwindow *window, int key,
                          int scancode, int action, int mods)
    {
        GUI *gui = *(get_gui_ptr());
        unsigned n = 4;
        if (*gui->nCarcin == 0) n = 3;

        switch (key) {
            case GLFW_KEY_ESCAPE:
                if (action == GLFW_PRESS) {
                    glfwSetWindowShouldClose(window, GLFW_TRUE);
                    gui->windowsShouldClose = true;
                }
                break;
            case GLFW_KEY_SPACE:
                if (action == GLFW_PRESS) {
                    if (gui->paused) gui->paused = false;
                    else gui->paused = true;
                }
                break;
            case GLFW_KEY_A:
                if (!gui->detached[0] && action == GLFW_PRESS)
                    change_context(gui->windows, &gui->currContext,
                                   gui->detached, n, false);
                break;
            case GLFW_KEY_D:
                if (!gui->detached[0] && action == GLFW_PRESS)
                    change_context(gui->windows, &gui->currContext,
                                   gui->detached, n);
                break;
            case GLFW_KEY_X:
                if (action == GLFW_PRESS)
                    detach_window(gui->windows, 0,
                                  &gui->currContext, gui->detached);
                break;
        }
    }

    static void key_cell(GLFWwindow *window, int key,
                         int scancode, int action, int mods)
    {
        char prefix[18] = { '\0' };
        GUI *gui = *(get_gui_ptr());
        unsigned n = 4;
        if (*gui->nCarcin == 0) n = 3;

        switch (key) {
            case GLFW_KEY_ESCAPE:
                if (action == GLFW_PRESS) {
                    glfwSetWindowShouldClose(window, GLFW_TRUE);
                    gui->windowsShouldClose = true;
                }
                break;
            case GLFW_KEY_SPACE:
                if (action == GLFW_PRESS) {
                    if (gui->paused) gui->paused = false;
                    else gui->paused = true;
                }
                break;
            case GLFW_KEY_RIGHT:
                if (action == GLFW_PRESS)
                    change_current(&gui->currCell[0],
                                   gui->gridSize);
                break;
            case GLFW_KEY_LEFT:
                if (action == GLFW_PRESS)
                    change_current(&gui->currCell[0],
                                   gui->gridSize,
                                   false);
                break;
            case GLFW_KEY_UP:
                if (action == GLFW_PRESS)
                    change_current(&gui->currCell[1],
                                   gui->gridSize);
                break;
            case GLFW_KEY_DOWN:
                if (action == GLFW_PRESS)
                    change_current(&gui->currCell[1],
                                   gui->gridSize,
                                   false);
                break;
            case GLFW_KEY_S:
                if (action == GLFW_PRESS && gui->paused) {
                    memset(prefix, '\0', 17);
                    sprintf(prefix, "cell(%d, %d)_", gui->currCell[0],
                            gui->currCell[1]);
                    save_image(gui->devPtrs[2], gui->width, 16, prefix,
                               gui->ticks-1, 0, 7);
                }
                break;
            case GLFW_KEY_A:
                if (!gui->detached[1] && action == GLFW_PRESS)
                    change_context(gui->windows, &gui->currContext,
                                   gui->detached, n, false);
                break;
            case GLFW_KEY_D:
                if (!gui->detached[1] && action == GLFW_PRESS)
                    change_context(gui->windows, &gui->currContext,
                                   gui->detached, n);
                break;
            case GLFW_KEY_X:
                if (action == GLFW_PRESS)
                    detach_window(gui->windows, 1, &gui->currContext,
                                  gui->detached);
                break;
        }
    }

    static void key_carcin(GLFWwindow *window, int key,
                           int scancode, int action, int mods)
    {
        GUI *gui = *(get_gui_ptr());

        switch (key) {
            case GLFW_KEY_ESCAPE:
                if (action == GLFW_PRESS) {
                    glfwSetWindowShouldClose(window, GLFW_TRUE);
                    gui->windowsShouldClose = true;
                }
                break;
            case GLFW_KEY_SPACE:
                if (action == GLFW_PRESS) {
                    if (gui->paused) gui->paused = false;
                    else gui->paused = true;
                }
                break;
            case GLFW_KEY_RIGHT:
                if (action == GLFW_PRESS)
                    do
                        change_current(&gui->currCarcin, gui->maxNCarcin);
                    while (!gui->carcinogens[gui->currCarcin]);
                break;
            case GLFW_KEY_LEFT:
                if (action == GLFW_PRESS)
                    do
                        change_current(&gui->currCarcin, gui->maxNCarcin, false);
                    while (!gui->carcinogens[gui->currCarcin]);
                break;
            case GLFW_KEY_A:
                if (!gui->detached[2] && action == GLFW_PRESS)
                    change_context(gui->windows, &gui->currContext,
                                   gui->detached, 4, false);
                break;
            case GLFW_KEY_D:
                if (!gui->detached[2] && action == GLFW_PRESS)
                    change_context(gui->windows, &gui->currContext,
                                   gui->detached, 4);
                break;
            case GLFW_KEY_X:
                if (action == GLFW_PRESS)
                    detach_window(gui->windows, 2, &gui->currContext,
                                  gui->detached);
                break;
        }
    }

    static void mouse_button_ca(GLFWwindow *window, int button,
                                int action, int mods)
    {
        GUI *gui = *(get_gui_ptr());
        double xpos, ypos;
        int cellX, cellY;

        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
            glfwGetCursorPos(window, &xpos, &ypos);

            cellX = (int) ((xpos / (double) gui->width)
                            * (double) gui->gridSize);
            cellY = (int) (((gui->height - ypos) / (double) gui->height)
                            * (double) gui->gridSize);

            if (cellX >= gui->gridSize) cellX = gui->gridSize - 1;
            if (cellY >= gui->gridSize) cellY = gui->gridSize - 1;
            gui->currCell[0] = cellX;
            gui->currCell[1] = cellY;

            if (!(*gui->perfectExcision) && gui->excise && gui->paused) {
                gui->centerX = cellX;
                gui->centerY = cellY;
                printf("Circle center location (%d, %d)\n",
                       gui->centerX, gui->centerY);
                printf("Pick the radius of excision: ");
                scanf("%d", &gui->radius);
            }
        }
    }

    void draw(unsigned windowIdx, unsigned w, unsigned h)
    {
        glfwMakeContextCurrent(windows[windowIdx]);
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        if (!paused) glfwSwapBuffers(windows[windowIdx]);
    }
};

#endif  // __GUI_H__
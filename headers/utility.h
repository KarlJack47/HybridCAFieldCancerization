#ifndef __UTILITY_H__
#define __UTILITY_H__

void reset_terminal_mode(struct termios *oldt)
{
    tcsetattr(STDIN_FILENO, TCSANOW, oldt);
}

void set_terminal_mode(struct termios *oldt)
{
    struct termios newt;

    tcgetattr(STDIN_FILENO, oldt);
    memcpy(&newt, oldt, sizeof(newt));
    cfmakeraw(&newt);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
}

int kbhit(void)
{
    struct timeval tv = { 0L, 0L };
    fd_set fds;

    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);

    return select(1, &fds, NULL, NULL, &tv);
}

int getch(void)
{
    int out;
    unsigned char c;

    if ((out = read(STDIN_FILENO, &c, sizeof(c))) < 0)
        return out;
    else
        return c;
}

__host__ __device__ unsigned num_digits(double x)
{
    x = abs(x);

    return (x <         10 ? 1 :
           (x <        100 ? 2 :
           (x <       1000 ? 3 :
           (x <      10000 ? 4 :
           (x <     100000 ? 5 :
           (x <    1000000 ? 6 :
           (x <   10000000 ? 7 :
           (x <  100000000 ? 8 :
           (x < 1000000000 ? 9 :
                            10
           )))))))));
}

void print_progress(unsigned currAmount, unsigned N)
{
    char format[40] = { '\0' };
    double progress = (currAmount / (double) N) * 100.0;
    unsigned numDig = num_digits(progress), limit = numDig + 10, k;

    if (trunc(progress * 1000.0) >= 9995 && numDig == 1) limit += 1;
    else if (trunc(progress * 1000.0) >= 99995 && numDig == 2) limit += 1;

    for (k = 0; k < limit; k++) strcat(format, "\b");
    strcat(format, "%.2f/%.2f");
    printf(format, progress, 100.0);
    fflush(stdout);
}

__host__ __device__ bool check_in_circle(unsigned x, unsigned y, unsigned gSize,
                                         unsigned r, unsigned cX, unsigned cY)
{
    unsigned rr, dd, eX, eY, incrLimit;
    int dX, dY, xDecr, yDecr;
    double bL, bR, bU, bD, aa, b;

    if (r == 0) return false;

    xDecr = (x - cX); yDecr = (y - cY);
    rr = r * r;
    if (xDecr * xDecr + yDecr * yDecr <= rr) {
        return true;
    }

    // Start of periodic cases
    incrLimit = gSize - 1;
    dX = (int) incrLimit - (int) cX; dY = (int) incrLimit - (int) cY;
    bL = (int) r - (int) cX; bR = (int) r - dX;
    bD = (int) r - (int) cY; bU = (int) r - dY;
    if (!(bL > 0.0 || bR > 0.0 || bD > 0.0 || bU > 0.0))
        return false;

    if (bR > 0.0) { // right
        aa = rr - dX * dX;
        eX = 0; eY = cY;
        xDecr = y - eY; yDecr = x - eX;
        if ((xDecr * xDecr) / aa + (yDecr * yDecr) / (bR * bR) <= 1.0)
            return true;
    }
    if (bL > 0.0) { // left
        aa = rr - cX * cX;
        eX = incrLimit; eY = cY;
        xDecr = y - eY; yDecr = x - eX;
        if ((xDecr * xDecr) / aa + (yDecr * yDecr) / (bL * bL) <= 1.0)
            return true;
    }
    if (bU > 0.0) { // up
        aa = rr - dY * dY;
        eX = cX; eY = 0;
        xDecr = x - eX; yDecr = y - eY;
        if ((xDecr * xDecr) / aa + (yDecr * yDecr) / (bU * bU) <= 1.0)
            return true;
    }
    if (bD > 0.0) { // down
        aa = rr - cY * cY;
        eX = cX; eY = incrLimit;
        xDecr = x - eX; yDecr = y - eY;
        if ((xDecr * xDecr) / aa + (yDecr * yDecr) / (bD * bD) <= 1.0)
            return true;
    }
    if (bU > 0.0 && bL > 0.0) { // up-left
        dd = pow((double) ((int) cX - dY), 2.0);
        aa = rr - dd; b = (double) r - sqrt((double) dd);
        eX = incrLimit; eY = 0;
        xDecr = x - eX; yDecr = y - eY;
        if ((xDecr * xDecr) / aa + (yDecr * yDecr) / (b * b) <= 1.0)
            return true;
    }
    if (bU > 0.0 && bR > 0.0) { // up-right
        dd = pow((double) (dX - dY), 2.0);
        aa = rr - dd; b = (double) r - sqrt((double) dd);
        eX = 0; eY = 0;
        xDecr = x - eX; yDecr = y - eY;
        if ((xDecr * xDecr) / aa + (yDecr * yDecr) / (b * b) <= 1.0)
            return true;
    }
    if (bD > 0.0 && bL > 0.0) { // down-left
        dd = pow((double) ((int) cX - (int) cY), 2.0);
        aa = rr - dd; b = (double) r - sqrt((double) dd);
        eX = incrLimit; eY = incrLimit;
        xDecr = x - eX; yDecr = y - eY;
        if ((xDecr * xDecr) / aa + (yDecr * yDecr) / (b * b) <= 1.0)
            return true;
    }
    if (bD > 0.0 && bR > 0.0) { // down-right
        dd = pow((double) (dX - (int) cY), 2.0);
        aa = rr - dd; b = (double) r - sqrt((double) dd);
        eX = 0; eY = incrLimit;
        xDecr = x - eX; yDecr = y - eY;
        if ((xDecr * xDecr) / aa + (yDecr * yDecr) / (b * b) <= 1.0)
            return true;
    }
    // End of periodic cases

    return false;
}

__device__ char* num_to_string(double val, size_t *numChar, char *out=NULL,
                               unsigned *start=NULL, bool displayFrac=false,
                               bool displaySign=false, unsigned precision=10)
{
    int i; unsigned idx = 0;
    unsigned long numDigInt=num_digits(val),
                  frac = abs(val - (int) val) * pow(10, precision),
                  numDigFrac=num_digits(frac),
                  numZeros = precision - numDigFrac;
    unsigned lastDigitFrac;
    char *outTemp;

    if (start != NULL) idx = *start;

    *numChar = numDigInt+1;
    if (frac != 0 || (displayFrac && frac == 0)) {
        *numChar = *numChar + numDigFrac + numZeros + 1;
        lastDigitFrac = (int) (abs(val - (int) val) * pow(10, precision+1)) % 10;
        if (lastDigitFrac >= 5) frac++;
    }
    if (val < 0 || (displaySign && val >= 0))
        *numChar = *numChar+1;
    if (out == NULL) {
        outTemp = (char*)malloc(*numChar);
        outTemp[*numChar-1] = '\0';
    } else outTemp = out;

    if (val < 0) {
        outTemp[idx++] = '-';
        val *= -1;
    } else if (val >= 0 && displaySign) outTemp[idx++] = '+';

    for (i = numDigInt-1; i > -1; i--)
        outTemp[idx++] = (int) (val / pow(10, i)) % 10 + '0';

    if (frac == 0 && !displayFrac) {
        if (start != NULL) *start = idx;
        return outTemp;
    }

    outTemp[idx++] = '.';
    for (i = 0; i < numZeros; i++) outTemp[idx++] = '0';

    for (i = numDigFrac-1; i > -1; i--)
        outTemp[idx++] = (int) (frac / pow(10, i)) % 10 + '0';

    if (start != NULL) *start = idx;
    return outTemp;
}

__device__ void num_to_string_with_padding(double val, unsigned maxNumDig,
                                           char *out, unsigned *start=NULL,
                                           bool displayFrac=false)
{
    unsigned i, numSpaces, idx = 0;
    char *temp; size_t numChar;

    if (start != NULL) idx = *start;

    temp = num_to_string(val, &numChar, NULL, NULL, displayFrac);
    numSpaces = maxNumDig - numChar + 1;
    for (i = 0; i < numSpaces; i++) out[idx++] = ' ';
    for (i = 0; i < numChar-1; i++) out[idx++] = temp[i];
    free(temp); temp = NULL;

    if (start != NULL) *start = idx;
}

__device__ double atomicMax(double *address, double val)
{
    unsigned long long int * address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val > __longlong_as_double(assumed)
                                             ? val
                                             : __longlong_as_double(assumed)));
   } while (assumed != old);

   return __longlong_as_double(old);
}

#endif // __UTILITY_H__
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

static int user_input_va_list(const char *question, const char *inputFormats, va_list args) {
    const int maxNumVals = 10;
    int len = strlen(inputFormats)+1, numVals = 0, checkChar = 0, returnVal = 0, lenVal, lenDefault, i;
    char buffer[1024], inputFormatsCpy[len], *format[maxNumVals], *value, *endptr;
    char *strVar; unsigned *unsgnVar; int *intVar; float *floatVar; double *doubVar;
    char *allowedVals; char *defaultVal;
    unsigned minValUnsgn; unsigned maxValUnsgn;
    int minValInt; int maxValInt;
    float minValFloat; float maxValFloat;
    double minValDoub; double maxValDoub;

    strcpy(inputFormatsCpy, inputFormats);

    vprintf(question, args);
    fflush(stdout);

    if (!fgets(buffer, sizeof(buffer), stdin)) {
        printf("There was an error getting the input.\n");
        return 1;
    }
    format[0] = strtok(inputFormatsCpy, ",");
    while (format[numVals] != NULL)
        format[++numVals] = strtok(NULL, ",");
    for (i = 0; i < numVals; i++) {
        if (i == 0) value = strtok(buffer, ",");
        else value = strtok(NULL, ",");
        if (value == NULL) {
            printf("You didn't enter enough values, you should have entered %d values.\n",
                   numVals);
            return 2;
        }
        if (strchr(format[i], 's') != NULL || (checkChar = (strchr(format[i], 'c') != NULL))) {
            strVar = va_arg(args, char*);
            allowedVals = va_arg(args, char*);
            if (checkChar) {
                *strVar = value[0];
                if (strchr(allowedVals, *strVar) == NULL) {
                    printf("You entered the invalid character %c for input %d, the value has been set to the default of %c.\n",
                           *strVar, i+1, allowedVals[0]);
                    *strVar = allowedVals[0];
                    returnVal = 3;
                }
            } else {
                len = strlen(value);
                if (i == numVals - 1) len--;
                if (len > (lenVal = va_arg(args, int))) len -= len - lenVal;
                strncpy(strVar, value, len);
                strVar[len] = '\0';
                if (strstr(allowedVals, strVar) == NULL) {
                    lenDefault = strlen(allowedVals) - strlen(strchr(allowedVals, ','));
                    defaultVal = (char*)malloc((lenDefault + 1)*sizeof(char));
                    strncpy(defaultVal, allowedVals, lenDefault);
                    defaultVal[lenDefault] = '\0';
                    printf("You entered the invalid string %s for input %d, the value has been set to the default of %s.\n",
                           strVar, i+1, defaultVal);
                    strcpy(strVar, defaultVal);
                    free(defaultVal);
                    returnVal = 3;
                }
            }
        } else if (strchr(format[i], 'd') != NULL) {
            intVar = va_arg(args, int*);
            *intVar = strtol(value, &endptr, 10);
            minValInt = va_arg(args, int); maxValInt = va_arg(args, int);
            if (*intVar < minValInt) {
                printf("You entered the value %d which is too small for input %d, the value has been set to the minimum of %d.\n",
                       *intVar, i+1, minValInt);
                *intVar = minValInt;
                returnVal = 3;
            } else if (*intVar > maxValInt) {
                printf("You entered the value %d which is too large for input %d, the value has been set to the maximum of %d.\n",
                       *intVar, i+1, maxValInt);
                *intVar = maxValInt;
                returnVal = 3;
            }
        } else if (strchr(format[i], 'u') != NULL) {
            unsgnVar = va_arg(args, unsigned*);
            *unsgnVar = strtol(value, &endptr, 10);
            minValUnsgn = va_arg(args, unsigned); maxValUnsgn = va_arg(args, unsigned);
            if (*unsgnVar < minValUnsgn) {
                printf("You entered the value %u which is too small for input %d, the value has been set to the minimum of %u.\n",
                       *unsgnVar, i+1, minValUnsgn);
                *unsgnVar = minValUnsgn;
                returnVal = 3;
            } else if (*unsgnVar > maxValUnsgn) {
                printf("You entered the value %u which is too large for input %d, the value has been set to the maximum of %u.\n",
                       *unsgnVar, i+1, maxValUnsgn);
                *unsgnVar = maxValUnsgn;
                returnVal = 3;
            }
        } else if (strchr(format[i], 'f') != NULL) {
            floatVar = va_arg(args, float*);
            *floatVar = strtod(value, &endptr);
            minValFloat = va_arg(args, double); maxValFloat = va_arg(args, double);
            if (*floatVar < minValFloat) {
                printf("You entered the value %f which is too small for input %d, the value has been set to the minimum of %f.\n",
                       *floatVar, i+1, minValFloat);
                *floatVar = minValFloat;
                returnVal = 3;
            } else if (*floatVar > maxValFloat) {
                printf("You entered the value %f which is too large for input %d, the value has been set to the maximum of %f.\n",
                       *floatVar, i+1, maxValFloat);
                *floatVar = maxValFloat;
                returnVal = 3;
            }
        } else if (strchr(format[i], 'f') != NULL && strchr(format[i], 'l') != NULL) {
            doubVar = va_arg(args, double*);
            *doubVar = strtod(value, &endptr);
            minValDoub = va_arg(args, double); maxValDoub = va_arg(args, double);
            if (*doubVar < minValDoub) {
                printf("You entered the value %lf which is too small for input %d, the value has been set to the minimum of %lf.\n",
                       *doubVar, i+1, minValDoub);
                *doubVar = minValDoub;
                returnVal = 3;
            } else if (*doubVar > maxValDoub) {
                printf("You entered the value %lf which is too large for input %d, the value has been set to the maximum of %lf.\n",
                       *doubVar, i+1, maxValDoub);
                *intVar = maxValDoub;
                returnVal = 3;
            }
        }
        checkChar = 0;
    }

    return returnVal;
}

int user_input(const char* question, const char* inputFormats, ...) {
    int returnVal;
    va_list args;

    do {
        va_start(args, inputFormats);
        returnVal = user_input_va_list(question, inputFormats, args);
        va_end(args);
    } while (returnVal != 0);

    return returnVal;
}

int carcin_param_change(unsigned maxNCarcin, const char *question, const char *inputFormat, ...) {
    va_list args;
    unsigned carcinIdx;

    user_input("Enter a carcinogen index (0-%d): ", "%d", maxNCarcin-1, &carcinIdx, 0, maxNCarcin-1);
    if (carcinIdx < maxNCarcin) {
        va_start(args, inputFormat);
        user_input_va_list(question, inputFormat, args);
        va_end(args);
    }

    return carcinIdx;
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

// CUDA utility functions
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

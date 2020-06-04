#ifndef __UTILITY_H__
#define __UTILITY_H__

unsigned num_digits(double x)
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
}

#endif // __UTILITY_H__
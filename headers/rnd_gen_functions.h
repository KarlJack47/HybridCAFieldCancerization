#ifndef __RND_GEN_FUNCTIONS_H__
#define __RND_GEN_FUNCTIONS_H__

void set_seed(void)
{
    timespec seed;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &seed);
    srand(seed.tv_nsec);
}

#endif // __RND_GEN_FUNCTIONS_H__
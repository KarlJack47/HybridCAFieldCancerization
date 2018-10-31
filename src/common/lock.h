#ifndef __LOCK_H__
#define __LOCK_H__

#include "error_check.h"

struct Lock {
	int  *mutex;

	Lock() {
		int state = 0;

		CudaSafeCall(cudaMalloc((void**)&mutex, sizeof(int)));
		CudaSafeCall(cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice));
	}

	~Lock() {
		CudaSafeCall(cudaFree(mutex));
	}

	__device__ void lock() {
		while (atomicCAS(mutext, 0, 1) != 0);
	}

	__device__ void unlock() {
		atomicExch(mutex, 0);
	}
};

#endif // __LOCK_H__

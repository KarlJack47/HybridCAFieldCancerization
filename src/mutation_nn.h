#ifndef __MUTATION_NN_H__
#define __MUTATION_NN_H__

#ifndef MUTATION_PARAM
#define MUTATION_PARAM
#define RAND_INCR_A 5000000.0f // 0.005
#define RAND_INCR_B 10000000.0f // 0.01
#define RAND_DECR_A 10000000.0f // 0.01
#define RAND_DECR_B 100000000.0f // 0.1
#define CHANCE_UPREG 0.5f
#endif

__device__ void activation(int idx, double *input, double *output) {
	/*  Computes the value of the sigmoid function f(x) = 1/(1 + e^-2x).
            Inputs:
            input: array
            output: array, the results of the computation are to be stored here
	*/

	output[idx] = 1.0f / (1.0f + std::exp(-2*input[idx]));
}

__device__ double* dot(int idx, double *m1, double *m2, double *output, int m1_rows , int m1_columns, int m2_columns) {
	/*  Computes the product of two matrices: m1 x m2.
	    Inputs:
	    m1: array, left matrix of size m1_rows x m1_columns
	    m2: array, right matrix of size m1_columns x m2_columns (the number of rows in the right matrix
	        must be equal to the number of the columns in the left one)
	    output: array, the results of the computation are to be stored here
	    m1_rows: int, number of rows in the left matrix m1
	    m1_columns: int, number of columns in the left matrix m1
	    m2_columns: int, number of columns in the right matrix m2
	*/

	for (int i = idx; i < m1_rows*m2_columns; i += m2_columns*m1_rows) {
		int r = i / m2_columns;
		int c = i % m2_columns;
		double t_output = 0.0f;
		for(int k = 0; k < m1_columns; k++) {
			t_output += m1[r*m1_columns+k] * m2[k*m2_columns+c];
		}
		output[i] = t_output;
	}

	return output;
}

__device__ double* matrixAddMatrix(int idx, double *m1, double *m2, double *output) {
	/* Computes the (elementwise) addition between two arrays
	   Inputs:
	   m1: array
	   m2: array
	   output: array, the results of the computation are to be stored here
	*/

	output[idx] = m1[idx] + m2[idx];

	return output;
}

__device__ void feedforward(double *input, double *W_in, double *b_in, double *hidden, double *W_out, double *b_out, double *output,
			    int n_input, int n_hidden, int n_output) {

	for (int i = 0; i < n_hidden; i++) {
		activation(i, matrixAddMatrix(i, dot(i, W_in, input, hidden, n_hidden, n_input, 1), b_in, hidden), hidden);
	}

	for (int i = 0; i < n_output; i++) {
		activation(i, matrixAddMatrix(i, dot(i, W_out, hidden, output, n_hidden, n_output, 1), b_out, output), output);
	}
}

struct MutationNN {
	int device;
	double *input;
	double *output;
	double *hidden;
	double *W_in;
	double *W_out;
	double *b_in;
	double *b_out;

	int n_input;
	int n_hidden;
	int n_output;

	MutationNN(int n_in, int n_out) {
		n_input = n_in;
		n_hidden = n_out;
		n_output = n_out;
	}

	void memory_allocate(double *W_x, double *b_x, double *W_y, double *b_y, int dev) {
		device = dev;

		CudaSafeCall(cudaMallocManaged((void**)&input, n_input*sizeof(double)));
		memset(input, 0.0f, n_input*sizeof(double));
		CudaSafeCall(cudaMemPrefetchAsync(input, n_input*sizeof(double), device, NULL));
		CudaSafeCall(cudaMallocManaged((void**)&output, n_output*sizeof(double)));
		memset(output, 0.0f, n_output*sizeof(double));
		CudaSafeCall(cudaMemPrefetchAsync(output, n_output*sizeof(double), device, NULL));
		CudaSafeCall(cudaMallocManaged((void**)&W_in, n_hidden*n_input*sizeof(double)));
		CudaSafeCall(cudaMallocManaged((void**)&b_in, n_hidden*sizeof(double)));
		CudaSafeCall(cudaMallocManaged((void**)&hidden, n_hidden*sizeof(double)));
		memset(hidden, 0.0f, n_hidden*sizeof(double));
		CudaSafeCall(cudaMemPrefetchAsync(hidden, n_hidden*sizeof(double), device, NULL));
		CudaSafeCall(cudaMallocManaged((void**)&W_out, n_hidden*n_output*sizeof(double)));
		CudaSafeCall(cudaMallocManaged((void**)&b_out, n_output*sizeof(double)));

		memcpy(W_in, W_x, n_hidden*n_input*sizeof(double));
		CudaSafeCall(cudaMemPrefetchAsync(W_in, n_hidden*n_input*sizeof(double), device, NULL));
		memcpy(b_in, b_x, n_hidden*sizeof(double));
		CudaSafeCall(cudaMemPrefetchAsync(b_in, n_hidden*sizeof(double), device, NULL));
		memcpy(W_out, W_y, n_hidden*n_output*sizeof(double));
		CudaSafeCall(cudaMemPrefetchAsync(W_out, n_hidden*n_output*sizeof(double), device, NULL));
		memcpy(b_out, b_y, n_output*sizeof(double));
		CudaSafeCall(cudaMemPrefetchAsync(b_out, n_output*sizeof(double), device, NULL));
	}

	void free_resources(void) {
		CudaSafeCall(cudaDeviceSynchronize());
		CudaSafeCall(cudaFree(input));
		CudaSafeCall(cudaFree(output));
		CudaSafeCall(cudaFree(W_in));
		CudaSafeCall(cudaFree(b_in));
		CudaSafeCall(cudaFree(hidden));
		CudaSafeCall(cudaFree(W_out));
		CudaSafeCall(cudaFree(b_out));
	}

	__device__ void evaluate(void) {
		feedforward(input, W_in, b_in, hidden, W_out, b_out, output, n_input, n_hidden, n_output);
	}

	__device__ void mutate(int M, double *mutations, int cell, curandState_t *states) {
		if (M != 0) {
			if (curand_uniform_double(&states[cell]) <= CHANCE_UPREG) {
				double incr = (curand_uniform_double(&states[cell]) * (RAND_INCR_B - RAND_INCR_A + 0.999999999f) + RAND_INCR_A) / 1000000000.0f;
				W_out[M*n_output+M] += incr;
				mutations[M] += incr;
				W_out[0] = fmaxf(0.0f, W_out[0] - incr);
			} else {
				double decr = (curand_uniform_double(&states[cell]) * (RAND_DECR_B - RAND_DECR_A + 0.999999999f) + RAND_DECR_A) / 1000000000.0f;
				W_out[M*n_output+M] = fmaxf(0.0f, W_out[M*n_output+M] - decr);
				mutations[M] = fmaxf(0.0f, mutations[M] - decr);
				W_out[0] += decr;
			}
		}
		for (int i = 0; i < index_map[M*12]; i++) {
			if (curand_uniform_double(&states[cell]) <= CHANCE_UPREG) {
				double incr = (curand_uniform_double(&states[cell]) * (RAND_INCR_B - RAND_INCR_A + 0.999999999f) + RAND_INCR_A) / 1000000000.0f;
				W_out[index_map[M*12+(i+1)]*n_output+index_map[M*12+(i+1)]] += incr;
				W_out[index_map[M*12+(i+1)]*n_output] -= incr;
			} else {
				double decr = (curand_uniform(&states[cell]) * (RAND_DECR_B - RAND_DECR_A + 0.999999999f) + RAND_DECR_A) / 1000000000.0f;
				W_out[index_map[M*12+(i+1)]*n_output+index_map[M*12+(i+1)]] = fmaxf(0.0f, W_out[index_map[M*12+(i+1)]*n_output+index_map[M*12+(i+1)]] - decr);
				W_out[index_map[M*12+(i+1)]*n_output] = fminf(W_out[index_map[M*12+(i+1)]*n_output]+decr, 0.0f);
			}
			if (index_map[(i+1)*12+(i+1)] == M) {
				if (curand_uniform_double(&states[cell]) <= CHANCE_UPREG) {
					double incr = (curand_uniform_double(&states[cell]) * (RAND_INCR_B - RAND_INCR_A + 0.999999999f) + RAND_INCR_A) / 1000000000.0f;
					W_out[index_map[M*12+(i+1)]*n_output+M] += incr;
				} else {
					double decr = (curand_uniform_double(&states[cell]) * (RAND_DECR_B - RAND_DECR_A + 0.999999999f) + RAND_DECR_A) / 1000000000.0f;
					W_out[index_map[M*12+(i+1)]*n_output+M] = fmaxf(0, W_out[index_map[M*12+(i+1)]*n_output+M] - decr);
				}
			}
		}
	}
};

#endif // __MUTATION_NN_H__

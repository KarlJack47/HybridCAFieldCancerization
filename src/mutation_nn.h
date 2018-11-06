#ifndef __MUTATION_NN_H__
#define __MUTATION_NN_H__

__device__ void activation(int idx, float *input, float *output) {
	/*  Computes the value of the sigmoid function f(x) = 1/(1 + e^-2x).
            Inputs:
            input: array
            output: array, the results of the computation are to be stored here
	*/

	output[idx] = 1.0 / (1.0 + std::exp(-2*input[idx]));
}

__device__ float* dot(int idx, float *m1, float *m2, float *output, int m1_rows , int m1_columns, int m2_columns) {
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
		float t_output = 0.0f;
		for(int k = 0; k < m1_columns; k++) {
			t_output += m1[r*m1_columns+k] * m2[k*m2_columns+c];
		}
		output[i] = t_output;
	}

	return output;
}

__device__ float* matrixAddMatrix(int idx, float *m1, float *m2, float *output) {
	/* Computes the (elementwise) addition between two arrays
	   Inputs:
	   m1: array
	   m2: array
	   output: array, the results of the computation are to be stored here
	*/

	output[idx] = m1[idx] + m2[idx];

	return output;
}

__device__ void feedforward(float *input, float *W_in, float *b_in, float *hidden, float *W_out, float *b_out, float *output,
			    int n_input, int n_hidden, int n_output) {

	for (int i = 0; i < n_hidden; i++) {
		activation(i, matrixAddMatrix(i, dot(i, W_in, input, hidden, n_hidden, n_input, 1), b_in, hidden), hidden);
	}

	for (int i = 0; i < n_output; i++) {
		activation(i, matrixAddMatrix(i, dot(i, W_out, hidden, output, n_hidden, n_output, 1), b_out, output), output);
	}
}

struct MutationNN {
	float *input;
	float *output;
	float *hidden;
	float *W_in;
	float *W_out;
	float *b_in;
	float *b_out;

	int n_input;
	int n_hidden;
	int n_output;

	MutationNN(int n_in, int n_out) {
		n_input = n_in;
		n_hidden = n_out;
		n_output = n_out;
	}

	void memory_allocate(float *W_x, float *b_x, float *W_y, float *b_y) {
		CudaSafeCall(cudaMallocManaged((void**)&input, n_input*sizeof(float)));
		memset(input, 0.0f, n_input*sizeof(float));
		CudaSafeCall(cudaMallocManaged((void**)&output, n_output*sizeof(float)));
		memset(output, 0.0f, n_output*sizeof(float));
		CudaSafeCall(cudaMallocManaged((void**)&W_in, n_hidden*n_input*sizeof(float)));
		CudaSafeCall(cudaMallocManaged((void**)&b_in, n_hidden*sizeof(float)));
		CudaSafeCall(cudaMallocManaged((void**)&hidden, n_hidden*sizeof(float)));
		memset(hidden, 0.0f, n_hidden*sizeof(float));
		CudaSafeCall(cudaMallocManaged((void**)&W_out, n_hidden*n_output*sizeof(float)));
		CudaSafeCall(cudaMallocManaged((void**)&b_out, n_output*sizeof(float)));

		memcpy(W_in, W_x, n_hidden*n_input*sizeof(float));
		memcpy(b_in, b_x, n_hidden*sizeof(float));
		memcpy(W_out, W_y, n_hidden*n_output*sizeof(float));
		memcpy(b_out, b_y, n_output*sizeof(float));
	}

	void free_resources(void) {
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

	__device__ void mutate(int M, int *idx, float *mutations, int cell, curandState_t *states) {
		if (M != 0) {
			if ((int) ceilf(curand_uniform(&states[cell])*2) % 2 == 1) {
				float incr = ((((int) ceilf(curand_uniform(&states[cell])*1000000)) % (10000 - 5000 + 1)) + 5000) / 1000000.0f;
				W_out[M*n_output+M] += incr;
				mutations[M] += incr;
				W_out[0] -= incr;
			} else {
				float decr = ((((int) ceilf(curand_uniform(&states[cell])*1000000)) % (100000 - 10000 + 1)) + 10000) / 1000000.0f;
				W_out[M*n_output+M] -= decr;
				mutations[M] -= decr;
				W_out[0] += decr;
			}
		}
		for (int i = 0; i < idx[M*12]; i++) {
			if ((int) ceilf(curand_uniform(&states[cell])*2) % 2 == 1) {
				float incr = ((((int) ceilf(curand_uniform(&states[cell])*1000000)) % (10000 - 5000 + 1)) + 5000) / 1000000.0f;
				W_out[idx[M*12+(i+1)]*n_output+idx[M*12+(i+1)]] += incr;
				mutations[i+1] += incr;
				W_out[idx[M*12+(i+1)]*n_output] -= incr;
			} else {
				float decr = ((((int) ceilf(curand_uniform(&states[cell])*1000000)) % (100000 - 10000 + 1)) + 10000) / 1000000.0f;
				W_out[idx[M*12+(i+1)]*n_output+idx[M*12+(i+1)]] -= decr;
				mutations[i+1] -= decr;
				W_out[idx[M*12+(i+1)]*n_output] += decr;
			}
			if (idx[(i+1)*12+(i+1)] == M) {
				if ((int) ceilf(curand_uniform(&states[cell])*2) % 2 == 1) {
					float incr = ((((int) ceilf(curand_uniform(&states[cell])*1000000)) % (10000 - 5000 + 1)) + 5000) / 1000000.0f;
					W_out[idx[M*12+(i+1)]*n_output+M] += incr;
					mutations[M] += incr;
					W_out[M*n_output] -= incr;
				} else {
					float decr = ((((int) ceilf(curand_uniform(&states[cell])*1000000)) % (100000 - 10000 + 1)) + 10000) / 1000000.0f;
					W_out[idx[M*12+(i+1)]*n_output+M] -= decr;
					mutations[M] -= decr;
					W_out[M*n_output] += decr;
				}
			}
		}
	}
};

#endif // __MUTATION_NN_H__
